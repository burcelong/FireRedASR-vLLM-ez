from typing import List, Optional, Dict, Union, Tuple, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from flash_attn import flash_attn_with_kvcache, flash_attn_varlen_func


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        sos_id: int,
        eos_id: int,
        pad_id: int,
        odim: int,
        n_layers: int,
        n_head: int,
        d_model: int,
        residual_dropout: float = 0.1,
        pe_maxlen: int = 5000,
        attn_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.INF = 1e10
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.attn_dtype = attn_dtype

        self.tgt_word_emb = nn.Embedding(odim, d_model, padding_idx=self.pad_id)
        self.positional_encoding = PositionalEncoding(d_model, max_len=pe_maxlen)
        self.dropout = nn.Dropout(residual_dropout)

        self.layer_stack = nn.ModuleList([DecoderLayer(d_model, n_head, residual_dropout) for _ in range(n_layers)])

        self.tgt_word_prj = nn.Linear(d_model, odim, bias=False)
        self.layer_norm_out = nn.LayerNorm(d_model)
        self.tgt_word_prj.weight = self.tgt_word_emb.weight
        self.scale = (d_model ** 0.5)

    @torch.inference_mode()
    def batch_beam_search(
        self,
        encoder_outputs: Tensor,
        src_masks: Tensor,
        beam_size: int = 1,
        nbest: int = 1,
        decode_max_len: int = 0,
        softmax_smoothing: float = 1.0,
        length_penalty: float = 0.0,
        eos_penalty: float = 1.0,
        on_finish: Optional[Callable[[int, Dict[str, Tensor]], None]] = None,  # 新增：早回回调
    ):
        """
        on_finish(utt_idx, hyp): 当“原始样本 utt_idx”的所有 beam 均结束时立刻回调，hyp={'yseq': Tensor, 'score': float}
        注意：仅解耦“返回”，计算仍按批同步步进（未做 paged-KV）。
        """
        if self.attn_dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError("attn_dtype必须为torch.float16或torch.bfloat16（FlashAttention要求）")

        device = encoder_outputs.device
        original_batch_size = encoder_outputs.size(0)
        B = beam_size
        effective_batch_size = original_batch_size * B
        Ti = encoder_outputs.size(1)
        maxlen = decode_max_len if decode_max_len > 0 else Ti

        # 变长 cross-attn 需要的 encoder 有效长度
        if src_masks.dtype != torch.bool:
            src_masks_bool = (src_masks != 0).to(torch.bool)
        else:
            src_masks_bool = src_masks
        enc_pad_mask = src_masks_bool.view(original_batch_size, Ti)
        enc_lens = enc_pad_mask.sum(dim=1).to(torch.int32)
        max_seqlen_k = int(enc_lens.max().item())

        # 预先为每层构建 unpad 的 K/V 以及 cu_seqlens_k
        cross_k_unpad_layers: List[Tensor] = []
        cross_v_unpad_layers: List[Tensor] = []
        cu_seqlens_k_layers: List[Tensor] = []
        for layer_idx, dec_layer in enumerate(self.layer_stack):
            k_proj = dec_layer.cross_attn.w_ks(encoder_outputs)
            v_proj = dec_layer.cross_attn.w_vs(encoder_outputs)

            k = k_proj.view(original_batch_size, Ti, self.n_head, self.d_k).transpose(1, 2).contiguous()
            v = v_proj.view(original_batch_size, Ti, self.n_head, self.d_k).transpose(1, 2).contiguous()

            k = k.to(self.attn_dtype)
            v = v.to(self.attn_dtype)

            k_unpad, v_unpad, cu_seqlens_k = self.unpad_kv(
                k=k,
                v=v,
                lens=enc_lens,
                original_batch_size=original_batch_size,
                beam_size=B
            )
            assert cu_seqlens_k.shape == (effective_batch_size + 1,), \
                f"层{layer_idx} cu_seqlens_k形状错误: 预期{(effective_batch_size + 1,)}, 实际{cu_seqlens_k.shape}"

            cross_k_unpad_layers.append(k_unpad)
            cross_v_unpad_layers.append(v_unpad)
            cu_seqlens_k_layers.append(cu_seqlens_k)

        # 解码初值
        ys = torch.full((effective_batch_size, 1), self.sos_id, dtype=torch.long, device=device)

        # 预分配自注意力 KV 缓存
        big_self_k = torch.empty(
            self.n_layers, effective_batch_size, maxlen, self.n_head, self.d_k,
            device=device, dtype=self.attn_dtype
        )
        big_self_v = torch.empty_like(big_self_k)
        big_self_seqlen = torch.zeros(self.n_layers, effective_batch_size, dtype=torch.int32, device=device)

        # beam 分数与完成标记（每步更新）
        start_scores = torch.full((B,), -self.INF, device=device, dtype=torch.float32)
        start_scores[0] = 0.0
        scores = start_scores.repeat(original_batch_size).view(effective_batch_size, 1)
        is_finished = torch.zeros_like(scores, dtype=torch.bool)

        # 追踪“当前 beam 属于哪个原始样本”，以及样本是否已回调
        utt_ids = torch.arange(original_batch_size, device=device).repeat_interleave(B)      # [effB]
        done_utt = torch.zeros(original_batch_size, dtype=torch.bool, device=device)        # [BATCH]

        rotary_cos, rotary_sin = self.get_rotary_encoding(maxlen, device, dtype=self.attn_dtype)

        for step in range(maxlen):
            # 变长 cross-attn 的 query cu_seqlens（每步 q 长度=1，批内全生存者一起做）
            cu_seqlens_q = torch.arange(0, effective_batch_size + 1, dtype=torch.int32, device=device)

            # 准备当前步输入
            full_tgt_emb = self.tgt_word_emb(ys) * self.scale
            full_pos_emb = self.positional_encoding(ys)
            tgt_emb = full_tgt_emb[:, -1:]
            pos_emb = full_pos_emb[:, -1:]
            dec_input = self.dropout(tgt_emb + pos_emb)

            # L 层堆叠（自注意 -> 交叉注意 -> FFN）
            for layer_idx, dec_layer in enumerate(self.layer_stack):
                cache_view = {
                    "self_k": big_self_k[layer_idx],
                    "self_v": big_self_v[layer_idx],
                    "self_seqlen": big_self_seqlen[layer_idx],
                }
                dec_input, new_k, new_v = dec_layer.self_attn_step(
                    x=dec_input,
                    k_cache=cache_view["self_k"],
                    v_cache=cache_view["self_v"],
                    cache_seqlens=cache_view["self_seqlen"],
                    rotary_cos=rotary_cos,
                    rotary_sin=rotary_sin
                )
                cache_view["self_seqlen"] += 1

                current_cu_seqlens_k = cu_seqlens_k_layers[layer_idx]
                dec_input = dec_layer.cross_attn_step(
                    x=dec_input,
                    k_unpad=cross_k_unpad_layers[layer_idx],
                    v_unpad=cross_v_unpad_layers[layer_idx],
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=current_cu_seqlens_k,
                    max_seqlen_q=1,
                    max_seqlen_k=max_seqlen_k,
                    effective_batch_size=effective_batch_size
                )
                dec_input = dec_layer.ffn_step(dec_input)

            # 出 logits
            dec_output = self.layer_norm_out(dec_input)
            t_logit = self.tgt_word_prj(dec_output[:, 0])
            t_scores = F.log_softmax(t_logit / softmax_smoothing, dim=-1)
            if eos_penalty != 1.0:
                t_scores[:, self.eos_id] *= eos_penalty

            # 每条 beam 取 topB
            t_topB_scores, t_topB_ys = torch.topk(t_scores, k=B, dim=1)
            t_topB_scores = self.set_finished_beam_score_to_zero(t_topB_scores, is_finished)
            t_topB_ys = self.set_finished_beam_y_to_eos(t_topB_ys, is_finished)

            # 展开成 B*B 候选，再回到 B 条
            scores = scores + t_topB_scores  # [effB, 1] + [effB, B] -> 广播后 [effB, B]
            scores_reshaped = scores.view(original_batch_size, B * B)
            top_scores, top_indices = torch.topk(scores_reshaped, k=B, dim=1)
            scores = top_scores.view(-1, 1)

            # 选中保留的 beam，并相应重排缓存/序列/映射
            batch_indices = torch.arange(original_batch_size, device=device).view(original_batch_size, 1).repeat(1, B).view(-1)
            beam_indices = (top_indices // B).view(-1)
            selected_indices = batch_indices * B + beam_indices

            ys = ys.index_select(0, selected_indices)
            utt_ids = utt_ids.index_select(0, selected_indices)  # 同步原始样本映射

            new_tokens = torch.gather(t_topB_ys.view(original_batch_size, B * B), 1, top_indices).view(-1, 1)
            ys = torch.cat([ys, new_tokens], dim=1)

            big_self_k = big_self_k.index_select(1, selected_indices)
            big_self_v = big_self_v.index_select(1, selected_indices)
            big_self_seqlen = big_self_seqlen.index_select(1, selected_indices)

            # 更新完成标记：本步新 token 是否是 EOS（历史已完成的 beam 会持续为 True）
            is_finished = (new_tokens == self.eos_id).view(-1, 1)

            # ====== 早回：当某个原始样本的 B 条 beam 全部完成时，立刻回调 ======
            if on_finish is not None:
                # 将 is_finished 以原始样本为粒度聚合（当前排列下自然是 batch-major，每个样本聚合 B 条）
                beam_finished = is_finished.view(original_batch_size, B)
                utt_done = beam_finished.all(dim=1)  # [BATCH]
                newly_done = utt_done & (~done_utt)
                if newly_done.any():
                    # 取“当前步”的打分作为排名依据；若有 length_penalty，保持与最终口径一致
                    cur_scores = scores.view(original_batch_size, B)  # 正向 log 概率（越大越好）
                    if length_penalty > 0.0:
                        cur_ys = ys.view(original_batch_size, B, -1)
                        cur_lens = torch.sum(cur_ys.ne(self.eos_id), dim=-1).float()  # 不含 <eos> 后缀
                        cur_penalty = torch.pow((5 + cur_lens) / 6.0, length_penalty)
                        cur_rank_scores = cur_scores / cur_penalty
                    else:
                        cur_rank_scores = cur_scores

                    for bi in newly_done.nonzero(as_tuple=False).view(-1).tolist():
                        # 选出该样本当前最佳 beam
                        best_beam = int(torch.argmax(cur_rank_scores[bi]).item())
                        # 取 yseq（去掉 <s>；截到首个 <eos> 之前）
                        yseq_all = ys.view(original_batch_size, B, -1)[bi, best_beam]
                        ylen = int(torch.sum(yseq_all.ne(self.eos_id)).item())
                        yseq = yseq_all[1:ylen]  # 去 <s>
                        # 输出分数：与最终口径一致（最终会取 -score，这里直接返回正向“负损”）
                        best_score = float((-cur_rank_scores[bi, best_beam]).item())

                        on_finish(bi, {"yseq": yseq, "score": best_score})
                    done_utt[newly_done] = True

            # 如果整批都完成则退出
            if is_finished.all():
                break

        # ====== 常规收尾（兼容旧行为）：返回每个样本的 nbest（可能有些样本已通过回调提前返回） ======
        scores = scores.view(original_batch_size, B)
        ys = ys.view(original_batch_size, B, -1)
        ys_lengths = self.get_ys_lengths(ys)

        if length_penalty > 0.0:
            penalty = torch.pow((5 + ys_lengths.float()) / 6.0, length_penalty)
            scores = scores / penalty

        nbest = min(nbest, B)
        nbest_scores, nbest_indices = torch.topk(scores, k=nbest, dim=1)
        nbest_scores = -nbest_scores  # 维持原输出口径：越小越好

        batch_stride = B * torch.arange(original_batch_size, device=device).view(original_batch_size, 1)
        selected_ys_indices = nbest_indices + batch_stride
        nbest_ys = ys.view(effective_batch_size, -1)[selected_ys_indices.view(-1)].view(original_batch_size, nbest, -1)
        nbest_ys_lengths = ys_lengths.view(effective_batch_size)[selected_ys_indices.view(-1)].view(original_batch_size, nbest)

        nbest_hyps: List[List[Dict[str, Tensor]]] = []
        for batch_idx in range(original_batch_size):
            batch_hyps = []
            for n in range(nbest):
                yseq = nbest_ys[batch_idx, n, 1:nbest_ys_lengths[batch_idx, n]]
                score = float(nbest_scores[batch_idx, n].item())
                batch_hyps.append({"yseq": yseq, "score": score})
            nbest_hyps.append(batch_hyps)
        return nbest_hyps

    def unpad_kv(
        self,
        k: Tensor,
        v: Tensor,
        lens: Tensor,
        original_batch_size: int,
        beam_size: int
    ) -> Tuple[Tensor, Tensor, Tensor]:
        effective_batch_size = original_batch_size * beam_size
        h, Ti, d_k = k.shape[1], k.shape[2], k.shape[3]
        device = k.device

        k_expanded = k.repeat_interleave(beam_size, dim=0)
        v_expanded = v.repeat_interleave(beam_size, dim=0)
        lens_expanded = lens.repeat_interleave(beam_size, dim=0)
        total_k = int(lens_expanded.sum().item())

        cu_seqlens_k = torch.zeros(effective_batch_size + 1, dtype=torch.int32, device=device)
        k_unpad = torch.empty((total_k, h, d_k), dtype=k.dtype, device=device)
        v_unpad = torch.empty_like(k_unpad)

        ptr = 0
        for i in range(effective_batch_size):
            current_len = int(lens_expanded[i].item())
            if current_len <= 0:
                cu_seqlens_k[i + 1] = ptr
                continue
            end = ptr + current_len
            k_unpad[ptr:end] = k_expanded[i, :, :current_len, :].transpose(0, 1)
            v_unpad[ptr:end] = v_expanded[i, :, :current_len, :].transpose(0, 1)
            cu_seqlens_k[i + 1] = end
            ptr = end

        return k_unpad, v_unpad, cu_seqlens_k

    def get_rotary_encoding(self, max_seq_len: int, device: torch.device, dtype: Optional[torch.dtype] = None):
        if dtype is None:
            dtype = self.attn_dtype
        rotary_dim = self.d_k
        assert rotary_dim % 2 == 0, "旋转维度必须为偶数"
        pos = torch.arange(0, max_seq_len, device=device, dtype=dtype)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, rotary_dim, 2, device=device, dtype=dtype) / rotary_dim))
        sinusoid = torch.einsum("i,j->ij", pos, inv_freq)
        return torch.cos(sinusoid), torch.sin(sinusoid)

    def set_finished_beam_score_to_zero(self, scores: Tensor, is_finished: Tensor) -> Tensor:
        mask_row = torch.empty(scores.size(1), device=scores.device, dtype=scores.dtype)
        mask_row[0] = 0.0
        if scores.size(1) > 1:
            mask_row[1:] = -self.INF
        return scores * (~is_finished).float() + mask_row.view(1, -1) * is_finished.float()

    def set_finished_beam_y_to_eos(self, ys: Tensor, is_finished: Tensor) -> Tensor:
        return ys * (~is_finished).long() + self.eos_id * is_finished.long()

    def get_ys_lengths(self, ys: Tensor) -> Tensor:
        return torch.sum(torch.ne(ys, self.eos_id), dim=-1).int()


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.self_attn = DecoderMultiHeadAttention(d_model, n_head, dropout, causal=True)
        self.cross_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn = DecoderMultiHeadAttention(d_model, n_head, dropout, causal=False)
        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = PositionwiseFeedForward(d_model, d_model * 4, dropout)

    def self_attn_step(
        self,
        x: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
        cache_seqlens: Tensor,
        rotary_cos: Tensor,
        rotary_sin: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        residual = x
        x_norm = self.self_attn_norm(x)
        x_self, new_k, new_v = self.self_attn(
            q=x_norm, k=x_norm, v=x_norm,
            k_cache=k_cache, v_cache=v_cache, cache_seqlens=cache_seqlens,
            rotary_cos=rotary_cos, rotary_sin=rotary_sin
        )
        return residual + x_self, new_k, new_v

    def cross_attn_step(
        self,
        x: Tensor,
        k_unpad: Tensor,
        v_unpad: Tensor,
        cu_seqlens_q: Tensor,
        cu_seqlens_k: Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        effective_batch_size: int
    ) -> Tensor:
        assert cu_seqlens_k.shape == (effective_batch_size + 1,), \
            f"cross_attn_step cu_seqlens_k形状错误: 预期{(effective_batch_size + 1,)}, 实际{cu_seqlens_k.shape}"

        residual = x
        x_norm = self.cross_attn_norm(x)

        x_cross = self.cross_attn.varlen_cross_attention(
            q=x_norm,
            k_unpad=k_unpad,
            v_unpad=v_unpad,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            effective_batch_size=effective_batch_size
        )

        return residual + x_cross

    def ffn_step(self, x: Tensor) -> Tensor:
        residual = x
        x = self.mlp_norm(x)
        return residual + self.mlp(x)


class DecoderMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1, causal: bool = False):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.causal = causal
        self.softmax_scale = 1.0 / (self.d_k ** 0.5)
        self.dropout_p = dropout

        self.w_qs = nn.Linear(d_model, n_head * self.d_k)
        self.w_ks = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * self.d_k)
        self.fc = nn.Linear(n_head * self.d_k, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor] = None,
        k_cache: Optional[Tensor] = None,
        v_cache: Optional[Tensor] = None,
        cache_seqlens: Optional[Tensor] = None,
        rotary_cos: Optional[Tensor] = None,
        rotary_sin: Optional[Tensor] = None,
        need_kv_return: bool = False
    ):
        bs = q.size(0)
        seqlen_q = q.size(1)

        if self.causal:
            if k_cache is None:
                raise RuntimeError("自注意力需要k_cache")
            attn_dtype = k_cache.dtype

            q = self.w_qs(q).to(attn_dtype).view(bs, seqlen_q, self.n_head, self.d_k)
            k = self.w_ks(k).to(attn_dtype).view(bs, seqlen_q, self.n_head, self.d_k)
            v = self.w_vs(v).to(attn_dtype).view(bs, seqlen_q, self.n_head, self.d_k)

            out = flash_attn_with_kvcache(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                k=k,
                v=v,
                rotary_cos=rotary_cos,
                rotary_sin=rotary_sin,
                cache_seqlens=cache_seqlens,
                softmax_scale=self.softmax_scale,
                causal=True,
                rotary_interleaved=True
            )
            out = out.to(self.fc.weight.dtype).view(bs, seqlen_q, self.d_model)
            out = self.fc(out)
            return self.dropout(out), k, v

        else:
            q_proj = self.w_qs(q)
            k_proj = self.w_ks(k)
            v_proj = self.w_vs(v)

            seqlen_k = k.size(1)
            attn_dtype = k_proj.dtype
            q = q_proj.view(bs, seqlen_q, self.n_head, self.d_k).to(attn_dtype)
            k = k_proj.view(bs, seqlen_k, self.n_head, self.d_k).to(attn_dtype)
            v = v_proj.view(bs, seqlen_k, self.n_head, self.d_k).to(attn_dtype)

            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            attn = torch.matmul(q, k.transpose(2, 3)) * self.softmax_scale
            if mask is not None:
                attn = attn.masked_fill(mask.eq(0), torch.finfo(attn.dtype).min)
            attn = F.softmax(attn, dim=-1)
            output = torch.matmul(attn, v)

            output = output.transpose(1, 2).contiguous().view(bs, seqlen_q, self.d_model)
            output = output.to(self.fc.weight.dtype)
            output = self.fc(output)
            output = self.dropout(output)

            if need_kv_return:
                return output, k_proj, v_proj
            else:
                return output

    def varlen_cross_attention(
        self,
        q: Tensor,
        k_unpad: Tensor,
        v_unpad: Tensor,
        cu_seqlens_q: Tensor,
        cu_seqlens_k: Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        effective_batch_size: int
    ) -> Tensor:
        assert cu_seqlens_k.shape == (effective_batch_size + 1,), \
            f"varlen_cross_attention cu_seqlens_k形状错误: 预期{(effective_batch_size + 1,)}, 实际{cu_seqlens_k.shape}"
        assert cu_seqlens_k.dtype == torch.int32, \
            f"cu_seqlens_k类型错误: 预期int32, 实际{cu_seqlens_k.dtype}"

        bs = q.size(0)
        seqlen_q = q.size(1)

        q = self.w_qs(q).view(bs, seqlen_q, self.n_head, self.d_k).to(k_unpad.dtype)
        q = q.reshape(-1, self.n_head, self.d_k)

        out = flash_attn_varlen_func(
            q=q,
            k=k_unpad,
            v=v_unpad,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=self.dropout_p if self.training else 0.0,
            softmax_scale=self.softmax_scale,
            causal=False,
            deterministic=not self.training
        )

        out = out.view(bs, seqlen_q, self.n_head * self.d_k)
        out = out.to(self.fc.weight.dtype)
        out = self.fc(out)
        out = self.dropout(out)

        return out


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.w_2(self.act(self.w_1(x))))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        assert d_model % 2 == 0, "位置编码维度必须为偶数"
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        return self.pe[:, : x.size(1)]