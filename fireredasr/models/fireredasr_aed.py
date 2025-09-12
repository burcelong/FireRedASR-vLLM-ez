import torch

from fireredasr.models.module.conformer_encoder import ConformerEncoder
from fireredasr.models.module.varlen_decoder import TransformerDecoder
from typing import Optional, Callable, Dict


class FireRedAsrAed(torch.nn.Module):
    @classmethod
    def from_args(cls, args):
        return cls(args)

    def __init__(self, args):
        super().__init__()
        self.sos_id = args.sos_id
        self.eos_id = args.eos_id

        self.encoder = ConformerEncoder(
            args.idim, args.n_layers_enc, args.n_head, args.d_model,
            args.residual_dropout, args.dropout_rate,
            args.kernel_size, args.pe_maxlen)

        self.decoder = TransformerDecoder(
            args.sos_id, args.eos_id, args.pad_id, args.odim,
            args.n_layers_dec, args.n_head, args.d_model,
            args.residual_dropout, args.pe_maxlen)

    @torch.inference_mode()
    def transcribe(self, padded_input, input_lengths,
                   beam_size=1, nbest=1, decode_max_len=0,
                   softmax_smoothing=1.0, length_penalty=0.0, eos_penalty=1.0,
                   on_finish: Optional[Callable[[int, Dict], None]] = None):
        """
        若提供 on_finish(utt_idx, hyp)，则在该 batch 内“原始样本 utt_idx”完成时立刻回调；
        函数返回值仍保留完整 nbest（用于兼容旧路径或补齐未早回的样本）。
        """
        enc_outputs, _, enc_mask = self.encoder(padded_input, input_lengths)
        nbest_hyps = self.decoder.batch_beam_search(
            enc_outputs, enc_mask,
            beam_size, nbest, decode_max_len,
            softmax_smoothing, length_penalty, eos_penalty,
            on_finish=on_finish
        )
        return nbest_hyps

    @torch.inference_mode()
    def encoding(self, padded_input, input_lengths):
        """仅执行编码器部分，返回中间特征"""
        enc_outputs, _, enc_mask = self.encoder(padded_input, input_lengths)
        return enc_outputs, enc_mask, input_lengths

    @torch.inference_mode()
    def decoding(self, enc_outputs, enc_mask,
                 beam_size=1, nbest=1, decode_max_len=0,
                 softmax_smoothing=1.0, length_penalty=0.0, eos_penalty=1.0,
                 on_finish: Optional[Callable[[int, Dict], None]] = None):
        """
        解码器封装；同样支持 on_finish 早回。
        """
        nbest_hyps = self.decoder.batch_beam_search(
            enc_outputs, enc_mask,
            beam_size, nbest, decode_max_len,
            softmax_smoothing, length_penalty, eos_penalty,
            on_finish=on_finish
        )
        return nbest_hyps