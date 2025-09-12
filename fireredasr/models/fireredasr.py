import os
import time
from typing import Any, Dict, List, Optional

import torch

from fireredasr.data.asr_feat import ASRFeatExtractor
from fireredasr.models.fireredasr_aed import FireRedAsrAed
from fireredasr.models.fireredasr_llm import FireRedAsrLlm
from fireredasr.tokenizer.aed_tokenizer import ChineseCharEnglishSpmTokenizer
from fireredasr.tokenizer.llm_tokenizer import LlmTokenizerWrapper


class FireRedAsr:
    @classmethod
    def from_pretrained(cls, asr_type, model_dir, device=None):
        assert asr_type in ["aed", "llm"]

        cmvn_path = os.path.join(model_dir, "cmvn.ark")
        feat_extractor = ASRFeatExtractor(cmvn_path)

        if asr_type == "aed":
            model_path = os.path.join(model_dir, "model.pth.tar")
            dict_path = os.path.join(model_dir, "dict.txt")
            spm_model = os.path.join(model_dir, "train_bpe1000.model")
            model = load_fireredasr_aed_model(model_path)
            tokenizer = ChineseCharEnglishSpmTokenizer(dict_path, spm_model)
        elif asr_type == "llm":
            model_path = os.path.join(model_dir, "model.pth.tar")
            encoder_path = os.path.join(model_dir, "asr_encoder.pth.tar")
            llm_dir = os.path.join(model_dir, "Qwen2-7B-Instruct")
            model, tokenizer = load_firered_llm_model_and_tokenizer(
                model_path, encoder_path, llm_dir
            )

        if device is not None:
            model = model.to(device)
        model.eval()
        return cls(asr_type, feat_extractor, model, tokenizer)

    def __init__(self, asr_type, feat_extractor, model, tokenizer):
        self.asr_type = asr_type
        self.feat_extractor = feat_extractor
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def transcribe(self, batch_uttid: List[str], batch_wav_path: List[str], args: Dict[str, Any] = {}):
        """
        - 保持原有返回格式不变（每个样本 -> nbest 列表）。
        - 若 args 包含 `on_finish` 回调：对 AED 路径启用“先完成先返回”，
          回调参数为 (utt_idx, result_dict)，其中 result_dict 已转为文本：
          {"uttid", "text", "score", "wav"}。
        """
        feats, lengths, durs = self.feat_extractor(batch_wav_path)
        total_dur = sum(durs)

        use_gpu = bool(args.get("use_gpu", False))
        if use_gpu:
            feats, lengths = feats.cuda(), lengths.cuda()
            self.model.cuda()
        else:
            self.model.cpu()

        # ---------- AED 路径 ----------
        if self.asr_type == "aed":
            beam_size = int(args.get("beam_size", 1))
            nbest = int(args.get("nbest", 1))
            decode_max_len = int(args.get("decode_max_len", 0))
            softmax_smoothing = float(args.get("softmax_smoothing", 1.0))
            length_penalty = float(args.get("aed_length_penalty", 0.0))
            eos_penalty = float(args.get("eos_penalty", 1.0))

            start_time = time.time()

            # 如果提供了 on_finish，则包一层把 yseq -> 文本，并附带 uttid/wav
            user_on_finish = args.get("on_finish", None)

            if user_on_finish is not None:
                def _wrap_on_finish(utt_idx: int, hyp: Dict[str, Any]):
                    # hyp 形如 {"yseq": Tensor/list, "score": float}
                    yseq = hyp.get("yseq", [])
                    if torch.is_tensor(yseq):
                        hyp_ids = [int(x) for x in yseq.tolist()]
                    else:
                        hyp_ids = [int(x) for x in yseq]
                    text = self.tokenizer.detokenize(hyp_ids)
                    result = {
                        "uttid": batch_uttid[utt_idx],
                        "text": text,
                        "score": float(hyp.get("score", 0.0)),
                        "wav": batch_wav_path[utt_idx],
                    }
                    user_on_finish(utt_idx, result)

                # 走 encoding + decoding（把回调传给解码器）
                enc_outputs, enc_mask, _ = self.model.encoding(feats, lengths)
                hyps = self.model.decoding(
                    enc_outputs, enc_mask,
                    beam_size=beam_size, nbest=nbest, decode_max_len=decode_max_len,
                    softmax_smoothing=softmax_smoothing, length_penalty=length_penalty, eos_penalty=eos_penalty,
                    on_finish=_wrap_on_finish
                )
            else:
                # 兼容旧行为：不提前返回，整批完成后再出
                hyps = self.model.transcribe(
                    feats, lengths,
                    beam_size, nbest, decode_max_len,
                    softmax_smoothing, length_penalty, eos_penalty
                )

            elapsed = time.time() - start_time
            rtf = elapsed / total_dur if total_dur > 0 else 0.0

            # 统一把最终 hyps -> 文本（保持老返回结构）
            results: List[List[Dict[str, Any]]] = []
            for uttid, wav, hyps_per_sample in zip(batch_uttid, batch_wav_path, hyps):
                sample_results = []
                for hyp in hyps_per_sample:
                    yseq = hyp["yseq"]
                    if torch.is_tensor(yseq):
                        hyp_ids = [int(x) for x in yseq.cpu().tolist()]
                    else:
                        hyp_ids = [int(x) for x in yseq]
                    text = self.tokenizer.detokenize(hyp_ids)
                    sample_results.append({
                        "uttid": uttid,
                        "text": text,
                        "score": float(hyp["score"]),
                        "wav": wav,
                        "rtf": f"{rtf:.4f}",
                    })
                results.append(sample_results)
            return results

        # ---------- LLM 路径（保持不变） ----------
        elif self.asr_type == "llm":
            input_ids, attention_mask, _, _ = LlmTokenizerWrapper.preprocess_texts(
                origin_texts=[""] * feats.size(0), tokenizer=self.tokenizer,
                max_len=128, decode=True
            )
            if use_gpu:
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()

            start_time = time.time()
            generated_ids = self.model.transcribe(
                feats, lengths, input_ids, attention_mask,
                int(args.get("beam_size", 1)),
                int(args.get("decode_max_len", 0)),
                int(args.get("decode_min_len", 0)),
                float(args.get("repetition_penalty", 1.0)),
                float(args.get("llm_length_penalty", 0.0)),
                float(args.get("temperature", 1.0))
            )
            elapsed = time.time() - start_time
            rtf = elapsed / total_dur if total_dur > 0 else 0.0

            texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            results = []
            for uttid, wav, text in zip(batch_uttid, batch_wav_path, texts):
                results.append({
                    "uttid": uttid,
                    "text": text,
                    "wav": wav,
                    "rtf": f"{rtf:.4f}"
                })
            return results


def load_fireredasr_aed_model(model_path):
    package = torch.load(model_path, map_location=lambda storage, loc: storage)
    print("model args:", package["args"])
    model = FireRedAsrAed.from_args(package["args"])
    model.load_state_dict(package["model_state_dict"], strict=True)
    return model


def load_firered_llm_model_and_tokenizer(model_path, encoder_path, llm_dir):
    package = torch.load(model_path, map_location=lambda storage, loc: storage)
    package["args"].encoder_path = encoder_path
    package["args"].llm_dir = llm_dir
    print("model args:", package["args"])
    model = FireRedAsrLlm.from_args(package["args"])
    model.load_state_dict(package["model_state_dict"], strict=False)
    tokenizer = LlmTokenizerWrapper.build_llm_tokenizer(llm_dir)
    return model, tokenizer