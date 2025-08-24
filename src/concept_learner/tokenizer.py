from dataclasses import dataclass
from typing import List, Optional

try:
    from transformers import AutoTokenizer  # type: ignore
except Exception:  # transformers may be unavailable in CI
    AutoTokenizer = None  # type: ignore


@dataclass
class EncodeOutput:
    ids: List[int]
    mask: List[int]


class _SimpleTokenizer:
    """A tiny offline tokenizer used as a fallback when `transformers` is
    unavailable or a pretrained tokenizer cannot be downloaded.

    Behavior:
      - whitespace tokenization
      - adds [CLS] at start and [SEP] at end
      - PAD=0, CLS=1, SEP=2, UNK=3
      - grows vocab per-instance on the fly
    """

    def __init__(self):
        self.pad_token_id = 0
        self.cls_token_id = 1
        self.sep_token_id = 2
        self.unk_token_id = 3
        self._id_to_tok = {
            self.pad_token_id: "[PAD]",
            self.cls_token_id: "[CLS]",
            self.sep_token_id: "[SEP]",
            self.unk_token_id: "[UNK]",
        }
        self._tok_to_id = {v: k for k, v in self._id_to_tok.items()}

    @property
    def vocab_size(self) -> int:
        return len(self._id_to_tok)

    def _tok2id(self, tok: str) -> int:
        if tok in self._tok_to_id:
            return self._tok_to_id[tok]
        # grow vocab
        idx = len(self._id_to_tok)
        self._id_to_tok[idx] = tok
        self._tok_to_id[tok] = idx
        return idx

    def __call__(
        self,
        text: str,
        max_length: int = 64,
        padding: str = "max_length",
        truncation: bool = True,
        return_attention_mask: bool = True,
    ):
        # simple whitespace split; keep case to preserve test content
        toks = text.strip().split()
        ids = [self.cls_token_id] + [self._tok2id(t) for t in toks] + [
            self.sep_token_id
        ]
        if truncation and len(ids) > max_length:
            ids = ids[:max_length]
        if padding == "max_length" and len(ids) < max_length:
            ids = ids + [self.pad_token_id] * (max_length - len(ids))
        attn = [1 if i != self.pad_token_id else 0 for i in ids]
        return {"input_ids": ids, "attention_mask": attn}

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        toks: List[str] = []
        for i in ids:
            tok = self._id_to_tok.get(i, "[UNK]")
            if skip_special_tokens and tok in {"[PAD]", "[CLS]", "[SEP]"}:
                continue
            toks.append(tok)
        return " ".join(toks)


class HFTokenizerWrapper:
    def __init__(self, name: str = "bert-base-cased"):
        tok: Optional[object] = None
        # try huggingface tokenizer if available
        if AutoTokenizer is not None:
            try:
                tok = AutoTokenizer.from_pretrained(name)
            except Exception:
                tok = None
        # fallback to simple offline tokenizer
        if tok is None:
            tok = _SimpleTokenizer()
        self.tok = tok
        # expose IDs for special tokens (with robust defaults)
        self.pad_id = getattr(self.tok, "pad_token_id", 0)
        self.cls_id = getattr(self.tok, "cls_token_id", 1)
        self.sep_id = getattr(self.tok, "sep_token_id", 2)
        self.unk_id = getattr(self.tok, "unk_token_id", 3)
        self.vocab_size = getattr(self.tok, "vocab_size", None) or 30522

    def encode(self, text: str, max_len: int = 64) -> EncodeOutput:
        enc = self.tok(
            text,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )
        return EncodeOutput(ids=enc["input_ids"], mask=enc["attention_mask"])

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        return self.tok.decode(ids, skip_special_tokens=skip_special_tokens)
