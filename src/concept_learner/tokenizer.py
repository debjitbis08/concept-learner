from dataclasses import dataclass, field
from typing import List, Optional

try:
    from transformers import AutoTokenizer  # type: ignore
except Exception:  # transformers may be unavailable in CI
    AutoTokenizer = None  # type: ignore


@dataclass
class EncodeOutput:
    ids: List[int]
    mask: List[int]
    # Optional per-token digit-position ids (e.g., ones=0, tens=1, ...). 0 for non-digits by default.
    digit_pos: Optional[List[int]] = field(default=None)


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
        text,
        max_length: int = 64,
        padding: str = "max_length",
        truncation: bool = True,
        return_attention_mask: bool = True,
        is_split_into_words: bool | None = None,
        add_special_tokens: bool = True,
    ):
        # accept either a raw string or a pretokenized list of strings
        if isinstance(text, list):
            toks = list(text)
        else:
            toks = text.strip().split()
        ids = []
        if add_special_tokens:
            ids.append(self.cls_token_id)
        ids.extend([self._tok2id(t) for t in toks])
        if add_special_tokens:
            ids.append(self.sep_token_id)
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
                self.backend = "hf"
                self.backend_name = name
            except Exception:
                tok = None
                self.backend = "simple"
                self.backend_name = "_simple_fallback"
        # fallback to simple offline tokenizer
        if tok is None:
            tok = _SimpleTokenizer()
            # set backend indicators if not set above
            if not hasattr(self, "backend"):
                self.backend = "simple"
                self.backend_name = "_simple_fallback"
        self.tok = tok
        # expose IDs for special tokens (with robust defaults)
        self.pad_id = getattr(self.tok, "pad_token_id", 0)
        self.cls_id = getattr(self.tok, "cls_token_id", 1)
        self.sep_id = getattr(self.tok, "sep_token_id", 2)
        self.unk_id = getattr(self.tok, "unk_token_id", 3)
        self.vocab_size = getattr(self.tok, "vocab_size", None) or 30522

    def _digit_aware_pretokenize(self, text: str) -> list[str]:
        # Split into tokens such that sequences of digits become per-digit tokens with sign preserved,
        # alphabetic sequences stay grouped, and punctuation becomes separate tokens.
        import string
        out: list[str] = []
        buf = ""
        mode = None  # 'alpha'
        def flush():
            nonlocal buf
            if buf:
                out.append(buf)
                buf = ""
        i = 0
        while i < len(text):
            ch = text[i]
            if ch.isdigit():
                flush()
                out.append(ch)
                mode = None
                i += 1
                continue
            # handle signed numbers: capture leading '-' or '+' before a digit
            if ch in "+-" and (i + 1) < len(text) and text[i + 1].isdigit():
                flush()
                out.append(ch)
                mode = None
                i += 1
                continue
            elif ch.isspace():
                flush()
                mode = None
                i += 1
                continue
            elif ch in string.punctuation:
                flush()
                out.append(ch)
                mode = None
                i += 1
                continue
            else:
                if mode != 'alpha':
                    flush()
                    buf = ch
                    mode = 'alpha'
                else:
                    buf += ch
                i += 1
        flush()
        return out

    def _digit_positions(self, toks: List[str]) -> List[int]:
        # Compute per-token digit place (0=ones,1=tens,...) for contiguous numeric spans,
        # non-digit tokens get 0 by default. We scan contiguous [sign][digits...] groups.
        pos = [0] * len(toks)
        i = 0
        while i < len(toks):
            # detect signed number span
            j = i
            saw_sign = False
            if toks[j] in ['+', '-']:
                saw_sign = True
                j += 1
            k = j
            while k < len(toks) and toks[k].isdigit():
                k += 1
            if k > j:  # we have digits from j..k-1
                width = k - j
                # assign positions from right to left
                for t in range(j, k):
                    # t from j..k-1, position index from rightmost
                    pos[t] = (k - 1 - t)
                # optional: mark sign token position as 0
                if saw_sign:
                    pos[i] = 0
                i = k
            else:
                i = i + 1
        return pos

    def encode(self, text: str, max_len: int = 64) -> EncodeOutput:
        toks = self._digit_aware_pretokenize(text)
        # Prefer HF backend with pretokenized input; fallback tokenizer also supports lists
        enc = self.tok(
            toks,
            is_split_into_words=True,
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )
        # compute digit positions matching the produced tokens (before padding)
        try:
            # account for special tokens we added ([CLS] at start, [SEP] at end) in our own pass
            pos_ids = [0] + self._digit_positions(toks) + [0]
            # pad/truncate to max_len
            if len(pos_ids) > max_len:
                pos_ids = pos_ids[:max_len]
            if len(pos_ids) < max_len:
                pos_ids = pos_ids + [0] * (max_len - len(pos_ids))
        except Exception:
            pos_ids = [0] * max_len
        return EncodeOutput(ids=enc["input_ids"], mask=enc["attention_mask"], digit_pos=pos_ids)

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        return self.tok.decode(ids, skip_special_tokens=skip_special_tokens)
