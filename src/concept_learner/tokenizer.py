from dataclasses import dataclass
from typing import List
from transformers import AutoTokenizer


@dataclass
class EncodeOutput:
    ids: List[int]
    mask: List[int]


class HFTokenizerWrapper:
    def __init__(self, name: str = "bert-base-cased"):
        self.tok = AutoTokenizer.from_pretrained(name)
        # expose IDs for special tokens
        self.pad_id = self.tok.pad_token_id
        self.cls_id = getattr(self.tok, "cls_token_id", None)
        self.sep_id = getattr(self.tok, "sep_token_id", None)
        self.unk_id = self.tok.unk_token_id

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
