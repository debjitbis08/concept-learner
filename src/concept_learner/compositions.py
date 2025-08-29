from __future__ import annotations

import random
from typing import List, Tuple

import torch


def _ones(x: int) -> int:
    return x % 10


def _tens(x: int) -> int:
    return (x // 10) % 10


def _wrap(v: int, mod: int) -> int:
    if mod <= 0:
        return v
    return ((v % mod) + mod) % mod


def make_numeric_compositions(
    n_items: int, batch: int, device: str | torch.device = "cpu"
) -> Tuple[List[str], torch.Tensor]:
    """
    Generate numeric-target composition questions and their targets.

    Expressions include small compositions like:
      - succ(succ(a)), pred(pred(a)), succ(pred(a)), pred(succ(a))
      - succ(ones(a)), pred(ones(a))
      - succ(tens(a)), pred(tens(a))

    Returns:
      (texts, targets) where targets are in [0, n_items-1].
    """
    texts: List[str] = []
    y = torch.zeros(batch, dtype=torch.long, device=device)
    A = torch.randint(0, max(1, n_items), (batch,), device=device).tolist()

    for i, a in enumerate(A):
        kind = random.randrange(0, 8)
        if kind == 0:  # succ(succ(a))
            r = _wrap(a + 2, n_items)
            txt = random.choice(
                [
                    f"What is the next of next of {a}?",
                    f"What is succ(succ({a}))?",
                    f"What comes two after {a}?",
                ]
            )
        elif kind == 1:  # pred(pred(a))
            r = _wrap(a - 2, n_items)
            txt = random.choice(
                [
                    f"What is the previous of previous of {a}?",
                    f"What is pred(pred({a}))?",
                    f"What comes two before {a}?",
                ]
            )
        elif kind == 2:  # succ(pred(a))
            r = _wrap(a + (-1) + 1, n_items)
            txt = random.choice(
                [f"What is succ(pred({a}))?", f"What is next of previous of {a}?" ]
            )
        elif kind == 3:  # pred(succ(a))
            r = _wrap(a + 1 - 1, n_items)
            txt = random.choice(
                [f"What is pred(succ({a}))?", f"What is previous of next of {a}?" ]
            )
        elif kind == 4:  # succ(ones(a))
            base = _ones(a)
            r = _wrap(base + 1, 10)
            txt = random.choice(
                [
                    f"What is the successor of the ones digit of {a}?",
                    f"What is succ(ones({a}))?",
                    f"Next of ones digit of {a}?",
                ]
            )
        elif kind == 5:  # pred(ones(a))
            base = _ones(a)
            r = _wrap(base - 1, 10)
            txt = random.choice(
                [
                    f"What is the predecessor of the ones digit of {a}?",
                    f"What is pred(ones({a}))?",
                    f"Previous of ones digit of {a}?",
                ]
            )
        elif kind == 6:  # succ(tens(a))
            base = _tens(a)
            r = _wrap(base + 1, 10)
            txt = random.choice(
                [
                    f"What is the successor of the tens digit of {a}?",
                    f"What is succ(tens({a}))?",
                    f"Next of tens digit of {a}?",
                ]
            )
        else:  # pred(tens(a))
            base = _tens(a)
            r = _wrap(base - 1, 10)
            txt = random.choice(
                [
                    f"What is the predecessor of the tens digit of {a}?",
                    f"What is pred(tens({a}))?",
                    f"Previous of tens digit of {a}?",
                ]
            )
        # targets must be in [0, n_items-1]
        y[i] = int(r) if r < n_items else (r % n_items)
        texts.append(txt)
    return texts, y


def make_equality_compositions(
    n_items: int, batch: int, device: str | torch.device = "cpu"
) -> Tuple[List[str], torch.Tensor]:
    """
    Generate yes/no equality questions with small compositions.

    Produces half positives (true) and half negatives (false) with phrasing like:
      - "Is succ(succ(a)) == b?"
      - "Is the next of next of a equal to b?"
      - "Is the successor of the ones digit of a equal to r?"
    """
    texts: List[str] = []
    y = torch.zeros(batch, dtype=torch.long, device=device)
    A = torch.randint(0, max(1, n_items), (batch,), device=device).tolist()
    pos_mask = [False] * batch
    for i in range(batch // 2):
        pos_mask[i] = True
    random.shuffle(pos_mask)

    for i, a in enumerate(A):
        kind = random.randrange(0, 8)
        # compute the true result r_true according to kind
        if kind == 0:
            r_true = _wrap(a + 2, n_items)
            cand = [f"Is succ(succ({a})) == {{b}}?", f"Is the next of next of {a} equal to {{b}}?"]
        elif kind == 1:
            r_true = _wrap(a - 2, n_items)
            cand = [f"Is pred(pred({a})) == {{b}}?", f"Is the previous of previous of {a} equal to {{b}}?"]
        elif kind == 2:
            r_true = _wrap(a, n_items)  # succ(pred(a))
            cand = [f"Is succ(pred({a})) == {{b}}?", f"Is the next of previous of {a} equal to {{b}}?"]
        elif kind == 3:
            r_true = _wrap(a, n_items)  # pred(succ(a))
            cand = [f"Is pred(succ({a})) == {{b}}?", f"Is the previous of next of {a} equal to {{b}}?"]
        elif kind == 4:
            d = _ones(a)
            r_true = _wrap(d + 1, 10)
            cand = [
                f"Is the successor of the ones digit of {a} equal to {{b}}?",
                f"Is succ(ones({a})) == {{b}}?",
            ]
        elif kind == 5:
            d = _ones(a)
            r_true = _wrap(d - 1, 10)
            cand = [
                f"Is the predecessor of the ones digit of {a} equal to {{b}}?",
                f"Is pred(ones({a})) == {{b}}?",
            ]
        elif kind == 6:
            d = _tens(a)
            r_true = _wrap(d + 1, 10)
            cand = [
                f"Is the successor of the tens digit of {a} equal to {{b}}?",
                f"Is succ(tens({a})) == {{b}}?",
            ]
        else:
            d = _tens(a)
            r_true = _wrap(d - 1, 10)
            cand = [
                f"Is the predecessor of the tens digit of {a} equal to {{b}}?",
                f"Is pred(tens({a})) == {{b}}?",
            ]
        if pos_mask[i]:
            b = r_true
            y[i] = 1
        else:
            # choose a near-miss negative
            delta = random.choice([-1, 1, 2, -2])
            b = _wrap(r_true + delta, n_items if r_true >= 10 else 10)
            if b == r_true:
                b = _wrap(b + 1, n_items if r_true >= 10 else 10)
            y[i] = 0
        tmpl = random.choice(cand)
        texts.append(tmpl.format(b=b))
    return texts, y

