from __future__ import annotations

import random
from typing import Any, Dict, List, Optional
import json
import os

try:
    from openai import OpenAI  # OpenAI Python SDK v1+
except Exception:  # pragma: no cover - optional runtime dep
    OpenAI = None  # type: ignore


class LLMTeacher:
    """
    LLM-powered episode teacher.

    This implementation provides:
    - Deterministic prompt builders for the generator and critic/validator as
      described in the curriculum document.
    - Lightweight, programmatic validators for common relation schemas
      (numbers, taxonomies, spatial) to catch basic errors before training.
    - A stub gen_batch method that shows how the client would be called, while
      still returning validated, synthetic placeholders if an LLM client is
      not wired in yet.
    """

    def __init__(self, client: Any | None, seed: int = 0, model: str = "gpt-4o-mini"):
        # If client is None, try to construct from OPENAI_API_KEY
        if client is None and OpenAI is not None and os.environ.get("OPENAI_API_KEY"):
            client = OpenAI()
        self.client = client
        self.model = model
        random.seed(seed)

    # ---------- Prompt builders ----------
    def build_generator_prompt(self, phase: str, rel_set: list[str], seeds: list[str]) -> str:
        return (
            "You are creating kindergarten-level knowledge for "
            f"{phase}.\nKeep concepts concrete and culture-neutral. "
            f"Use allowed relations only: {rel_set}.\n\n"
            "Produce STRICT JSON with:\n"
            "- concepts (nouns, number words/digits as needed),\n"
            "- relations (subset of allowed),\n"
            "- triples (20–60),\n"
            "- analogies (8–16) that reflect the current phase transformations,\n"
            "- equivalences (e.g., synonyms, number word ↔ digit),\n"
            "- negatives (15–25%) that are plausible but wrong.\n\n"
            "Ensure cross-context overlap using these shared seeds: "
            f"{seeds}.\nOutput JSON only."
        )

    def build_validator_prompt(self) -> str:
        return (
            "Critic/validator: Check that JSON uses only allowed relations, "
            "has no trivial cycles (e.g., A is_a B and B is_a A), and that basic "
            "arithmetic relations are consistent (successor/predecessor, make-ten). "
            "Return a JSON report with {ok: bool, errors: [..], fixes:[..]}"
        )

    # ---------- OpenAI call helpers ----------
    def _chat_json(self, prompt: str) -> Optional[Dict[str, Any]]:
        if self.client is None:
            return None
        try:
            # Prefer JSON via system instruction and response_format where supported
            resp = self.client.chat.completions.create(  # type: ignore[attr-defined]
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a data generator. Always return strict JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
            text = resp.choices[0].message.content  # type: ignore[index]
            return json.loads(text) if text else None
        except Exception:
            return None

    # ---------- Programmatic validators ----------
    def validate_numbers(self, payload: Dict[str, Any]) -> tuple[bool, List[str]]:
        errors: List[str] = []
        triples: List[list] = payload.get("triples", [])
        rel_ok = set(payload.get("relations", []))

        def find_triples(s: Any, r: str, o: Any) -> List[list]:
            out = []
            for t in triples:
                if len(t) == 3 and (s == "*" or t[0] == s) and t[1] == r and (o == "*" or t[2] == o):
                    out.append(t)
            return out

        # Successor/predecessor consistency
        for t in triples:
            if len(t) != 3:
                errors.append(f"bad triple format: {t}")
                continue
            s, r, o = t
            if r not in rel_ok:
                errors.append(f"relation not allowed: {r}")
            if r == "successor_of":
                try:
                    si, oi = int(s), int(o)
                except Exception:
                    continue
                if si != oi + 1:
                    errors.append(f"succ: {s}!={o}+1")
                # Check existence of predecessor pair (optional, not fatal)
                if not find_triples(o, "predecessor_of", s):
                    pass
            if r == "predecessor_of":
                try:
                    si, oi = int(s), int(o)
                except Exception:
                    continue
                if si + 1 != oi:
                    errors.append(f"pred: {s}+1!={o}")

            if r == "makes_ten_with":
                try:
                    si, oi = int(s), int(o)
                except Exception:
                    continue
                if si + oi != 10:
                    errors.append(f"make10: {s}+{o}!=10")

            if r in ("has_tens", "has_ones"):
                try:
                    si, oi = int(s), int(o)
                except Exception:
                    continue
                tens, ones = si // 10, si % 10
                if r == "has_tens" and oi != tens:
                    errors.append(f"tens: {s} tens is {tens} not {o}")
                if r == "has_ones" and oi != ones:
                    errors.append(f"ones: {s} ones is {ones} not {o}")

        return (len(errors) == 0), errors

    def validate_taxonomy(self, payload: Dict[str, Any]) -> tuple[bool, List[str]]:
        errors: List[str] = []
        triples: List[list] = payload.get("triples", [])
        parents: dict[str, str] = {}
        for s, r, o in (t for t in triples if len(t) == 3):
            if r == "is_a":
                if s == o:
                    errors.append(f"self-loop in is_a: {s}")
                if parents.get(o, None) == s:
                    errors.append(f"2-cycle in is_a: {s}<->{o}")
                parents[s] = o
        return (len(errors) == 0), errors

    # ---------- Batch API ----------
    def gen_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """
        Return a list of episode dicts with schema:
        { "domain": "numbers|words|rhythm|shapes",
          "type": "analogy|triple|posneg|views",
          "payload": {...}, "answers": {...} }

        If an LLM client is configured, we construct prompts per phase and
        post-validate the results. Otherwise we return a tiny, fully
        validated synthetic set that follows the same schema (useful for
        plumbing and unit tests).
        """
        episodes: List[Dict[str, Any]] = []

        if self.client is None:
            # Provide a tiny numbers payload aligned with the curriculum doc
            payload = {
                "concepts": [str(i) for i in range(0, 13)] + ["one", "two", "three"],
                "relations": [
                    "successor_of",
                    "predecessor_of",
                    "has_tens",
                    "has_ones",
                    "makes_ten_with",
                ],
                "triples": [
                    ["1", "successor_of", "0"],
                    ["2", "successor_of", "1"],
                    ["3", "predecessor_of", "4"],
                    ["10", "has_tens", "1"],
                    ["10", "has_ones", "0"],
                    ["7", "makes_ten_with", "3"],
                ],
                "analogies": [
                    [["2", "3"], ["5", "6"]],
                    [["2", "4"], ["3", "5"]],
                    [["7", "10"], ["6", "?"]],
                ],
                "equivalences": [["three", "3"]],
            }
            ok, errs = self.validate_numbers(payload)
            if ok:
                episodes.append({
                    "domain": "numbers",
                    "type": "kg",
                    "payload": payload,
                    "answers": {},
                })
            return episodes

        # Attempt to synthesize with the OpenAI SDK for Phase 4 (numbers) by default
        try:
            from concept_learner.data.curriculum import allowed_relations_for_phase, SEEDS

            rel_set = allowed_relations_for_phase(4)
            prompt = self.build_generator_prompt("Phase 4 — Numbers", rel_set, SEEDS)
            data = self._chat_json(prompt)
            if isinstance(data, dict):
                ok, errs = self.validate_numbers(data)
                if ok:
                    episodes.append({
                        "domain": "numbers",
                        "type": "kg",
                        "payload": data,
                        "answers": {},
                    })
        except Exception:
            pass
        return episodes
