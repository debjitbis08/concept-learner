from __future__ import annotations

import random
from typing import Any, Dict, List, Optional
import json
import os
import threading
import queue
import time

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

    def __init__(self, client: Any | None, seed: int = 0, model: str = "gpt-4o-mini", request_timeout_s: float = 6.0):
        # If client is None, try to construct from OPENAI_API_KEY
        if client is None and OpenAI is not None and os.environ.get("OPENAI_API_KEY"):
            client = OpenAI()
        self.client = client
        self.model = model
        self.request_timeout_s = float(request_timeout_s)
        random.seed(seed)
        # Background producer state
        self._q: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._producer_thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()

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
                timeout=self.request_timeout_s,
            )
            text = resp.choices[0].message.content  # type: ignore[index]
            return json.loads(text) if text else None
        except Exception:
            return None

    # ---------- Multi-pass numbers synthesis ----------
    def _merge_payloads(self, base: Dict[str, Any], extra: Dict[str, Any], allowed: set[str]) -> Dict[str, Any]:
        def uniq(seq: List[Any]) -> List[Any]:
            seen = set()
            out = []
            for x in seq:
                key = json.dumps(x, sort_keys=True) if not isinstance(x, str) else x
                if key not in seen:
                    seen.add(key)
                    out.append(x)
            return out

        out: Dict[str, Any] = {}
        out["concepts"] = uniq(list(base.get("concepts", [])) + list(extra.get("concepts", [])))
        # Restrict relations to allowed
        rels = list(base.get("relations", [])) + list(extra.get("relations", []))
        out["relations"] = uniq([r for r in rels if r in allowed])
        # Keep only well-formed triples and allowed relations
        btri = [t for t in base.get("triples", []) if isinstance(t, list) and len(t) == 3 and t[1] in allowed]
        etri = [t for t in extra.get("triples", []) if isinstance(t, list) and len(t) == 3 and t[1] in allowed]
        out["triples"] = uniq(btri + etri)
        out["analogies"] = uniq(list(base.get("analogies", [])) + list(extra.get("analogies", [])))
        out["equivalences"] = uniq(list(base.get("equivalences", [])) + list(extra.get("equivalences", [])))
        return out

    def fetch_numbers_payload(self, min_triples: int = 256, max_calls: int = 4, sleep_s: float = 0.0) -> Optional[Dict[str, Any]]:
        """
        Call the LLM multiple times and merge JSON payloads until at least
        min_triples valid numeric triples are collected. Returns a validated
        payload or None if client is not configured or calls fail.
        """
        if self.client is None:
            return None
        try:
            from concept_learner.data.curriculum import allowed_relations_for_phase, SEEDS

            allowed = set(allowed_relations_for_phase(4)) | {"has_tens", "has_ones", "makes_ten_with"}
            prompt = self.build_generator_prompt("Phase 4 — Numbers", list(allowed), SEEDS)
        except Exception:
            allowed = {"successor_of", "predecessor_of", "has_tens", "has_ones", "makes_ten_with"}
            prompt = self.build_generator_prompt("Phase 4 — Numbers", list(allowed), ["1", "2", "3"])  # fallback

        merged: Dict[str, Any] = {"concepts": [], "relations": list(allowed), "triples": [], "analogies": [], "equivalences": []}
        calls = 0
        while calls < max_calls and len(merged.get("triples", [])) < int(min_triples):
            data = self._chat_json(prompt)
            calls += 1
            if isinstance(data, dict):
                merged = self._merge_payloads(merged, data, allowed)
            if sleep_s > 0:
                time.sleep(sleep_s)
        # Force relations to allowed set subset
        merged["relations"] = [r for r in merged.get("relations", []) if r in allowed]
        ok, errs = self.validate_numbers(merged)
        if not ok:
            # If we have some triples but minor errors, drop offending triples by relation and re-validate
            bad_rels = set()
            for e in errs:
                if e.startswith("relation not allowed:"):
                    bad_rels.add(e.split(":", 1)[1].strip())
            if bad_rels:
                merged["triples"] = [t for t in merged.get("triples", []) if t[1] not in bad_rels]
                merged["relations"] = [r for r in merged.get("relations", []) if r not in bad_rels]
                ok, _ = self.validate_numbers(merged)
        if ok and len(merged.get("triples", [])) >= 1:
            return merged
        return None

    # ---------- Background producer/consumer ----------
    def start_numbers_producer(self, min_triples: int = 512, buffer_size: int = 4, max_calls: int = 4, poll_sleep: float = 0.2) -> None:
        """
        Start a background thread that repeatedly fetches numbers payloads
        from the LLM and enqueues them as episodes. Acts as a producer in
        a producer-consumer setup for the training loop.
        """
        if self.client is None:
            return
        if self._producer_thread is not None and self._producer_thread.is_alive():
            return
        self._stop_flag.clear()

        def _run():
            while not self._stop_flag.is_set():
                try:
                    if self._q.qsize() >= buffer_size:
                        time.sleep(poll_sleep)
                        continue
                    payload = self.fetch_numbers_payload(min_triples=min_triples, max_calls=max_calls)
                    if payload is not None:
                        ep = {"domain": "numbers", "type": "kg", "payload": payload, "answers": {}}
                        self._q.put(ep)
                    else:
                        time.sleep(poll_sleep)
                except Exception:
                    time.sleep(poll_sleep)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        self._producer_thread = t

    def stop_producer(self) -> None:
        self._stop_flag.set()
        if self._producer_thread is not None:
            self._producer_thread.join(timeout=1.0)
        self._producer_thread = None

    def pop_numbers_episode(self, timeout: float = 0.0) -> Optional[Dict[str, Any]]:
        try:
            if timeout and timeout > 0:
                return self._q.get(timeout=timeout)
            if not self._q.empty():
                return self._q.get_nowait()
        except Exception:
            return None
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
            # Provide a sufficiently large, validated numeric payload so that
            # downstream code can build batches of the requested size.
            payload = self._make_synthetic_numbers_payload(min_triples=max(64, int(batch_size)))
            ok, errs = self.validate_numbers(payload)
            if ok:
                episodes.append({
                    "domain": "numbers",
                    "type": "kg",
                    "payload": payload,
                    "answers": {},
                })
            return episodes

        # Attempt to synthesize with the OpenAI SDK and collect enough data
        if self.client is not None:
            # Prefer consuming from background queue if started
            ep = self.pop_numbers_episode(timeout=0.0)
            if ep is None:
                payload = self.fetch_numbers_payload(min_triples=max(4 * batch_size, 256))
                if payload is not None:
                    episodes.append({
                        "domain": "numbers",
                        "type": "kg",
                        "payload": payload,
                        "answers": {},
                    })
            else:
                episodes.append(ep)
        return episodes

    # ---------- Local synthetic fallback ----------
    def _make_synthetic_numbers_payload(self, min_triples: int = 128) -> Dict[str, Any]:
        """
        Build a deterministic numeric knowledge payload with enough triples
        to satisfy typical training batch sizes. Includes successor/
        predecessor, place value, and make-ten facts over 0..99.
        """
        # Concepts: digits 0..99 plus a few number words for equivalences
        concepts = [str(i) for i in range(0, 100)] + [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
        ]
        relations = [
            "successor_of",
            "predecessor_of",
            "has_tens",
            "has_ones",
            "makes_ten_with",
        ]
        triples: List[List[str]] = []

        # successor_of: 0..98 -> (i+1 successor_of i)
        for i in range(0, 99):
            triples.append([str(i + 1), "successor_of", str(i)])

        # predecessor_of: 1..99 -> (i-1 predecessor_of i)
        for i in range(1, 100):
            triples.append([str(i - 1), "predecessor_of", str(i)])

        # place value for 0..99
        for i in range(0, 100):
            tens = i // 10
            ones = i % 10
            triples.append([str(i), "has_tens", str(tens)])
            triples.append([str(i), "has_ones", str(ones)])

        # make-ten within 0..10 (covers all 0..9 pairs at least once)
        for i in range(0, 11):
            j = 10 - i
            if 0 <= j <= 10:
                triples.append([str(i), "makes_ten_with", str(j)])

        # Analogies: a few simple templates
        analogies: List[List[List[str]]] = []
        # successor analogy: 2:3 :: 5:6, etc.
        for a in [(2, 3, 5, 6), (2, 4, 3, 5), (7, 10, 6, 9)]:
            analogies.append([[str(a[0]), str(a[1])], [str(a[2]), str(a[3])]])

        # Equivalences: number words ↔ digits
        eq = [
            ["zero", "0"],
            ["one", "1"],
            ["two", "2"],
            ["three", "3"],
            ["four", "4"],
            ["five", "5"],
            ["six", "6"],
            ["seven", "7"],
            ["eight", "8"],
            ["nine", "9"],
            ["ten", "10"],
        ]

        # If, for some reason, trimming is requested (tests), keep at least min_triples
        if len(triples) < int(min_triples):
            # With 0..99 construction we typically have > 500 triples; this branch is
            # defensive and should not trigger.
            pass

        return {
            "concepts": concepts,
            "relations": relations,
            "triples": triples,
            "analogies": analogies,
            "equivalences": eq,
        }
