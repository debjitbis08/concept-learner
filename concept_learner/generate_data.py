from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

from .teacher.llm_teacher import LLMTeacher


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Generate pretraining data via LLMTeacher and save to JSON")
    parser.add_argument("--out", required=True, help="output JSON path")
    parser.add_argument("--count", type=int, default=1, help="number of payloads to generate")
    parser.add_argument("--min_triples", type=int, default=256, help="minimum triples per payload")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument("--offline", action="store_true", help="force offline synthetic generation (no API calls)")
    args = parser.parse_args()

    teacher = LLMTeacher(None, model=args.model)
    if args.offline:
        teacher.client = None

    episodes: List[Dict[str, Any]] = []
    for _ in range(args.count):
        if teacher.client is None:
            payload = teacher._make_synthetic_numbers_payload(min_triples=args.min_triples)
        else:
            payload = teacher.fetch_numbers_payload(min_triples=args.min_triples)
            if payload is None:
                # Fallback to synthetic if API fails
                payload = teacher._make_synthetic_numbers_payload(min_triples=args.min_triples)
        episodes.append({"domain": "numbers", "type": "kg", "payload": payload, "answers": {}})

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"episodes": episodes}, f, indent=2)
    print(f"[generate_data] Wrote {len(episodes)} episodes to {args.out}")


if __name__ == "__main__":
    main()

