Repository Agent Guidelines (Codex CLI)

This document defines how automated coding agents (e.g., Codex CLI) should work in this repository. Follow these rules unless the task explicitly requests otherwise.

Scope & Principles
- Minimize scope: change only what’s needed to satisfy the request.
- Prefer additive changes over refactors; avoid breaking training logic.
- Keep solutions simple; fix root causes, not symptoms.
- Ask when requirements are ambiguous or affect multiple components.

Files & Imports
- Place new Python modules under `src/concept_learner/` and import via `concept_learner.<module>`.
- Scripts live under `scripts/`. Avoid adding top‑level importable modules at repo root.
- Do not rename existing public modules or scripts unless requested.

Commits
- Use Conventional Commits.
  - Types: `fix:`, `feat:`, `docs:`, `refactor:`, `test:`, `chore:`
  - Subject: present tense, <= 50 chars, no trailing period.
  - One focused commit per logical change when possible.

Testing & Validation
- If tests exist, run them locally when feasible before finalizing.
- Validate only the affected area first; expand as needed for confidence.
- Do not add new dependencies unless necessary; if added, update `pyproject.toml`.

Code Style
- Match existing style and structure; keep functions small and clear.
- Prefer explicit names over one‑letter vars (unless idiomatic).
- Avoid adding heavy logging; use concise `print` in scripts when needed.

Evaluation & Logging
- For pair relations, print canonical symbolic forms in logs:
  - `succ(a) == b ?`, `pred(a) == b ?`, `greater(a,b) ?`, `smaller(a,b) ?`, `same_ones(a,b) ?`, `same_tens(a,b) ?`.
- Keep any verbose natural language phrasing optional or for samples only.
- When adding utilities used by scripts, put them in `src/concept_learner/` and import from there (avoid root‑level modules).

Safety & Sandboxing
- Assume workspace‑write filesystem and restricted networking unless the user approves otherwise.
- Avoid destructive actions (`rm -rf`, history rewrites) unless explicitly asked.

Planning & Communication
- For multi‑step tasks, outline a short plan and keep the user informed at key checkpoints.
- Prefer concise, actionable updates over long explanations.

Performance
- Use efficient file/code search (`rg`) and avoid reading huge files at once.
- Don’t introduce O(N^2) patterns on hot paths without reason.

Documentation
- Update README/docs when user‑facing behavior changes (especially eval output format).
- Keep docs succinct; show how to run and what changed.

