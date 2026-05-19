Qwen3.5-9B input bundle.

Use:
- `--ref examples/Qwen3.5-9B/reference`
- `--acc-checker examples/Qwen3.5-9B/accuracy_checker`
- `--bench examples/Qwen3.5-9B/benchmark`

Each folder contains scripts plus a short README.
- `README.md` — this file.

Expected files (for agent_system orchestrator):
- reference_inference.py
- accuracy_check.py
- benchmark.py
- config.json
- requirements.txt (GPU dependencies for verifier/benchmark)

The CLI reads these files from `examples/Qwen3.5-9B` by default.

A separate `.venv/` is auto-created here by the verifier using `uv` with the dependencies from `requirements.txt` (torch, transformers, httpx).
