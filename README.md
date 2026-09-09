# WordMaze RL

Minimal reproduction of a GRPO experiment on the WordMaze dataset with
`Qwen/Qwen3.5-4B`.

## Files

- `evaluate.py`: reproducible base-model and LoRA evaluation with vLLM.
- `train_grpo.py`: LoRA GRPO training with TRL.
- `verifier.py`: shared deterministic reward/verifier.
- `plan.md`: experiment order, completion criteria, and budget limits.
- `m3-4/`, `m4-6/`: frozen local Parquet splits used by both scripts.
- `docs/original-readme.md`: original Hugging Face dataset README.

## Run

Use Linux with an NVIDIA GPU and persistent storage:

```bash
export HF_HOME=/workspace/cache/huggingface
export VLLM_CACHE_ROOT=/workspace/cache/vllm

# One-item smoke test, then a 20-item validation baseline.
uv run evaluate.py --config m3-4 --split validation --limit 1
uv run evaluate.py --config m3-4 --split validation --limit 20

# Only after evaluation works.
uv run train_grpo.py --max-steps 5
```

See [`plan.md`](plan.md) before running the full test baseline or training.
