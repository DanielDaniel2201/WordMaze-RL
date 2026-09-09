# WordMaze RL

Minimal reproduction of a GRPO experiment on the WordMaze dataset with
`Qwen/Qwen3.5-4B`.

## Files

- `evaluate.py`: reproducible base-model and LoRA evaluation with vLLM.
- `train_grpo.py`: LoRA GRPO training with TRL.
- `verifier.py`: shared deterministic reward/verifier.
- `run_baseline.sh`: one-command Qwen baseline on RunPod.
- `plan.md`: experiment order, completion criteria, and budget limits.
- `m3-4/`, `m4-6/`: frozen local Parquet splits used by both scripts.
- `docs/original-readme.md`: original Hugging Face dataset README.

## Run

Use Linux with an NVIDIA GPU and persistent storage. The script verifies the
data, runs a one-item smoke test, then evaluates validation and both test sets:

```bash
bash run_baseline.sh
```

See [`plan.md`](plan.md) before running the full test baseline or training.
