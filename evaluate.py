"""Run a reproducible WordMaze pass@1 evaluation with vLLM.

Run on a Linux GPU machine, for example:
    uv run evaluate.py --config m3-4 --split validation --limit 20
    uv run evaluate.py --config m3-4 --split test
    uv run evaluate.py --config m3-4 --split test --adapter outputs/grpo
"""
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#     "pandas>=2.2",
#     "pyarrow>=17",
#     "vllm==0.26.0",
#     "wordfreq==3.1.1",
# ]
# ///
from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from verifier import CHECK_NAMES, verify


DEFAULT_MODEL = "Qwen/Qwen3.5-4B"


def load_rows(root: Path, config: str, split: str, limit: int | None) -> list[dict]:
    import pandas as pd

    path = root / config / f"{split}-00000-of-00001.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    if limit is not None:
        frame = frame.head(limit)
    return frame.to_dict("records")


def messages_for(row: dict) -> list[dict[str, str]]:
    messages = row["messages"]
    if hasattr(messages, "tolist"):
        messages = messages.tolist()
    return [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in messages
    ]


def default_output(root: Path, model: str, config: str, split: str, adapter: str | None) -> Path:
    model_name = model.rstrip("/").split("/")[-1]
    suffix = "grpo" if adapter else "base"
    return root / "results" / f"{model_name}-{suffix}-{config}-{split}.jsonl"


def summarize(records: list[dict]) -> dict:
    total = len(records)
    return {
        "total": total,
        "accuracy": sum(record["score"] for record in records) / total if total else 0.0,
        "check_rates": {
            name: sum(record["checks"][name] for record in records) / total if total else 0.0
            for name in (*CHECK_NAMES, "fully_valid")
        },
        "truncated": sum(record["truncated"] for record in records),
        "prompt_tokens": sum(record["prompt_tokens"] for record in records),
        "completion_tokens": sum(record["completion_tokens"] for record in records),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--adapter", help="Optional local LoRA adapter path")
    parser.add_argument("--config", choices=("m3-4", "m4-6"), default="m3-4")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    from vllm import LLM, SamplingParams

    args = build_parser().parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.max_model_len <= args.max_tokens:
        raise SystemExit("--max-model-len must be larger than --max-tokens")

    root = Path(__file__).resolve().parent
    rows = load_rows(root, args.config, args.split, args.limit)
    output_path = args.output or default_output(root, args.model, args.config, args.split, args.adapter)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_kwargs = {
        "model": args.model,
        "dtype": "bfloat16",
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enable_lora": bool(args.adapter),
    }
    if args.revision:
        model_kwargs["revision"] = args.revision
    llm = LLM(**model_kwargs)
    sampling = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    lora_request = None
    if args.adapter:
        from vllm.lora.request import LoRARequest

        lora_request = LoRARequest("wordmaze", 1, str(Path(args.adapter).resolve()))

    started = time.perf_counter()
    outputs = llm.chat(
        [messages_for(row) for row in rows],
        sampling,
        use_tqdm=True,
        lora_request=lora_request,
        chat_template_kwargs={"enable_thinking": args.thinking},
    )
    elapsed = time.perf_counter() - started

    records = []
    for row, output in zip(rows, outputs, strict=True):
        candidate = output.outputs[0]
        checks = verify(candidate.text, row)
        records.append(
            {
                "id": row["id"],
                "completion": candidate.text,
                "checks": checks,
                "score": int(checks["fully_valid"]),
                "prompt_tokens": len(output.prompt_token_ids),
                "completion_tokens": len(candidate.token_ids),
                "truncated": candidate.finish_reason == "length",
            }
        )

    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    metrics = {
        "created_at": datetime.now(UTC).isoformat(),
        "model": args.model,
        "revision": args.revision,
        "adapter": args.adapter,
        "config": args.config,
        "split": args.split,
        "generation": {
            "thinking": args.thinking,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
        },
        "elapsed_seconds": elapsed,
        **summarize(records),
    }
    metrics_path = output_path.with_name(f"{output_path.stem}-metrics.json")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"wrote {output_path}")
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()
