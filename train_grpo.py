"""Train a Qwen3.5-4B LoRA on WordMaze with binary-reward GRPO.

Run on a Linux GPU machine, starting with a smoke test:
    uv run train_grpo.py --max-steps 5
    uv run train_grpo.py --max-steps 100 --use-vllm
"""
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#     "datasets>=4.0",
#     "peft>=0.17",
#     "trl[peft,vllm]>=0.28",
#     "wordfreq==3.1.1",
# ]
# ///
from __future__ import annotations

import argparse
from pathlib import Path

from verifier import CHECK_NAMES, verify


DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
KEY_FIELDS = ("start", "goal", "password", "word_length", "max_moves")
GOLD_COLUMNS = ("solution", "answer", "path", "num_moves")


def completion_text(completion: object) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        message = completion[0]
        if isinstance(message, dict):
            return str(message.get("content", ""))
    return str(completion)


def wordmaze_reward(
    completions: list,
    start: list,
    goal: list,
    word_length: list,
    max_moves: list,
    password: list,
    **kwargs,
) -> list[float]:
    results = [
        verify(
            completion_text(completion),
            {
                "start": row_start,
                "goal": row_goal,
                "word_length": row_length,
                "max_moves": row_moves,
                "password": row_password,
            },
        )
        for completion, row_start, row_goal, row_length, row_moves, row_password in zip(
            completions, start, goal, word_length, max_moves, password, strict=True
        )
    ]
    if log_extra := kwargs.get("log_extra"):
        for name in (*CHECK_NAMES, "fully_valid"):
            log_extra(name, [int(result[name]) for result in results])
    return [float(result["fully_valid"]) for result in results]


def load_parquet(root: Path, config: str, split: str):
    from datasets import load_dataset

    path = root / config / f"{split}-00000-of-00001.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    return load_dataset("parquet", data_files={"train": str(path)}, split="train")


def key_for(row: dict) -> tuple:
    return tuple(row[name] for name in KEY_FIELDS)


def held_out_keys(root: Path) -> set[tuple]:
    return {
        key_for(row)
        for config in ("m3-4", "m4-6")
        for split in ("validation", "test")
        for row in load_parquet(root, config, split)
    }


def prepare_dataset(dataset):
    dataset = dataset.remove_columns([name for name in ("prompt", *GOLD_COLUMNS) if name in dataset.column_names])
    return dataset.rename_column("messages", "prompt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--config", choices=("m3-4", "m4-6"), default="m3-4")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/grpo"))
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-limit", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument("--resume-from-checkpoint")
    return parser


def main() -> None:
    import torch
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    args = build_parser().parse_args()
    if args.max_steps <= 0 or args.num_generations < 2:
        raise SystemExit("--max-steps must be positive and --num-generations must be at least 2")

    root = Path(__file__).resolve().parent
    blocked = held_out_keys(root)
    train = load_parquet(root, args.config, "train")
    before = len(train)
    train = train.filter(lambda row: key_for(row) not in blocked)
    train = prepare_dataset(train)

    validation = load_parquet(root, args.config, "validation")
    if args.eval_limit:
        validation = validation.select(range(min(args.eval_limit, len(validation))))
    validation = prepare_dataset(validation)
    print(f"training rows after held-out filtering: {before} -> {len(train)}")

    model_init_kwargs = {"dtype": torch.bfloat16}
    if args.revision:
        model_init_kwargs["revision"] = args.revision

    training_args = GRPOConfig(
        output_dir=str(args.output_dir),
        model_init_kwargs=model_init_kwargs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=1,
        generation_batch_size=args.num_generations,
        num_generations=args.num_generations,
        num_generations_eval=1,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        beta=args.beta,
        loss_type="grpo",
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        log_completions=True,
        report_to="none",
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        per_device_eval_batch_size=1,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        seed=args.seed,
        chat_template_kwargs={"enable_thinking": args.thinking},
        use_vllm=args.use_vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_length=args.max_completion_length + 2048,
    )
    trainer = GRPOTrainer(
        model=args.model,
        args=training_args,
        reward_funcs=wordmaze_reward,
        train_dataset=train,
        eval_dataset=validation,
        peft_config=LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"],
        ),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output_dir))


if __name__ == "__main__":
    main()
