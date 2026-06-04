"""Generate the Wordmaze dataset: synthetic word-ladder puzzles with an F/B password.

Build a graph where words of a fixed length are nodes and edges join words that are one
letter apart, sample simple paths on it, and read (start, goal, password) off each path.
Because the password is derived from the walk, every generated puzzle is solvable by
construction. Output is a Hugging Face DatasetDict, per-split JSONL, and a dataset card.

Dependencies are declared inline below (PEP 723), so just run it with uv, which builds an
ephemeral environment on the fly. Nothing else to install:

    # easier config (m3-4): 3-4 moves, 4-6 letter words
    uv run create_dataset.py --move-counts 3-4 --word-lengths 4-6 --num-examples 2000 --seed 13

    # default/hard config (m4-6), pushed to the Hub as a named config
    uv run create_dataset.py --move-counts 4-6 --word-lengths 4-6 --num-examples 2000 --seed 42 --push-to-hub --repo-id immortal3/wordmaze --config-name m4-6
"""
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "wordfreq>=3.0",
#     "nltk>=3.8",
#     "datasets>=2.14",
#     "huggingface_hub>=0.20",
# ]
# ///
from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from wordfreq import top_n_list


PROPER_NOUN_PLACES = {
    # --- Countries (4-7 letters) ---
    "chad", "cuba", "fiji", "iran", "iraq", "laos", "mali", "oman", "peru", "togo",
    "china", "egypt", "ghana", "haiti", "india", "italy", "japan", "kenya", "libya",
    "nepal", "qatar", "spain", "sudan", "syria", "yemen", "gabon", "congo", "chile",
    "kuwait", "brazil", "canada", "france", "greece", "guinea", "israel", "jordan",
    "kosovo", "latvia", "malawi", "mexico", "monaco", "norway", "panama", "poland",
    "russia", "rwanda", "serbia", "sweden", "taiwan", "turkey", "uganda", "zambia",
    "albania", "algeria", "andorra", "angola", "armenia", "austria", "bahrain",
    "belarus", "belgium", "bolivia", "burundi", "comoros", "croatia", "cyprus",
    "denmark", "ecuador", "eritrea", "estonia", "ethiopia", "finland", "georgia",
    "germany", "hungary", "iceland", "ireland", "jamaica", "lebanon", "lesotho",
    "liberia", "moldova", "morocco", "myanmar", "namibia", "nigeria", "romania",
    "senegal", "somalia", "tunisia", "ukraine", "uruguay", "vanuatu", "vietnam",
    # --- Major cities (mostly capitals + biggest) ---
    "oslo", "rome", "kiev", "lima", "baku", "doha",
    "paris", "tokyo", "seoul", "miami", "tampa", "cairo", "dakar", "lagos", "rabat",
    "dubai", "kabul", "milan", "osaka", "delhi", "boise", "tulsa", "bronx", "fargo",
    "london", "berlin", "madrid", "moscow", "lisbon", "vienna", "prague", "dublin",
    "athens", "boston", "geneva", "beirut", "warsaw", "ankara", "tehran", "munich",
    "havana", "manila", "jakarta", "nairobi", "bangkok", "seattle", "sydney",
    "atlanta", "chicago", "houston", "phoenix", "detroit", "denver", "memphis",
    "buenos", "caracas", "bogota", "shanghai", "beijing", "karachi", "tbilisi",
    # --- US states ---
    "ohio", "iowa", "utah",
    "texas", "idaho", "maine",
    "alaska", "hawaii", "kansas", "nevada", "oregon",
    "alabama", "arizona", "florida", "georgia", "indiana", "montana", "vermont",
    "wyoming", "arkansas", "delaware", "illinois", "kentucky", "maryland",
    "michigan", "missouri", "nebraska", "oklahoma", "virginia", "wisconsin",
    "colorado", "louisiana", "minnesota", "tennessee", "wisconsin", "california",
    "connecticut", "mississippi",
}


def load_proper_nouns() -> set[str]:
    """Lowercased personal names (from NLTK) + place names (hardcoded above)."""
    import nltk
    from nltk.corpus import names

    try:
        first_names = {n.lower() for n in names.words()}
    except LookupError:
        nltk.download("names", quiet=True)
        first_names = {n.lower() for n in names.words()}
    return first_names | PROPER_NOUN_PLACES


DEFAULT_SYSTEM_PROMPT = """You solve Wordmaze puzzles: word-ladder problems with a password constraint.

Rules:
- Change exactly one letter per move.
- Every word in the path must be a valid English word of the requested length.
- The path must reach the goal in exactly `max_moves` moves (so it has `max_moves + 1` words).
- For each move, record F if the changed letter moves forward alphabetically (e.g. l -> r), B if it moves backward (e.g. o -> a).
- Concatenating F/B for every move gives the path's password — it must equal the given password exactly.

Output only the final path inside <answer>...</answer>, with words joined by " -> ".

Example input:
start: cold
goal: warm
word_length: 4
max_moves: 4
password: FFBF

Example output:
<answer>cold -> cord -> word -> ward -> warm</answer>"""


DEFAULT_USER_PROMPT_TEMPLATE = """start: {start}
goal: {goal}
word_length: {word_length}
max_moves: {max_moves}
password: {password}"""


@dataclass(frozen=True)
class GenerationConfig:
    num_examples: int
    word_lengths: tuple[int, ...]
    move_counts: tuple[int, ...]
    seed: int
    attempts_per_example: int


def parse_int_range(raw: str, name: str) -> tuple[int, ...]:
    values: set[int] = set()

    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue

        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            if start > end:
                raise ValueError(f"{name} range must be ascending: {part}")
            values.update(range(start, end + 1))
        else:
            values.add(int(part))

    if not values:
        raise ValueError(f"{name} cannot be empty")

    return tuple(sorted(values))


def move_count_slug(move_counts: tuple[int, ...]) -> str:
    if len(move_counts) == 1:
        return f"m{move_counts[0]}"
    return f"m{min(move_counts)}-{max(move_counts)}"


def load_common_word_source(top_n: int, filter_proper_nouns: bool = True) -> list[str]:
    words = top_n_list("en", top_n)
    if filter_proper_nouns:
        proper_nouns = load_proper_nouns()
        words = [w for w in words if w not in proper_nouns]
    return words


def clean_word(raw: str) -> str | None:
    word = raw.strip().lower()
    if not word:
        return None

    if not re.fullmatch(r"[a-z]+", word):
        return None

    return word


def load_words(source: Iterable[str], word_lengths: Iterable[int]) -> dict[int, list[str]]:
    wanted_lengths = set(word_lengths)
    words_by_length: dict[int, set[str]] = {length: set() for length in wanted_lengths}

    for raw in source:
        word = clean_word(raw)
        if word is None:
            continue
        if len(word) in wanted_lengths:
            words_by_length[len(word)].add(word)

    return {
        length: sorted(words)
        for length, words in sorted(words_by_length.items())
        if words
    }


def build_graph(words: list[str]) -> dict[str, tuple[str, ...]]:
    if not words:
        return {}

    word_length = len(words[0])
    buckets: dict[str, list[str]] = defaultdict(list)
    for word in words:
        for index in range(word_length):
            buckets[f"{word[:index]}*{word[index + 1:]}"].append(word)

    adjacency: dict[str, set[str]] = {word: set() for word in words}
    for bucket_words in buckets.values():
        if len(bucket_words) < 2:
            continue

        bucket_set = set(bucket_words)
        for word in bucket_words:
            adjacency[word].update(bucket_set)
            adjacency[word].discard(word)

    return {
        word: tuple(sorted(neighbors))
        for word, neighbors in adjacency.items()
        if neighbors
    }


def changed_index(left: str, right: str) -> int | None:
    changed = [
        index
        for index, pair in enumerate(zip(left, right, strict=True))
        if pair[0] != pair[1]
    ]
    if len(changed) != 1:
        return None
    return changed[0]


def password_for_path(path: list[str]) -> str:
    password = []
    for left, right in zip(path, path[1:], strict=False):
        index = changed_index(left, right)
        if index is None:
            raise ValueError(f"Invalid path transition: {left} -> {right}")

        password.append("F" if right[index] > left[index] else "B")

    return "".join(password)


def validate_path(
    path: list[str],
    words: set[str],
    word_length: int,
    max_moves: int,
    password: str,
) -> None:
    if len(path) != max_moves + 1:
        raise ValueError(f"Expected {max_moves + 1} words, got {len(path)}")

    for word in path:
        if len(word) != word_length:
            raise ValueError(f"Incorrect word length for {word}")
        if word not in words:
            raise ValueError(f"Unknown dictionary word: {word}")

    if password_for_path(path) != password:
        raise ValueError(f"Password mismatch for path: {' -> '.join(path)}")


def random_walk(
    graph: dict[str, tuple[str, ...]],
    nodes: tuple[str, ...],
    move_count: int,
    rng: random.Random,
) -> list[str] | None:
    if not nodes:
        return None

    start = rng.choice(nodes)
    path = [start]
    seen = {start}
    current = start

    for _ in range(move_count):
        candidates = [neighbor for neighbor in graph[current] if neighbor not in seen]
        if not candidates:
            return None

        current = rng.choice(candidates)
        path.append(current)
        seen.add(current)

    return path


def build_user_prompt(start: str, goal: str, word_length: int, max_moves: int, password: str) -> str:
    return DEFAULT_USER_PROMPT_TEMPLATE.format(
        start=start,
        goal=goal,
        word_length=word_length,
        max_moves=max_moves,
        password=password,
    )


def build_example(
    example_index: int,
    path: list[str],
    word_length: int,
    move_count: int,
    seed: int,
) -> dict[str, object]:
    start = path[0]
    goal = path[-1]
    password = password_for_path(path)
    solution = " -> ".join(path)
    user_prompt = build_user_prompt(start, goal, word_length, move_count, password)

    return {
        "id": f"wordmaze-{seed}-{example_index:05d}",
        "start": start,
        "goal": goal,
        "word_length": word_length,
        "max_moves": move_count,
        "password": password,
        "prompt": user_prompt,
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "solution": solution,
        "answer": f"<answer>{solution}</answer>",
        "path": path,
        "num_moves": move_count,
    }


def build_quota(total: int, combos: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    base, remainder = divmod(total, len(combos))
    return {
        combo: base + (1 if index < remainder else 0)
        for index, combo in enumerate(combos)
    }


def generate_examples(
    words_by_length: dict[int, list[str]],
    config: GenerationConfig,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rng = random.Random(config.seed)

    graphs = {
        word_length: build_graph(words)
        for word_length, words in words_by_length.items()
        if word_length in config.word_lengths
    }
    graph_nodes = {
        word_length: tuple(graph.keys())
        for word_length, graph in graphs.items()
    }
    word_sets = {word_length: set(words) for word_length, words in words_by_length.items()}

    combos = [
        (word_length, move_count)
        for word_length in config.word_lengths
        for move_count in config.move_counts
        if graphs.get(word_length)
    ]
    if not combos:
        raise RuntimeError("No usable word-ladder graphs were built from the supplied word list")

    rng.shuffle(combos)
    quota = build_quota(config.num_examples, combos)

    examples: list[dict[str, object]] = []
    seen: set[tuple[str, str, str, int, int]] = set()
    failures: Counter[str] = Counter()
    combo_counts: Counter[str] = Counter()

    def try_add_example(word_length: int, move_count: int) -> bool:
        graph = graphs[word_length]
        nodes = graph_nodes[word_length]
        words = word_sets[word_length]

        for _ in range(config.attempts_per_example):
            path = random_walk(graph, nodes, move_count, rng)
            if path is None:
                failures[f"{word_length}/{move_count}/dead_end"] += 1
                continue

            password = password_for_path(path)
            key = (path[0], path[-1], password, word_length, move_count)
            if key in seen:
                failures[f"{word_length}/{move_count}/duplicate"] += 1
                continue

            validate_path(path, words, word_length, move_count, password)
            seen.add(key)
            examples.append(build_example(len(examples), path, word_length, move_count, config.seed))
            combo_counts[f"{word_length}_letters/{move_count}_moves"] += 1
            return True

        failures[f"{word_length}/{move_count}/exhausted"] += 1
        return False

    for (word_length, move_count), target_count in quota.items():
        for _ in range(target_count):
            try_add_example(word_length, move_count)

    fill_attempts = 0
    max_fill_attempts = config.num_examples * config.attempts_per_example
    while len(examples) < config.num_examples and fill_attempts < max_fill_attempts:
        fill_attempts += 1
        word_length, move_count = rng.choice(combos)
        try_add_example(word_length, move_count)

    if len(examples) < config.num_examples:
        raise RuntimeError(
            f"Only generated {len(examples)} examples out of {config.num_examples}. "
            "Use a larger word list, reduce --num-examples, increase --attempts-per-example, "
            "or narrow --word-lengths/--move-counts."
        )

    rng.shuffle(examples)
    for index, example in enumerate(examples):
        example["id"] = f"wordmaze-{config.seed}-{index:05d}"

    stats = {
        "requested_examples": config.num_examples,
        "generated_examples": len(examples),
        "word_lengths": list(config.word_lengths),
        "move_counts": list(config.move_counts),
        "seed": config.seed,
        "words_by_length": {
            str(word_length): len(words)
            for word_length, words in sorted(words_by_length.items())
        },
        "graph_nodes_by_length": {
            str(word_length): len(graph)
            for word_length, graph in sorted(graphs.items())
        },
        "examples_by_combo": dict(sorted(combo_counts.items())),
        "failures": dict(sorted(failures.items())),
    }
    return examples, stats


def split_examples(
    examples: list[dict[str, object]],
    train_ratio: float,
    validation_ratio: float,
) -> dict[str, list[dict[str, object]]]:
    if train_ratio <= 0 or validation_ratio < 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("Expected ratios where 0 < train and train + validation < 1")

    total = len(examples)
    train_count = int(total * train_ratio)
    validation_count = int(total * validation_ratio)

    return {
        "train": examples[:train_count],
        "validation": examples[train_count : train_count + validation_count],
        "test": examples[train_count + validation_count :],
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_dataset_card(path: Path, stats: dict[str, object]) -> None:
    path.write_text(
        "\n".join(
            [
                f"# Wordmaze ({stats.get('move_count_range', 'unknown')} moves)",
                "",
                "Synthetic word-ladder puzzles for RL training.",
                "",
                "Each example asks the model to find a path from a start word to a goal word. ",
                "Every move changes exactly one letter, every intermediate token must be a valid ",
                "dictionary word of the requested length, and the path must match the supplied ",
                "forward/backward password pattern.",
                "",
                "## Fields",
                "",
                "- `id`: stable example id",
                "- `start`: start word",
                "- `goal`: goal word",
                "- `word_length`: required word length",
                "- `max_moves`: maximum number of one-letter moves",
                "- `password`: F/B pattern for the required path",
                "- `prompt`: plain text user prompt",
                "- `messages`: chat-format user prompt",
                "- `solution`: known valid path",
                "- `answer`: known valid XML answer",
                "- `path`: solution words as a list",
                "- `num_moves`: number of moves in `solution`",
                "",
                "## Generation Stats",
                "",
                "```json",
                json.dumps(stats, indent=2, sort_keys=True),
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )


def save_huggingface_dataset(
    splits: dict[str, list[dict[str, object]]],
    out_dir: Path,
    repo_id: str | None,
    push_to_hub: bool,
    private: bool,
    token: str | None,
    config_name: str | None,
) -> None:
    try:
        from datasets import Dataset, DatasetDict
    except ImportError as exc:
        if push_to_hub:
            raise RuntimeError("Install datasets to push to Hugging Face") from exc
        print("datasets is not installed; wrote JSONL files only")
        return

    dataset_dict = DatasetDict(
        {
            split_name: Dataset.from_list(rows)
            for split_name, rows in splits.items()
        }
    )
    dataset_dict.save_to_disk(str(out_dir / "hf_dataset"))

    if not push_to_hub:
        return

    if not repo_id:
        raise ValueError("--repo-id is required when --push-to-hub is set")

    push_kwargs = {
        "private": private,
        "token": token,
        "commit_message": f"Upload wordmaze config {config_name or 'default'}",
    }
    if config_name:
        push_kwargs["config_name"] = config_name

    dataset_dict.push_to_hub(repo_id, **push_kwargs)


def write_outputs(
    splits: dict[str, list[dict[str, object]]],
    stats: dict[str, object],
    out_dir: Path,
    repo_id: str | None,
    push_to_hub: bool,
    private: bool,
    token: str | None,
    config_name: str | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out_dir / "data"
    for split_name, rows in splits.items():
        write_jsonl(data_dir / f"{split_name}.jsonl", rows)

    (out_dir / "metadata.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_dataset_card(out_dir / "README.md", stats)
    save_huggingface_dataset(
        splits, out_dir, repo_id, push_to_hub, private, token, config_name
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate wordmaze puzzles in Hugging Face dataset format.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory for JSONL splits, metadata, dataset card, and save_to_disk output. "
        "Defaults to <script-dir>/data/wordmaze-<move-slug>, e.g. wordmaze/data/wordmaze-m3-4.",
    )
    parser.add_argument("--num-examples", type=int, default=5000)
    parser.add_argument(
        "--word-lengths",
        default="4-6",
        help="Word lengths to include, e.g. 4-6 or 4,5.",
    )
    parser.add_argument(
        "--move-counts",
        default="3-4",
        help="Move counts/max moves to include, e.g. 3-4 or 3,4,5.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10000,
        help="Use the top-N most frequent English words (via wordfreq). "
        "Smaller N = more common words but smaller graph; larger N = more obscure words.",
    )
    parser.add_argument(
        "--no-filter-proper-nouns",
        action="store_true",
        help="Disable filtering of personal first names (NLTK) and place names "
        "(countries / major cities / US states). Default is to filter them out.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--attempts-per-example",
        type=int,
        default=200,
        help="Random-walk attempts before giving up on a target length/move combo.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the generated DatasetDict to the Hugging Face Hub.",
    )
    parser.add_argument(
        "--repo-id",
        help="Hugging Face dataset repo id, e.g. immortal3/wordmaze. "
        "Different --move-counts get pushed as different configs (variants) "
        "to the SAME repo, loadable via load_dataset(repo, 'm3-4').",
    )
    parser.add_argument(
        "--config-name",
        default=None,
        help="HF dataset config (variant) name. Defaults to the move slug, e.g. m3-4.",
    )
    parser.add_argument("--private", action="store_true", help="Create/upload as a private Hub dataset.")
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"),
        help="Hugging Face token. Defaults to HF_TOKEN, then HUGGINGFACE_TOKEN, "
        "then cached huggingface-cli login.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.num_examples <= 0:
        parser.error("--num-examples must be positive")
    if args.push_to_hub and not args.repo_id:
        parser.error("--repo-id is required with --push-to-hub")

    word_lengths = parse_int_range(args.word_lengths, "--word-lengths")
    move_counts = parse_int_range(args.move_counts, "--move-counts")
    if any(length < 2 for length in word_lengths):
        parser.error("--word-lengths must be >= 2")
    if any(move_count < 1 for move_count in move_counts):
        parser.error("--move-counts must be >= 1")

    move_slug = move_count_slug(move_counts)
    move_range = (
        f"{min(move_counts)}-{max(move_counts)}"
        if len(move_counts) > 1
        else str(move_counts[0])
    )

    config = GenerationConfig(
        num_examples=args.num_examples,
        word_lengths=word_lengths,
        move_counts=move_counts,
        seed=args.seed,
        attempts_per_example=args.attempts_per_example,
    )

    filter_pn = not args.no_filter_proper_nouns
    print(
        f"Loading top {args.top_n} common English words from wordfreq "
        f"(filter_proper_nouns={filter_pn})"
    )
    word_source = load_common_word_source(args.top_n, filter_proper_nouns=filter_pn)
    words_by_length = load_words(word_source, word_lengths)
    examples, stats = generate_examples(words_by_length, config)
    splits = split_examples(examples, args.train_ratio, args.validation_ratio)

    split_counts = {split_name: len(rows) for split_name, rows in splits.items()}
    stats["splits"] = split_counts
    stats["source_word_file"] = (
        f"wordfreq:top-{args.top_n}"
        + ("" if not filter_pn else "+filter-proper-nouns")
    )
    stats["move_count_slug"] = move_slug
    stats["move_count_range"] = move_range

    out_dir = Path(
        args.out_dir or Path(__file__).parent / "data" / f"wordmaze-{move_slug}"
    )
    write_outputs(
        splits=splits,
        stats=stats,
        out_dir=out_dir,
        repo_id=args.repo_id,
        push_to_hub=args.push_to_hub,
        private=args.private,
        token=args.hf_token,
        config_name=args.config_name or move_slug,
    )

    print(f"Wrote dataset to {out_dir}")
    print(json.dumps(split_counts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
