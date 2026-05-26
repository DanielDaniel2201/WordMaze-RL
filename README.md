---
dataset_info:
- config_name: m4-6
  features:
  - name: id
    dtype: string
  - name: start
    dtype: string
  - name: goal
    dtype: string
  - name: word_length
    dtype: int64
  - name: max_moves
    dtype: int64
  - name: password
    dtype: string
  - name: prompt
    dtype: string
  - name: messages
    list:
    - name: role
      dtype: string
    - name: content
      dtype: string
  - name: solution
    dtype: string
  - name: answer
    dtype: string
  - name: path
    list: string
  - name: num_moves
    dtype: int64
  splits:
  - name: train
    num_bytes: 1861091
    num_examples: 1600
  - name: validation
    num_bytes: 232282
    num_examples: 200
  - name: test
    num_bytes: 232588
    num_examples: 200
  download_size: 1975228
  dataset_size: 2325961
configs:
- config_name: m4-6
  data_files:
  - split: train
    path: m4-6/train-*
  - split: validation
    path: m4-6/validation-*
  - split: test
    path: m4-6/test-*
---
# Wordmaze

Synthetic word-ladder puzzles with an extra constraint: a **password** you have to satisfy along the way.

I built this for RL experiments (GRPO on small LMs) after running Wordle-style tasks and wanting something with the same "one shot, easy to grade" feel, but a bit more planning-heavy. Each row is a puzzle plus a known solution path, so you can train or eval without human labels.

## The puzzle

You get a **start** word and a **goal** word. Build a path:

1. Change **exactly one letter** per step.
2. Every word must be a real English word of the given length (same length as start/goal).
3. Reach the goal in **exactly** `max_moves` steps (so the path has `max_moves + 1` words).
4. Match the **password**: for each step, look at the letter that changed. If it moved **forward** in the alphabet (e.g. `l` → `r`), record `F`. If **backward** (e.g. `o` → `a`), record `B`. Concatenate those into a string; it must equal the puzzle's `password`.

Classic example from the generator:

```text
start: cold
goal: warm
word_length: 4
max_moves: 4
password: FFBF
```

```text
cold -> cord -> word -> ward -> warm
       F      F      B      F
```

The model is asked to reply with only:

```xml
<answer>cold -> cord -> word -> ward -> warm</answer>
```

That format is intentional. Parsing `<answer>...</answer>` and splitting on ` -> ` is enough for a strict verifier.

## What's in this release (`wordmaze-m4-6`)

| | |
|---|---|
| **Examples** | 2,000 |
| **Splits** | train 1,600 · validation 200 · test 200 |
| **Word lengths** | 4, 5, and 6 letters (balanced across combos) |
| **Path length** | 4, 5, or 6 moves per puzzle |
| **Seed** | 42 (reproducible with `create_dataset.py`) |
| **Vocabulary** | [wordfreq](https://github.com/rspeer/wordfreq) English top 10,000, lowercased a–z only, with a blocklist of common proper nouns (places, names) |

Roughly ~222 examples per `(word_length, max_moves)` pair; see `data/wordmaze-m4-6/metadata.json` for exact counts and generation failure stats.

**Important:** The word list is **not** censored. It follows frequency data, so some start/goal pairs may be offensive. Filter at load time if that matters for your use case.

## Fields

| Field | Description |
|-------|-------------|
| `id` | Stable id, e.g. `wordmaze-42-00042` |
| `start` | Start word |
| `goal` | Goal word |
| `word_length` | Required length for every word in the path |
| `max_moves` | Number of one-letter transitions (path length − 1) |
| `password` | Required F/B pattern for the solution path |
| `prompt` | Plain-text user turn (`start:` / `goal:` / …) |
| `messages` | Chat format: system rules + user prompt (ready for instruct models) |
| `solution` | Reference path, words joined by ` -> ` |
| `answer` | Same path wrapped in `<answer>...</answer>` |
| `path` | Solution as a list of strings |
| `num_moves` | Same as `max_moves` for generated rows |

The bundled `solution` / `answer` / `path` are **one** valid path found during generation. Other paths may exist for the same `(start, goal, password)`; we do **not** guarantee uniqueness.

## How puzzles were generated

1. Take wordfreq's top 10k English lemmas, drop proper-noun-style tokens.
2. For each target length, build an undirected graph: edge between two words iff they differ by exactly one letter.
3. Sample a random **simple path** of the requested length on that graph (no repeated words).
4. Derive `start`, `goal`, and `password` from that walk.
5. Deduplicate on `(start, goal, password, word_length, max_moves)`.
6. Shuffle, assign ids, split 80% / 10% / 10%.

Six-letter puzzles are harder to sample (many random walks hit dead ends); the metadata `failures` block records how often that happened during the run.

Regenerate or change move counts with `create_dataset.py` in this directory.

## Loading

```python
from datasets import load_dataset

# After publishing on Hugging Face:
ds = load_dataset("YOUR_HF_USERNAME/wordmaze-m4-6")

# Or from local JSONL under data/wordmaze-m4-6/data/:
# ds = load_dataset("json", data_files={
#     "train": "data/wordmaze-m4-6/data/train.jsonl",
#     "validation": "data/wordmaze-m4-6/data/validation.jsonl",
#     "test": "data/wordmaze-m4-6/data/test.jsonl",
# })
```

Minimal eval loop:

```python
row = ds["test"][0]
# Model input: row["messages"] or row["prompt"]
# Grade against row["start"], row["goal"], row["password"], row["word_length"], row["max_moves"]
```

A reference grader (format checks, dictionary membership, password match) lives in `eval/grading.py`. For RL, a simple binary reward works: `1` if the parsed path passes all checks, else `0`. Partial credit on individual checks is possible if you want denser signal.

## What this is good for

- Benchmarking constraint-following on small instruct models (path + password + dictionary).
- GRPO / RLVR-style training where reward comes from a verifier, not a human.
- Studying whether models can "track" a symbolic side constraint (F/B) while doing search in word space.

It is **not** a general NLP benchmark and not tied to a single "correct" human strategy. It's a synthetic graph game.

## Limitations

- Dictionary = wordfreq top 10k, not Scrabble/OED; models trained on other corpora may disagree on validity.
- Password constraint can admit multiple solutions; labels are generator samples, not exhaustive answers.
- Difficulty varies a lot by length (6-letter graphs are sparser).
- No multimodal content, no natural-language story. Just structured prompts.

## License

Add your chosen license on the Hub dataset card. Puzzle **text** is derived from wordfreq's frequency lists; check [wordfreq's license](https://github.com/rspeer/wordfreq) if you redistribute commercially.

## Citation

If you use this dataset, cite the Hugging Face dataset URL you publish. No paper yet. This is a small research artifact from a personal GRPO / word-reasoning stack.
