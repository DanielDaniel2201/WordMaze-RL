"""Pure-Python verifier and binary reward for WordMaze model outputs.

Run the self-check with: uv run verifier.py
"""
# /// script
# requires-python = ">=3.11"
# dependencies = ["wordfreq==3.1.1"]
# ///
from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from functools import lru_cache

from wordfreq import word_frequency

from create_dataset import changed_index


CHECK_NAMES = (
    "valid_format",
    "starts_correctly",
    "reaches_goal",
    "within_max_moves",
    "exact_moves",
    "one_letter_changes",
    "all_valid_words",
    "correct_word_length",
    "password_matches",
)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


@lru_cache(maxsize=None)
def is_valid_english_word(word: str) -> bool:
    """Match the blog's verifier policy: accept words known to wordfreq."""
    return word_frequency(word, "en") > 0


def parse_answer(completion: str) -> list[str] | None:
    """Extract exactly one answer tag containing two or more equal-length words."""
    if not isinstance(completion, str):
        return None

    matches = ANSWER_RE.findall(completion)
    if len(matches) != 1:
        return None

    words = [word.strip().lower() for word in matches[0].split("->")]
    if (
        len(words) < 2
        or any(re.fullmatch(r"[a-z]+", word) is None for word in words)
        or len({len(word) for word in words}) != 1
    ):
        return None
    return words


def verify(
    completion: str,
    puzzle: Mapping[str, object],
    *,
    valid_word: Callable[[str], bool] = is_valid_english_word,
) -> dict[str, bool]:
    """Return the WordMaze checks without consulting the reference solution."""
    checks = dict.fromkeys(CHECK_NAMES, False)
    words = parse_answer(completion)
    if words is None:
        return {**checks, "fully_valid": False}

    start = str(puzzle["start"]).lower()
    goal = str(puzzle["goal"]).lower()
    word_length = int(puzzle["word_length"])
    max_moves = int(puzzle["max_moves"])
    expected_password = str(puzzle["password"]).upper()
    moves = len(words) - 1
    changed = [changed_index(left, right) for left, right in zip(words, words[1:])]

    checks.update(
        valid_format=True,
        starts_correctly=words[0] == start,
        reaches_goal=words[-1] == goal,
        within_max_moves=moves <= max_moves,
        exact_moves=moves == max_moves,
        one_letter_changes=all(index is not None for index in changed),
        all_valid_words=all(valid_word(word) for word in words),
        correct_word_length=all(len(word) == word_length for word in words),
    )
    if checks["one_letter_changes"]:
        actual_password = "".join(
            "F" if right[index] > left[index] else "B"
            for left, right, index in zip(words, words[1:], changed)
        )
        checks["password_matches"] = actual_password == expected_password

    return {**checks, "fully_valid": all(checks.values())}


def reward(completion: str, puzzle: Mapping[str, object]) -> float:
    """Binary RLVR reward."""
    return float(verify(completion, puzzle)["fully_valid"])


def _self_check() -> None:
    puzzle = {
        "start": "cold",
        "goal": "warm",
        "word_length": 4,
        "max_moves": 4,
        "password": "FFBF",
    }
    vocabulary = {"cold", "cord", "word", "ward", "warm"}
    known = vocabulary.__contains__
    answer = "<answer>cold -> cord -> word -> ward -> warm</answer>"

    assert verify(answer, puzzle, valid_word=known)["fully_valid"]
    assert not verify("<answer>cold -> warm</answer>", puzzle, valid_word=known)["fully_valid"]
    assert not verify(answer, {**puzzle, "password": "FFFF"}, valid_word=known)["fully_valid"]
    assert not verify(answer, puzzle, valid_word={"cold"}.__contains__)["all_valid_words"]
    assert not verify("no answer tag", puzzle, valid_word=known)["valid_format"]
    print("verifier self-check passed")


if __name__ == "__main__":
    _self_check()
