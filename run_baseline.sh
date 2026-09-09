#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/workspace/cache/huggingface}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/workspace/cache/vllm}"

command -v uv >/dev/null || { echo "uv is required" >&2; exit 1; }
git lfs pull
uv run verifier.py

# The first call installs vLLM, downloads Qwen, and proves one inference works.
uv run evaluate.py --config m3-4 --split validation --limit 1
uv run evaluate.py --config m3-4 --split validation --limit 20
uv run evaluate.py --config m3-4 --split test
uv run evaluate.py --config m4-6 --split test
