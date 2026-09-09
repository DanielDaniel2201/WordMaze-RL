#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/workspace/cache/huggingface}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/workspace/cache/vllm}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/cache/uv}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"

command -v uv >/dev/null || { echo "uv is required" >&2; exit 1; }
git lfs pull
venv="${WORDMAZE_VENV:-/workspace/cache/wordmaze-venv}"
uv venv --allow-existing --python 3.11 "$venv"
uv pip install --python "$venv/bin/python" --torch-backend=cu128 \
  --extra-index-url https://wheels.vllm.ai/0.24.0/cu129 \
  "vllm==0.24.0+cu129" "pandas>=2.2" "pyarrow>=17" "wordfreq==3.1.1"
python="$venv/bin/python"
"$python" -c 'import torch; assert torch.version.cuda == "12.8", torch.version.cuda; assert torch.cuda.is_available(); print("torch", torch.__version__, "CUDA", torch.version.cuda)'
"$python" verifier.py

# The first call installs vLLM, downloads Qwen, and proves one inference works.
"$python" evaluate.py --config m3-4 --split validation --limit 1
"$python" evaluate.py --config m3-4 --split validation --limit 20
"$python" evaluate.py --config m3-4 --split test
"$python" evaluate.py --config m4-6 --split test
