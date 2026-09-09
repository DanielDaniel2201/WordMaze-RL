# RunPod baseline 排障记录

记录日期：2026-09-09

当前目标是在 RunPod A40 上使用 vLLM 运行 `Qwen/Qwen3.5-4B` baseline。以下问题按实际出现顺序记录。

## 1. CUDA 在 fork 子进程中重复初始化

错误：

```text
RuntimeError: Cannot re-initialize CUDA in forked subprocess; use the 'spawn' start method
```

根因：vLLM worker 使用 `fork` 启动，而主进程已经初始化 CUDA。

处理：在 `run_baseline.sh` 中设置：

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn
```

## 2. PyTorch CUDA 版本高于主机驱动能力

错误：

```text
The NVIDIA driver on your system is too old (found version 12080)
```

现场版本：

```text
GPU: NVIDIA A40
Driver: 570.195.03
torch: 2.11.0+cu130
CUDA runtime: 13.0
vLLM: 0.26.0
```

根因不是 A40，也不是驱动真的需要在容器内更新，而是依赖解析安装了 CUDA 13.0 版 PyTorch；当前 Pod 提供的是 CUDA 12.8 兼容驱动。

处理：安装时显式指定 `--torch-backend=cu128`，并在运行前断言 `torch.version.cuda == "12.8"`。

## 3. vLLM 0.26.0 与 CUDA 12.8 依赖无法解析

错误：

```text
vllm==0.26.0 depends on torchcodec>=0.14
only torchcodec<=0.11.1+cu128 is available
```

根因：vLLM 0.26.0 的 `torchcodec` 要求在所选 CUDA 12.8 软件源中无法满足。

处理：固定为仍支持 Qwen3.5 的 `vllm==0.24.0`。

## 4. `/root` 容器盘空间耗尽

错误：

```text
No space left on device (os error 28)
```

根因：20GB 容器盘同时保存 uv 下载缓存和 Python 虚拟环境，安装约 203 个依赖时被占满。

处理：清理失败的 `/root/.cache/wordmaze-venv` 和 uv 缓存；脚本随后把 uv 缓存及虚拟环境移至 `/workspace/cache`。

## 5. `/workspace` 网络卷配额耗尽

错误：

```text
Disk quota exceeded (os error 122)
```

根因：挂载的网络卷个人配额不足。`df -h /workspace` 显示的数百 TB 是共享存储池剩余空间，不代表当前账户或网络卷的可用配额。

这也曾导致 `git pull` 无法写入 loose object；不是 Git 分支损坏。

处理：重新创建具有足够持久化磁盘的新 Pod。建议至少预留 60GB 给 `/workspace`，用于 Python 环境、uv 缓存、模型缓存、数据集和结果。

## 非致命提示

```text
You are sending unauthenticated requests to the HF Hub
```

这只是 Hugging Face 未登录提示，不是本次失败原因。公开模型可以自动下载；登录可改善限流和下载稳定性。

## 当前验证状态

- 本地 verifier 自检已通过。
- 依赖解析已经能够完成到安装阶段。
- GPU baseline 尚未成功运行；只有出现 `torch 2.11.0+cu128 CUDA 12.8`、模型成功加载并产生第一条评测结果后，才能视为 smoke test 通过。

