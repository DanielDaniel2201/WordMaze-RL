# WordMaze GRPO 学习计划

目标：用同一套 prompt、verifier 和 test split，对比 `Qwen/Qwen3.5-4B` 在 GRPO 前后的 WordMaze 表现。Frontier 模型只用于确认任务可解，不作为训练对象。

## 平台分工

| 平台 | 做什么 |
| --- | --- |
| Windows 本机 | 写代码、验证数据、跑 verifier、分析结果、画图 |
| Gemini API | 跑一次 frontier baseline |
| RunPod Linux Pod | 跑 Qwen baseline、rollout pilot 和 GRPO |
| Hugging Face | 下载数据/模型；可选保存私有 LoRA adapter |

## 1. Windows：冻结数据和 verifier

- [x] 保留原始 `m3-4`、`m4-6` Parquet，不修改 test 数据。
- [x] 用 `verifier.py` 实现格式、起点、终点、步数、单字母变化、词典和密码检查。
- [x] 运行 `uv run verifier.py`，内置自检通过。
- [x] verifier 检查六个 split 的全部 4,000 条参考答案：4,000 通过，0 失败。
- [x] 当前 dataset Git revision：`9183e34fdc425dc90960905be9a47776f1079288`；verifier 固定 `wordfreq==3.1.1`。

数据边界：两个 config 单独使用时，各自的 train/validation/test 没有重复；跨 config 有重复。若使用两个 config，训练时排除两个 config 的全部 validation/test，并按 `(start, goal, password, word_length, max_moves)` 去重。当前清洗结果应为：

- `m3-4/train`: 1600 -> 1578
- `m4-6/train`: 1600 -> 1580
- 合并训练集再次去重：3076

完成条件：所有参考答案通过，故意构造的错误答案失败。

## 2. Windows：实现统一评测入口

- [x] 写 `evaluate.py`，用 vLLM 跑 Qwen base/LoRA，并复用同一个 verifier。
- [x] 每题保存 JSONL：`id`、原始 completion、所有 checks、score、输出 token 和是否截断。
- [x] 通过命令行参数记录 model revision、thinking mode、temperature、top-p、seed 和最大输出长度。
- [ ] 先在 validation 跑 10～20 题，不用 test 调参数。

主要指标：`fully_valid` pass@1。辅助指标：每个 check 的通过率、按 `max_moves`/`word_length` 分组的准确率、输出长度和截断率。

完成条件：同一份保存的 completion 可以重复离线评分，并得到相同结果。

## 3. Gemini API：frontier baseline

- [ ] 使用一个固定模型版本，优先复现博客的 `gemini-3.1-pro-preview`。
- [ ] 先跑 validation 10 题确认接口。
- [ ] 正式跑 `m4-6/test` 200 题。
- [ ] 保存原始输出到 `results/frontier/`，回 Windows 离线评分。

目的只是确认任务与 verifier 正常。设置约 15 美元的 API 支出提醒，避免 thinking tokens 失控。

## 4. RunPod：Qwen base baseline

建议租 48GB A40 或 A6000，使用官方 PyTorch 模板和至少 50GB 持久存储。Token 只放环境变量，不提交到 Git。

- [ ] 拉取本仓库。
- [ ] 安装锁定版本的 PyTorch、Transformers、vLLM、TRL、PEFT、Datasets、wordfreq。
- [ ] 启动 `Qwen/Qwen3.5-4B`，先回答 1 题，再跑 validation 20 题。
- [ ] 用完全固定的推理配置跑 `m3-4/test` 和 `m4-6/test`。
- [ ] 保存到 `results/base/`，下载结果后关闭 Pod。

完成条件：400 题都有原始输出，无意外截断，Windows 可以重新评分。

## 5. RunPod：训练前 rollout pilot

- [ ] 从清洗后的 `m3-4/train` 取 50 题。
- [ ] 每题采样 8 个回答，只评分，不更新模型。
- [ ] 统计全 0 group、同时包含 0/1 的 group、completion 长度和截断率。
- [ ] 人工检查成功样本，确认没有利用 verifier 漏洞。

如果几乎所有 group 都是 0：先增加最大输出长度，再把每题采样数从 8 提到 16；仍无信号才考虑更简单的数据。不要先添加部分奖励。

完成条件：有足够的组内 reward 差异供 GRPO 学习。

## 6. RunPod：第一次 GRPO

第一轮只做最小实验（`train_grpo.py` 已实现）：

- 模型：`Qwen/Qwen3.5-4B`
- 数据：清洗后的 `m3-4/train`
- 方法：TRL `GRPOTrainer` + LoRA + colocated vLLM
- reward：仅 `float(fully_valid)`
- `num_generations`: 8
- `max_completion_length`: 先 1024，根据 pilot 调整
- 训练步数：先 100
- 每 25 steps 保存并跑 validation

监控 reward mean/std、zero-variance group 比例、completion 长度、截断率、KL 和 validation `fully_valid`。只按 validation 选择 checkpoint。

显存不足时依次减 batch、减输出长度、开 gradient checkpointing、开 vLLM sleep mode；最后才换更贵的 GPU。

完成条件：训练正常结束，LoRA、配置、日志和选中 checkpoint 均已保存。

## 7. 同协议复测和结论

- [ ] 用 baseline 完全相同的推理设置评测选中的 LoRA。
- [ ] 分别跑 `m3-4/test`、`m4-6/test`。
- [ ] 按题比较 base 与 GRPO，而不只比较两个平均数。
- [ ] 报告提升、持平或退化；检查 reward hacking 和跨难度泛化。

最终结果表：

| 模型 | m3-4 fully_valid | m4-6 fully_valid | 单字母变化 | 密码匹配 | 截断率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemini frontier | - |  |  |  |  |
| Qwen base |  |  |  |  |  |
| Qwen GRPO |  |  |  |  |  |

## 预算护栏

- Gemini API：15 美元上限。
- Qwen baseline：3 美元上限。
- rollout pilot：3 美元上限。
- 第一次 GRPO：10 美元上限。
- 第二次训练只有在第一次结果分析完成后再决定。

暂不做：9B、全参数训练、verl、多 GPU、复杂 dense reward。4B LoRA 闭环跑通后再考虑。
