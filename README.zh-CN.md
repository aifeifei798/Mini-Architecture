# Mini-Architecture

[English](README.md) | [简体中文](README.zh-CN.md)

微型 DeepSeek 风格 MoE LLM 玩具项目：用 20 条训练数据、约 45 秒训练，研究
DeepSeek V4.1 的最小可复现架构。纯文本（无 vision、无 fp8 量化）。

架构只写一份（`model_v41.py`），训练/对话脚本都 import 它，不存在复制粘贴的
第二份实现。本体 3.56M 参数（含 embedding+head 总 36.7M，大头是 vocab
129280 的输入输出层）。

## 快速开始

**v2 · 20 条记忆基线**（架构可训练性）：

```bash
.venv/bin/python train_v41.py    # 训练（20 条数据，300 epochs，~45s，CUDA）
.venv/bin/python chat_v41.py     # 交互对话 + 专家/indexer 诊断热力图
```

**v3 · 长上下文**（真实语料，S=512，需 `./minimind_dataset/*.jsonl`）：

```bash
.venv/bin/python data_long.py    # jsonl → token 化打包 .bin（~1 分钟）
.venv/bin/python train_long.py   # 预训练 5000 步 + SFT 3000 步，~33 分钟
.venv/bin/python ablate_long.py  # 6 配置同预算 ablation，~2 小时（可选）
MODEL_DIR=./my_deepseek_v41_long .venv/bin/python chat_v41.py
```

加载产物前必须先 `import model_v41`（注册 `deepseek_v41_mini` 架构），
`chat_v41.py` 已处理。模型产物（`./my_deepseek_v41_*`）不入库。

## 架构要点（V4.1 最小版）

- **mHC（Hyper-Connections）**：每子层（attn/ffn）各一份，流 = 4 份 embedding
  拷贝，Sinkhorn 双随机混合旧流，末端 HyperHead 收拢；块内无残差
- **注意力 MiniMLA**：q 低秩 + 双 norm；K=V 共用单头直投（V4 弃用 V3 的
  kv_lora）；部分 interleaved RoPE 只旋尾部 rope 维，输出共轭旋转（-sin）；
  每头可学习 attention sink；o 分组低秩
- **稀疏注意力**：indexer 打分 ∑_h w_h·ReLU(q·K)·d^-0.5，indexer 跨层共享
  （L2/L6 为源）
- **MoE**：8 路由专家 top-2 + 1 共享专家；sqrtsoftplus 打分，修正 bias 只参与
  选择（官方 noaux_tc 语义）；SwiGLU clamp 在激活前。专家分发为 **Triton
  grouped GEMM**（路由行 argsort+bincount 成 CSR 段，前后向都是 kernel，
  无 Python 循环），CPU/无 triton 自动回退 einsum 参考实现
- **跨层 KV 压缩复用** [INFERRED]、**MTP** [INFERRED]（fuse(h_t, emb(t+1))
  预测 t+2，loss 系数 0.3）、**engram** [INFERRED]（n-gram 哈希查表注入）
- `dspark_*` 仅占位未实现

完整对齐说明见 `model_v41.py` 顶部 docstring；工程约定与陷阱见 `AGENTS.md`。

## v3 长上下文（真实语料 + 稀疏注意力真正生效）

数据来自 [MiniMind](https://github.com/jingyaogong/minimind) 的开源数据集
（`pretrain_t2t_mini.jsonl` 纯文本 / `sft_t2t_mini.jsonl` 多轮对话），
reservoir 采样后 eos 打包成 S=512 定长窗口（pretrain 18.4M + sft 12.2M token）。

**基线**（5000 步 ≈ 41M token 预训练 + 3000 步 SFT，~33 分钟）：

| 阶段 | best val | ppl |
|---|---|---|
| 预训练（pretrain val）| 5.369 | 215 |
| SFT（sft val）| 5.009 | 150 |

S=512 下 indexer 密度 ~0.065（S=48 时 0.4）——每查询只"看见"约 33/512 位置，
稀疏注意力在做真实的块选择。

**架构 ablation**（同预算 5000 步、固定 val 批次，`ablate_long.py`）：

| 配置 | 变更 | best val | indexer 密度 |
|---|---|---|---|
| dense | 索引全选（≈全注意力上界）| **5.348** | 0.501 |
| idx32 | index_topk 8→32 | 5.545 | 0.076 |
| base | 默认（index_topk=8）| 5.648 | 0.042 |
| no_engram | 关 engram（省 1.09M 参数）| 5.701 | 0.041 |
| idx4 | index_topk 8→4 | 5.733 | 0.037 |
| no_reuse | 关跨层 KV 复用 | 5.936 | 0.043 |

（ablation 跑在滑窗修复前的语义上，各配置间仍同语义可比。）

三个结论：
1. **稀疏-质量权衡单调**：密度 4%→全量换 0.30 loss；top-8 用 1/12 的可见
   token 换 90% 的质量
2. **跨层 KV 复用是最大单项**：关掉差 0.29 loss——它不只是省 KV，还是
   跨层表征共享的正则/深度增益
3. **engram 划算**：3% 的参数（1.09M）换 0.05 loss

**v3.1 滑窗修复**：`SparseIndexer` 的滑窗差值曾误写成 `j-i`，`& causal`
后整个 sliding_window=16 退化为"只看自己"；修复为 `i-j` 回看语义后
pretrain val 5.65→5.37。修复前的旧 checkpoint 与新语义不匹配，已全部重训。

**KV cache 增量解码**：`prefill` 缓存各层 indexer 的 k 投影与 own 层 raw KV，
每 token 一步 O(S)（indexer 只算新查询一行，与全量前向逐位同语义，贪心
生成 token 完全一致）。注意：S≤512 且 GPU 较强时两条路径都被 kernel
launch 次数主导，实测加速仅 ~1.1×；收益（O(n·S²)→O(n·S)）在更长序列
或更大 batch 下才体现。

## 测试

无 pytest，两个直接跑的验证脚本：

```bash
.venv/bin/python test/smoke_v41.py    # 冒烟：causality 断言（改未来 token →
                                      # 前缀 logits diff < 1e-4）+ S∈{1,3,7,48,49}
                                      # 前向/反向有限 + 诊断 API（CPU 可跑）
.venv/bin/python test/moe_eq_test.py  # MoE：Triton vs einsum 参考路径等价
                                      # （前向逐位 + 全参数梯度，fp32/bf16，需 CUDA）
.venv/bin/python test/kv_cache_test.py # KV cache：增量解码 vs 全量前向贪心
                                       # token 一致 + 计时（需 CUDA + long checkpoint）
```

改模型后先跑冒烟再训练；改 MoE 还要跑等价测试；改注意力/缓存路径跑 kv 测试。

## 验收基线

- **v2（20 条记忆）**：train loss 0.0056（44.5s），4 条 prompt greedy 全命中
- **v3（长上下文）**：pretrain val 5.369（ppl 215）→ SFT val 5.009（ppl 150）

## 目录

| 路径 | 说明 |
|---|---|
| `model_v41.py` | V4.1 架构唯一源（config + 模型 + KV cache，注册 `deepseek_v41_mini`）|
| `train_v41.py` | v2 训练：20 条记忆基线 + 保存 + greedy 验证 |
| `chat_v41.py` | 对话（增量解码）+ 专家/indexer 诊断，`MODEL_DIR` 环境变量切 checkpoint |
| `data_v41.py` | 20 条训练样本（`问：… 答：…`，「答」前半角空格分隔）|
| `data_long.py` | v3 数据管道：minimind jsonl → reservoir 采样 → S=512 打包 .bin |
| `train_long.py` | v3 训练：随机切窗全标签 LM，预训练→SFT 两阶段（各自 val 门控）|
| `ablate_long.py` | 6 配置同预算 ablation（KV 复用/稀疏度/engram），results.json 断点续跑 |
| `test/` | 冒烟 + MoE Triton 等价 + KV cache 一致性测试 |
| `tokenizer/` | DeepSeek 系 tokenizer（`PreTrainedTokenizerFast`，vocab 129280）|

## 环境

- Python 3.12 + torch 2.14（CUDA 13）+ transformers 5.20，venv 由 uv 构建
- triton 随 torch 自带（MoE grouped GEMM 分发用，无需另装）
- 无 requirements.txt：依赖以 `.venv` 为准，缺包用 uv 装

## 协议

[MIT](./LICENSE)
