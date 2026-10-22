# Mini-Architecture

微型 DeepSeek 风格 MoE LLM 玩具项目：用 20 条训练数据、约 45 秒训练，研究
DeepSeek V4.1 的最小可复现架构。纯文本（无 vision、无 fp8 量化）。

架构只写一份（`model_v41.py`），训练/对话脚本都 import 它，不存在复制粘贴的
第二份实现。本体 3.56M 参数（含 embedding+head 总 36.7M，大头是 vocab
129280 的输入输出层）。

## 快速开始

```bash
.venv/bin/python train_v41.py    # 训练：20 条数据 × 300 epochs，~45s（CUDA）
.venv/bin/python chat_v41.py     # 交互对话 + 专家使用/indexer 密度诊断
```

自定义架构类在**模块导入时注册**（`model_type = "deepseek_v41_mini"`），用
`AutoModelForCausalLM.from_pretrained` 加载前必须先 `import model_v41`；
`chat_v41.py` 已处理。模型产物（`./my_deepseek_v41_mini`）不入库。

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

## 测试

无 pytest，两个直接跑的验证脚本：

```bash
.venv/bin/python test/smoke_v41.py    # 冒烟：causality 断言（改未来 token →
                                      # 前缀 logits diff < 1e-4）+ S∈{1,3,7,48,49}
                                      # 前向/反向有限 + 诊断 API（CPU 可跑）
.venv/bin/python test/moe_eq_test.py  # MoE：Triton vs einsum 参考路径等价
                                      # （前向逐位 + 全参数梯度，fp32/bf16，需 CUDA）
```

改模型后先跑冒烟再训练；改 MoE 还要跑等价测试。

## 验收基线

train loss 15.36 → **0.0056**（44.5s），4 条训练内 prompt greedy 全命中，
indexer 密度 ~0.31。

## 目录

| 路径 | 说明 |
|---|---|
| `model_v41.py` | V4.1 架构唯一源（config + 模型，注册 `deepseek_v41_mini`）|
| `train_v41.py` | 训练 + 保存 + 重新加载 greedy 验证 |
| `chat_v41.py` | 对话 + 专家使用/indexer 密度诊断 |
| `data_v41.py` | 20 条训练样本（`问：… 答：…`，「答」前半角空格分隔）|
| `test/` | 冒烟测试 + MoE Triton 等价测试 |
| `tokenizer/` | DeepSeek 系 tokenizer（`PreTrainedTokenizerFast`，vocab 129280）|
| `legacy/` | 第一代实现（只读对照，见 `legacy/README.md`）|

## 环境

- Python 3.12 + torch 2.14（CUDA 13）+ transformers 5.20，venv 由 uv 构建
- triton 随 torch 自带（MoE grouped GEMM 分发用，无需另装）
- 无 requirements.txt：依赖以 `.venv` 为准，缺包用 uv 装

## 协议

[MIT](./LICENSE)
