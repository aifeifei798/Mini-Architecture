# legacy — 第一代 `deepseek_pro_mini`（旧版）

**这是研究起点，已被根目录的 V4.1 最小架构（`*_v41.py`）取代，不再维护。**
保留原因：对照验证 + 回退参考。新工作请去仓库根目录看 `model_v41.py`。

| 文件 | 说明 |
|---|---|
| `train.py` | 训练 → `./my_deepseek_pro_model` + 4 条样例验证 |
| `chat.py` | 加载上述产物，交互对话 + MoE 专家热力图 |
| `data.py` | 100 条问答（`raw_data`），格式 `"问：xxx 答：yyy"` |

## 与 V4.1 版的差异（关键）

- `model_type = "deepseek_pro_mini"`（V4.1 版是 `deepseek_v41_mini`），**权重互不兼容**
- 架构**复制粘贴**在 train.py 和 chat.py 里（改要两处同步）；V4.1 版收敛为单一 `model_v41.py`
- 注意力是无 RoPE 的缩放点积；V4.1 版有解耦 RoPE、稀疏 indexer、Hyper-Connections、跨层 KV 复用
- `routed_scaling_factor=2.5`、4 专家；V4.1 版是 1.5、8 专家 + noaux_tc 均衡偏置

## 运行（从**仓库根目录**跑，路径是 CWD 相对的）

```bash
.venv/bin/python legacy/train.py    # 产出 ./my_deepseek_pro_model
.venv/bin/python legacy/chat.py     # 需先训练
```

脚本内 `from data import raw_data` 靠 Python 的脚本目录入 sys.path 生效，
`./tokenizer`、`./my_deepseek_pro_model` 靠 CWD 解析——所以必须在根目录执行。

## 已知坑（原样继承自 AGENTS.md）

1. 架构整段复制在两个文件里，改架构必须同步改，否则加载失败
2. 别让 chat.py 去 import train.py：train.py 模块级就加载 tokenizer、建 GPU 数据集
3. train.py 的 `torch.cuda.Event` 计时无条件执行，**只能在 CUDA 机器上跑**
4. 数据切分靠精确的 `" 答："`（含半角空格）；prompt 段 label=-100，序列截断 48 token
