# AGENTS.md

微型 DeepSeek 风格 MoE LLM 玩具项目，无包结构、无测试、无 CI、无 requirements.txt。除下述陷阱外没有隐藏约定。

布局：根目录 `*_v41.py` 是 **V4.1 最小架构（研究主线）**；`legacy/` 是第一代 `deepseek_pro_mini` 旧版（只读对照，详见 `legacy/README.md`）。

## 环境与运行

- 用项目自带 venv 运行：`.venv/bin/python`（uv 建的，Python 3.12.3、torch 2.14.1+cu130、transformers 5.20.2，CUDA 已验证可用）。系统 python3 没有 torch。
- 无参数解析。主线：`.venv/bin/python train_v41.py` → `.venv/bin/python chat_v41.py`。
- 旧版（**必须从仓库根目录执行**，脚本内路径全按 CWD 解析）：
  `.venv/bin/python legacy/train.py` → `.venv/bin/python legacy/chat.py`。
- tokenizer 统一从 `tokenizer/` 子目录加载（vocab 129280）。模型产物不入库。

## 文件分工

- `model_v41.py` / `train_v41.py` / `chat_v41.py` / `data_v41.py` — V4.1 主线，见下节
- `tokenizer/` — DeepSeek 系 tokenizer（`PreTrainedTokenizerFast`，vocab 129280）
- `legacy/` — 旧版三件套 `train.py`（训练→`./my_deepseek_pro_model`）/ `chat.py`（对话+热力图）/ `data.py`（100 条问答），以及 `README.md` 说明

## V4.1 最小架构（`*_v41` 文件，研究主线）

复现官方 deepseek_v41 config 的最小版（无 vision、无 fp8 量化），**架构只写一份**：
`model_v41.py` 是唯一源，`train_v41.py`/`chat_v41.py` 都 import 它（不复制粘贴）。
`model_type = "deepseek_v41_mini"`，用 `AutoModelForCausalLM.from_pretrained`
加载前必须先 `import model_v41`（注册），否则 KeyError。

- 命令：`.venv/bin/python train_v41.py`（20 条数据，300 epochs ~48s，产出
  `./my_deepseek_v41_mini`）→ `.venv/bin/python chat_v41.py`（交互 + 专家/indexer 诊断）
- 已实现组件：MLA（q/kv/o 低秩 + 解耦 RoPE + o 分组）、MoE（sqrtsoftplus +
  noaux_tc 均衡偏置 + norm_topk）、稀疏注意力（滑窗16 ∪ topk8 token ∪ topk4 block）、
  Hyper-Connections（4 路残差流 + Sinkhorn）、跨层 KV 复用（source 层 2/6 + 池化压缩）
- 验收基线：train loss 0.0089，4 条训练内 prompt greedy 全命中，indexer 密度 ~0.30
- `model_v41.py` 内 [INFERRED] 标注处为按命名推断（无官方源码）；v2 字段
  `num_nextn_predict_layers`/`engram_*`/`dspark_*` 只在 config 占位，传非空不生效
- 修 meta 设备陷阱：`from_pretrained` 会把普通属性留在 meta 设备，
  `MiniMoE._ensure_counts()` 懒重建统计张量——别直接碰 `_sel_counts`
- 冒烟方法：causality 断言（改未来 token → 前缀 logits diff < 1e-4）+
  S ∈ {1,3,7,48,49} 前向有限，改模型后先跑这个再训练

## 陷阱（改动前必读）

1. **旧版架构在 `legacy/train.py` 和 `legacy/chat.py` 里整段复制**（约 150 行：`MiniDeepSeekConfig` / `BoundedSwiGLU` / `DeepSeekMiniMoE` / `MiniMLA` / `DeepSeekProBlock` / `ToyDeepSeekProForCausalLM`）。改架构必须**两个文件同步改**，否则 chat.py 加载不了训练产物。唯一有意差异：train.py 的 forward 计算 loss，chat.py 的固定返回 `loss=None`。
2. **不要让 `legacy/chat.py` 改成 import `legacy/train.py`**：train.py 在模块级（非 `__main__` 块内）就加载 tokenizer、构建 GPU 数据集——import 会触发整条训练前置流程。当前的复制粘贴是刻意的自包含。
3. **`legacy/train.py` 实际只能在 CUDA 上跑**：虽注释称硬件自适应，但无条件用 `torch.cuda.Event` 计时 + `torch.cuda.synchronize()`，非 CUDA 机器直接崩。`legacy/chat.py` 的 cpu/mps 回退是真实的。
4. **数据格式耦合**（新旧版同）：用 `text.split(" 答：")`（「答」前有一个半角空格）切分问答。往 `data.py`/`data_v41.py` 加样本必须保持这个精确分隔符，否则解包报错。Prompt 段 label 置 -100（只学答案），序列截断到 48 token。
5. HF 自定义类在**模块导入时注册**，这是 `from_pretrained` 能识别架构的前提：旧版是 `deepseek_pro_mini`（legacy 两脚本各注册一遍、独立进程，别合并），主线是 `deepseek_v41_mini`（只在 `model_v41.py` 注册）。

## 旧版模型速览（`legacy/`，仅对照）

tiny 尺寸（hidden 128、2 层、vocab 由 tokenizer 决定）：MLA 式低秩 q/kv/o 投影注意力（**无 RoPE**，就是缩放点积因果注意力）+ MoE FFN（4 个 routed 专家取 top-2，输出乘 `routed_scaling_factor=2.5`，外加 1 个 shared 专家）+ `BoundedSwiGLU`（激活 clamp 到 ±10）+ RMSNorm。`model_type = "deepseek_pro_mini"`。chat.py 生成为贪心 argmax，每轮最多 30 token，无采样参数。
