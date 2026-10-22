# flashcodec

> 把 Transformer 的时序 KV Cache 张量当作**高维多媒体图像信号**来压缩。

传统 LLM 研究把隐空间张量视为孤立标量因子；flashcodec 认为 `(B, H, S, D)`
布局里的 **seq 轴是一幅高维图像的行/列**——相邻 token 隐向量强自相关，
能量集中在低频、差分后熵骤降。于是数字多媒体领域的两大经典压缩理论可以
直接迁移：

| 模式 | 灵感 | 变换 | 性质 | 存储 |
|---|---|---|---|---|
| `mode="jpeg"` | JPEG | 分块正交 1D DCT-II，保留最低 K 个低频系数 | **有损频域** | `K/r`（4×~8× 显存节省） |
| `mode="png"` | PNG | 块首锚点（I-Token）+ 一阶差分 Δ，Top-M 裁剪 | **无损/近无损时域** | 稠密等大；残差稀疏性留给熵编码 |

数学形式（块长 r，操作轴 dim=-2）：

- **JPEG**：$X_{\text{blk}} \in \mathbb{R}^{r \times D}$，$C = M_{\text{enc}} X_{\text{blk}}$，
  $M_{\text{enc}} \in \mathbb{R}^{K \times r}$，$M[j,n] = c_j \cos\frac{\pi}{r}(n+\tfrac12)j$，
  行正交归一（$K=r$ 时逐块无损）；重构 $\hat X = M_{\text{enc}}^\top C$。
- **PNG**：$\Delta_i = x_i - x_{i-1}$（$i \ge 1$），$x_0$ 为全精度锚点；
  重建 $\hat x_i = x_0 + \sum_{j=1}^{i} \Delta_j$（`cumsum`，纯线性、autograd 可微）。
  相邻隐向量自相关 ⇒ $\mathrm{Var}(\Delta)/\mathrm{Var}(X) \approx 0.03$（**30× 方差压缩**，
  实测），Top-M 裁剪后残差近零稀疏。

```
                    ┌────────────────────────────────────────────┐
  (B,H,S,D) ──append──▶  raw_buffer（未满 r 的尾部，原始高保真）  │
                    │        │ 满 r 个 token，批量归档（一次 GEMM）│
                    │        ▼                                    │
                    │  compressed_storage                         │
                    │   jpeg: 低频系数 (B,H,n,K,D)                │
                    │   png:  (deltas (B,H,n,r-1,D), anchor)      │
                    └───────────────┬────────────────────────────┘
                                    │ read()：逐块解压 + raw 尾拼接
                                    ▼
                        重构 KV 视图 (B,H,S,D)
```

## 因果安全设计

只归档**完整块**；当前未满块永远留在 `raw_buffer`（原始精度）。增量解码时
任意时刻的未来扰动都不会改写已归档张量（测试逐位验证）。`mode="png",
keep_ratio=1.0` 的无损档位下，逐步解码注意力与全量前向注意力对齐 ≤1e-4。

## 安装

```bash
pip install -e .        # 依赖：torch>=2.0
```

## 快速上手

```python
import torch
from flashcodec import CodecKVCache, apply_flash_codec

# 1) 直接用缓存管理器（双接口）
cache = CodecKVCache(mode="jpeg", block_size=8, keep_freqs=2)   # 4× 压缩
k, v = cache.forward(k_seq, v_seq)          # 全量前向：分块平铺一次算完
k, v = cache.decode_step(new_k, new_v)      # 增量解码：跨块边界自动归档
print(cache.stats(0))                       # tokens / archived / storage_bytes

# 2) 一行热补丁注入 HuggingFace 因果 LM（走标准 Cache 接口）
apply_flash_codec(model, mode="png", block_size=8, keep_ratio=1.0)
out = model.generate(**inputs, use_cache=True)   # KV 自动走编解码缓存
```

## 基准（实测，B=2 H=8 S=2048 D=64，随机游走自相关信号，fp32）

### 重构质量（`benchmarks/bench_reconstruct.py`）

| 配置 | PSNR (dB) | cosine | 存储比 |
|---|---|---|---|
| jpeg r=8 k=1（8×） | 42.95 | 0.99943 | 12.5% |
| jpeg r=8 k=2（4×） | 47.22 | 0.99985 | 25.0% |
| jpeg r=8 k=4（2×） | 51.66 | 1.00000 | 50.0% |
| png r=8 keep=0.5 | 50.11 | 0.99997 | 100%（稀疏留熵编码）|
| png r=8 keep=1.0 | ~196（fp 舍入级）| 1.00000 | 100% |

### 缓存占用与耗时（`benchmarks/bench_vram.py`，CPU）

| seq | 模式 | storage | 原始比 | fill | read |
|---|---|---|---|---|---|
| 4096 | jpeg k=2 | 4.0MB | 0.25 | 8.3ms | 2.4ms |
| 16384 | jpeg k=2 | 16.0MB | 0.25 | 28.3ms | 13.0ms |
| 65536 | jpeg k=2 | 64.0MB | 0.25 | 132.6ms | 50.2ms |
| 65536 | png keep=1.0 | 256.0MB | 1.00 | 196.3ms | 82.9ms |

PNG 模式不省稠密存储（无损基线的宿命）——它的价值在 30× 方差压缩 + Top-M
近零稀疏残差，是熵编码器的理想输入（本库不内置熵编码）。

## 目录结构

```
flashcodec/
├── flashcodec/
│   ├── __init__.py        # apply_flash_codec / CodecKVCache / JPEGCodec / PNGCodec
│   ├── codecs/jpeg.py     # DCT-II 基底、投影/重构（纯 GEMM，无循环）
│   ├── codecs/png.py      # Anchor + 一阶差分、cumsum 还原、Top-M 阈值滤波
│   ├── cache.py           # CodecKVCache：raw 尾 + 分块归档，全量/增量双接口
│   └── patch.py           # 一行热补丁：Cache 层注入（HF Cache 鸭子类型）
├── tests/                 # jpeg / png / causal 三组测试（直接运行或 pytest）
├── benchmarks/            # 显存压测 + PSNR/余弦质量压测
├── pyproject.toml
└── README.md
```

## 测试

```bash
python tests/test_jpeg.py     # 正交归一、K=r 无损、k=2 余弦>0.99、K 单调
python tests/test_png.py      # 无损 maxdiff<1e-6、方差压缩比打印、可微
python tests/test_causal.py   # 未来扰动零泄露、增量 vs 全量 ≤1e-4
```

## 边界与说明（诚实清单）

- 研究原型：`read()` 物化完整重构视图参与注意力，显存收益在**缓存存储层**
  （如上表）；与 flash-attention 类不物化内核结合才能量化端到端收益。
- 全量 `forward(seq)` 预填会把完整块直接压缩，块内位置的注意力会看到块级
  重构（块内近似）；增量 `decode_step` 路径无此问题（当前块恒为 raw）。
- 压缩张量为稠密布局；PNG 的 Top-M 稀疏性尚未接熵编码器。
- 与 [Mini-Architecture](../README.md) 中 v4.2 原型（模型内 KV 复用层的
  DCT/差分读取）同源：那里是"每位置视图"语义，这里是标准 cache 存储语义。

## License

MIT
