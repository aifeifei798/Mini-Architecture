"""v4.2 分块 DCT 频域压缩 KV 单元测试：
1) 基底正交归一 + keep=block 重构无损
2) 低频截断质量：平滑信号高余弦相似度，K 越大误差越小（单调）
3) 因果安全：扰动未来 latent 不影响历史读取（逐位一致）
4) mean 模式与 v41 逐位一致（回归）
5) 端到端：dct 模式 fwd+bwd 有限无 NaN；KV cache 增量解码与全量前向一致
运行：.venv/bin/python test/test_dct_kv.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

import model_v41
import model_v42
from model_v42 import (MiniV42Config, MiniV42ForCausalLM, causal_dct_pool_read,
                       causal_pool_read, dct_basis)

torch.manual_seed(0)
fail = 0


def check(name, ok, extra=""):
    global fail
    fail |= not bool(ok)
    print(f"{'✅' if ok else '❌'} {name}  {extra}")


# ---- 1) 正交性与无损重构 ----
r, B, S, H = 8, 2, 64, 32
enc = dct_basis(r, r)
check("基底行正交归一 (keep=r)",
      torch.allclose(enc @ enc.t(), torch.eye(r), atol=1e-6))
x = torch.randn(B, S, 1, H)
ref = x.reshape(B, S // r, r, H)                # (B, nF, r, H)
hp = ref.transpose(-2, -1)                      # (B, nF, H, r)
rec = torch.matmul(torch.matmul(hp, enc.t()), enc).transpose(-2, -1)
check("keep=r 分块重构无损", torch.allclose(rec, ref, atol=1e-5),
      f"maxdiff {(rec - ref).abs().max():.2e}")

# ---- 2) 低频截断质量（压缩器本身 + 相对无损基线的视图质量）----
smooth = torch.cumsum(torch.randn(B, S, 1, H) * 0.1, dim=1)
head_s = smooth.reshape(B, S // r, r, H)          # (B, nF, r, H)


def recon(k):
    e = dct_basis(r, k)
    return torch.matmul(torch.matmul(head_s.transpose(-2, -1).float(), e.t()),
                        e).transpose(-2, -1)


e2 = (recon(2) - head_s).abs().mean().item()
e4 = (recon(4) - head_s).abs().mean().item()
check("压缩重构误差随 K 单调下降 (K=4 < K=2)", e4 < e2,
      f"err2={e2:.4f} err4={e4:.4f}")
# 读取视图（含因果延迟语义）与无损块基线（keep=block）的相似度：
# 截断只应带来小扰动，K 越大越接近无损基线
v2 = causal_dct_pool_read(smooth, r, 2)
v4 = causal_dct_pool_read(smooth, r, 4)
v8 = causal_dct_pool_read(smooth, r, 8)           # 无损块 = 纯因果延迟基线
seg = slice(r, None)                              # 跳过 p<block 的 raw 段


def cos(a, b):
    return F.cosine_similarity(a[:, seg].reshape(1, -1),
                               b[:, seg].reshape(1, -1), dim=1).item()


cos2, cos4 = cos(v2, v8), cos(v4, v8)
check("K=2 视图接近无损块基线", cos2 > 0.9, f"cos={cos2:.4f}")
check("视图相似度随 K 单调上升", cos4 >= cos2, f"cos2={cos2:.4f} cos4={cos4:.4f}")
white = torch.randn(B, S, 1, H)
check("输出无 NaN", torch.isfinite(v2).all() and torch.isfinite(v4).all())

# ---- 3) 因果安全 ----
lat = torch.randn(2, 48, 1, 32)
view = causal_dct_pool_read(lat, r, 2)
all_ok = True
for cut in (8, 13, 24, 33, 48):
    lat2 = lat.clone()
    lat2[:, cut:] = torch.randn_like(lat2[:, cut:])
    v2p = causal_dct_pool_read(lat2, r, 2)
    all_ok &= torch.equal(view[:, :cut], v2p[:, :cut])
check("因果安全（扰动未来不影响历史，逐位一致）", all_ok)

# ---- 4) mean 模式回归 ----
lat = torch.randn(2, 40, 1, 16)
check("mean 模式与 v41 逐位一致",
      torch.equal(model_v41.causal_pool_read(lat, 2),
                  causal_pool_read(lat, 2)))

# ---- 5) 端到端 ----
torch.manual_seed(7)
m = MiniV42ForCausalLM(MiniV42Config(
    vocab_size=1000, kv_compress_mode="dct",
    dct_block_size=8, dct_keep_freqs=2)).train()
out = m(torch.randint(0, 1000, (2, 64)),
        labels=torch.randint(0, 1000, (2, 64)))
out.loss.backward()
gbad = [n for n, p in m.named_parameters()
        if p.grad is not None and not torch.isfinite(p.grad).all()]
check("dct 端到端 fwd+bwd 有限（无 NaN/Inf）",
      torch.isfinite(out.loss) and not gbad, f"loss={out.loss.item():.3f}")

m_mean = MiniV42ForCausalLM(MiniV42Config(vocab_size=1000)).eval()
m.eval()
xq = torch.randint(0, 1000, (2, 64))
with torch.no_grad():
    y_mean = m_mean(xq).logits
    y_dct = m(xq).logits
check("dct 与 mean 输出不同（模式确实生效）",
      not torch.allclose(y_mean, y_dct, atol=1e-4))

# KV cache 增量解码 vs 全量前向（同一前缀对照，喂相同 token）
with torch.no_grad():
    ids = torch.randint(0, 1000, (1, 40))
    state = {}
    lg_c = m(ids, state=state).logits[:, -1]     # 前缀 ids 下的预测
    maxd = 0.0
    for _ in range(8):
        lg_f = m(ids).logits[:, -1]              # 同一前缀全量对照
        maxd = max(maxd, (lg_f - lg_c).abs().max().item())
        nxt = lg_c.argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], -1)
        lg_c = m.decode_step(nxt, state)[:, -1]  # 前缀+1 下的预测
check("KV cache 增量解码与全量前向一致（dct）", maxd < 1e-3,
      f"maxdiff={maxd:.2e}")

print("\nALL PASS" if not fail else "\nFAILURES")
sys.exit(1 if fail else 0)
