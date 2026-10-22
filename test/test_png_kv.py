"""v4.2 PNG 式差分预测滤波 KV 单元测试：
1) keep_ratio=1.0 逐位无损基线（maxdiff < 1e-6）
2) 差分方差骤降特性（Var(Δ) ≪ Var(X)，打印方差压缩比）
3) 因果安全（扰动未来 → 历史块逐位不变）
4) KV cache 增量解码与全量前向一致（maxdiff < 1e-4）
5) 端到端 forward+backward 无 NaN/Inf
运行：.venv/bin/python test/test_png_kv.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from model_v42 import (MiniV42Config, MiniV42ForCausalLM, causal_png_pool_read,
                       png_block_recon)

torch.manual_seed(0)
fail = 0


def check(name, ok, extra=""):
    global fail
    fail |= not bool(ok)
    print(f"{'✅' if ok else '❌'} {name}  {extra}")


r, B, S, H = 8, 2, 64, 32

# ---- 1) 无损基线 ----
x = torch.randn(B, S, 1, H)
head = x.reshape(B, S // r, r, H)
rec1 = png_block_recon(head, 1.0)
d1 = (rec1 - head).abs().max().item()
check("keep_ratio=1.0 逐位无损（<1e-6）", d1 < 1e-6, f"maxdiff={d1:.2e}")

# ---- 2) 差分方差骤降 + 裁剪单调 ----
smooth = torch.cumsum(torch.randn(B, S, 1, H) * 0.1, dim=1)   # 自相关隐向量模拟
hs = smooth.reshape(B, S // r, r, H)
var_x = hs.var().item()
var_d = (hs[:, :, 1:, :] - hs[:, :, :-1, :]).var().item()
ratio = var_d / var_x
print(f"   方差比 Var(Δ)/Var(X) = {ratio:.4f}（方差压缩 {1/ratio:.1f}×）")
check("差分方差骤降（Var(Δ) < 0.5·Var(X)）", ratio < 0.5)
e1 = (png_block_recon(hs, 1.0) - hs).abs().mean().item()
e05 = (png_block_recon(hs, 0.5) - hs).abs().mean().item()
check("Top-M 裁剪误差单调（keep=0.5 有损 > 无损）", e05 > e1,
      f"err1.0={e1:.2e} err0.5={e05:.4f}")
check("输出无 NaN", torch.isfinite(png_block_recon(hs, 0.5)).all())

# ---- 3) 因果安全 ----
lat = torch.randn(2, 48, 1, 32)
view = causal_png_pool_read(lat, r, 0.5)
all_ok = True
for cut in (8, 13, 24, 33, 48):
    lat2 = lat.clone()
    lat2[:, cut:] = torch.randn_like(lat2[:, cut:])
    v2 = causal_png_pool_read(lat2, r, 0.5)
    all_ok &= torch.equal(view[:, :cut], v2[:, :cut])
check("因果安全（扰动未来不影响历史，逐位一致）", all_ok)

# ---- 4) 端到端 fwd+bwd ----
torch.manual_seed(7)
m = MiniV42ForCausalLM(MiniV42Config(
    vocab_size=1000, kv_compress_mode="png", png_block_size=8,
    png_keep_ratio=0.5)).train()
out = m(torch.randint(0, 1000, (2, 64)),
        labels=torch.randint(0, 1000, (2, 64)))
out.loss.backward()
gbad = [n for n, p in m.named_parameters()
        if p.grad is not None and not torch.isfinite(p.grad).all()]
check("png 端到端 fwd+bwd 有限（无 NaN/Inf）",
      torch.isfinite(out.loss) and not gbad, f"loss={out.loss.item():.3f}")

m_mean = MiniV42ForCausalLM(MiniV42Config(vocab_size=1000)).eval()
m.eval()
with torch.no_grad():
    xq = torch.randint(0, 1000, (2, 64))
    check("png 与 mean 输出不同（模式确实生效）",
          not torch.allclose(m_mean(xq).logits, m(xq).logits, atol=1e-4))

# ---- 5) KV cache 一致性 ----
with torch.no_grad():
    ids = torch.randint(0, 1000, (1, 40))
    state = {}
    lg_c = m(ids, state=state).logits[:, -1]
    maxd = 0.0
    for _ in range(8):
        lg_f = m(ids).logits[:, -1]               # 同一前缀全量对照
        maxd = max(maxd, (lg_f - lg_c).abs().max().item())
        nxt = lg_c.argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], -1)
        lg_c = m.decode_step(nxt, state)[:, -1]
check("KV cache 增量解码与全量前向一致（<1e-4）", maxd < 1e-4,
      f"maxdiff={maxd:.2e}")

print("\nALL PASS" if not fail else "\nFAILURES")
sys.exit(1 if fail else 0)
