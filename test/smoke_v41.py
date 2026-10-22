"""V4.1 冒烟测试：causality 断言 + S 扫描前向/反向有限 + 诊断 API。
从仓库任意位置运行：.venv/bin/python test/smoke_v41.py
CPU 即可（MoE 走 einsum 参考路径）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from model_v41 import MiniV41Config, MiniV41ForCausalLM

torch.manual_seed(0)
cfg = MiniV41Config(vocab_size=1000)
m = MiniV41ForCausalLM(cfg).eval()
n_all = sum(p.numel() for p in m.parameters())
n_body = n_all - m.embed.weight.numel() - m.lm_head.weight.numel()
print(f"params: total {n_all/1e6:.2f}M, body {n_body/1e6:.2f}M")

# 1) causality: S=32, perturb last token, prefix logits must match
with torch.no_grad():
    x = torch.randint(0, 1000, (2, 32))
    logits_a = m(x).logits
    x2 = x.clone(); x2[:, -1] = (x2[:, -1] + 1) % 1000
    logits_b = m(x2).logits
    d = (logits_a[:, :-1] - logits_b[:, :-1]).abs().max().item()
    print(f"causality prefix diff: {d:.2e}", "OK" if d < 1e-4 else "FAIL")
    assert d < 1e-4, "causality violated"

# 2) forward+backward finite at various S (incl. MTP loss path)
for S in (1, 3, 7, 48, 49):
    x = torch.randint(0, 1000, (2, S))
    y = torch.randint(0, 1000, (2, S))
    m.train()
    out = m(x, labels=y)
    loss = out.loss
    loss.backward()
    gbad = [n for n, p in m.named_parameters()
            if p.grad is not None and not torch.isfinite(p.grad).all()]
    for p in m.parameters():
        p.grad = None
    m.eval()
    print(f"S={S:2d} loss={loss.item():.4f} grad_finite={'OK' if not gbad else 'BAD '+str(gbad[:3])}")
    assert torch.isfinite(loss) and not gbad

# 3) diagnostics API intact
m.reset_stats()
with torch.no_grad():
    m(torch.randint(0, 1000, (2, 48)), labels=torch.randint(0, 1000, (2, 48)))
u = m.expert_usage(); d = m.index_density()
print("usage shape:", tuple(u.shape), "density:", [round(x, 2) for x in d])
assert u.shape == (8, 8) and len(d) == 8 and all(0 <= x <= 1 for x in d)
print("ALL SMOKE OK")
