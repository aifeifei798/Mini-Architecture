"""MoE Triton grouped GEMM 路径 vs 纯 torch einsum 参考路径 等价测试：
前向输出 + 对 x / 全部专家参数的梯度逐一对比（fp32 与 bf16+autocast 两种）。
需要 CUDA；通过 monkeypatch model_v41.HAS_TRITON 切换两条路径。
运行：.venv/bin/python test/moe_eq_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import model_v41 as M

assert M.HAS_TRITON, "triton 不可用"
assert torch.cuda.is_available(), "需要 CUDA"
torch.manual_seed(0)

cfg = M.MiniV41Config()
moe = M.MiniMoE(cfg).cuda()
params = [p for p in moe.parameters() if p.requires_grad]
names = [n for n, p in moe.named_parameters() if p.requires_grad]


def run(path, x0, dtype):
    M.HAS_TRITON = (path == 'tri')
    for p in moe.parameters():
        p.grad = None
    xi = x0.detach().to(dtype).requires_grad_(True)
    with torch.autocast('cuda', dtype=torch.bfloat16,
                        enabled=(dtype == torch.bfloat16)):
        y = moe(xi)
    grads = torch.autograd.grad(
        y.float().sum(), [xi] + params, allow_unused=True)
    return y.detach().float(), grads


fail = 0
for dtype in (torch.float32, torch.bfloat16):
    x0 = torch.randn(20, 47, cfg.hidden_size, device='cuda')
    y_t, g_t = run('tri', x0, dtype)
    y_r, g_r = run('ref', x0, dtype)
    # fp32 双方都用 ieee 精度，应几乎逐位一致；bf16 下参考路径的 einsum
    # 会被 autocast 成 bf16，容差放宽但结构错误（错 gather）仍会暴露
    tol = 1e-5 if dtype == torch.float32 else 2e-2

    d = (y_t - y_r).abs().max().item()
    scale = y_r.abs().max().item()
    ok = d < tol * max(scale, 1.0)
    fail |= not ok
    print(f'[{dtype}] forward maxdiff={d:.3e}  (scale={scale:.3f})  '
          f'{"OK" if ok else "FAIL"}')

    for name, a, b in zip(['x'] + names, g_t, g_r):
        if a is None or b is None:
            print(f'  {name:28s} grad None? tri={a is None} ref={b is None}')
            fail |= (a is None) != (b is None)
            continue
        d = (a.float() - b.float()).abs().max().item()
        s = b.float().abs().max().item()
        ok = d < tol * max(s, 1.0) + 1e-7
        fail |= not ok
        if not ok or name.startswith('experts.0') or name == 'x':
            print(f'  {name:28s} maxdiff={d:.3e} (scale={s:.3f}) '
                  f'{"OK" if ok else "FAIL"}')

M.HAS_TRITON = True
print('ALL PASS' if not fail else 'FAILURES PRESENT')
sys.exit(1 if fail else 0)
