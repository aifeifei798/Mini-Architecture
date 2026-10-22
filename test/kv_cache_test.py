"""KV 缓存增量解码 vs 全量前向：贪心生成一致性 + 计时。
判据：token 全程一致 → PASS；若某步不同，只有当该步 full 路径 top1-top2
差距 > 1e-4（即模型本身有明确偏好，非 fp 噪声翻转）才算 FAIL。
1) CPU：./my_deepseek_v41_mini（若存在）或随机小模型，走 einsum 回退路径
2) CUDA：./my_deepseek_v41_long（若存在），真实长上下文 checkpoint
运行：.venv/bin/python test/kv_cache_test.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from model_v41 import MiniV41Config, MiniV41ForCausalLM

TIE_TOL = 1e-4


def compare(model, prompt, n, tag):
    ids_f = prompt                      # 全量前向路径
    ids_c = prompt                      # 缓存路径
    state = {}
    t0 = time.time()
    logits_c = model(ids_c, state=state).logits[:, -1, :]
    t_cache = time.time() - t0
    t0 = time.time()
    logits_f = model(ids_f).logits[:, -1, :]
    t_full = time.time() - t0

    fail = warn = 0
    maxdiff = 0.0
    for step in range(n):
        maxdiff = max(maxdiff, (logits_f - logits_c).abs().max().item())
        s = logits_f.topk(2, dim=-1).values
        gap = (s[..., 0] - s[..., 1]).max().item()
        nf = logits_f.argmax(dim=-1, keepdim=True)
        nc = logits_c.argmax(dim=-1, keepdim=True)
        if not torch.equal(nf, nc):
            if gap > TIE_TOL:
                fail += 1
                print(f"  ❌ step {step}: token 分歧且 top1 差距 {gap:.2e}"
                      f"（full={nf.item()} cached={nc.item()}）")
            else:
                warn += 1
            # 分歧后两路各自继续（缓存路径以自己的 token 往下）
        ids_f = torch.cat([ids_f, nf], dim=-1)
        ids_c = torch.cat([ids_c, nc], dim=-1)
        t0 = time.time()
        logits_f = model(ids_f).logits[:, -1, :]
        t_full += time.time() - t0
        t0 = time.time()
        logits_c = model.decode_step(nc, state)[:, -1, :]
        t_cache += time.time() - t0

    ok = fail == 0
    note = "" if warn == 0 else f"（{warn} 步平局翻转，gap<{TIE_TOL}，容忍）"
    print(f"[{tag}] {'PASS' if ok else 'FAIL'} | logits maxdiff {maxdiff:.2e} "
          f"| full {t_full:.2f}s vs cached {t_cache:.2f}s "
          f"（{t_full/max(t_cache,1e-9):.1f}×）| gen {n} tok{note}")
    return ok


torch.manual_seed(0)
ok = True

if Path("./my_deepseek_v41_mini/config.json").exists():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("./my_deepseek_v41_mini")
    m = MiniV41ForCausalLM.from_pretrained("./my_deepseek_v41_mini").eval().float()
    p = tok("问：中国的首都是哪里？ 答：", return_tensors="pt").input_ids
else:
    m = MiniV41ForCausalLM(MiniV41Config(vocab_size=1000)).eval().float()
    p = torch.randint(0, 1000, (1, 40))
ok &= compare(m, p, 40, "cpu")

if torch.cuda.is_available() and Path("./my_deepseek_v41_long/config.json").exists():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("./my_deepseek_v41_long")
    m2 = MiniV41ForCausalLM.from_pretrained("./my_deepseek_v41_long").cuda().eval()
    for text in ("人工智能是", "问：中国的首都是哪里？ 答："):
        p = tok(text, return_tensors="pt").input_ids.cuda()
        ok &= compare(m2, p, 60, f"cuda-long {text[:6]}")
    # 长前缀 + 长生成：全量重算是 O(n·S²)，缓存是 O(n·S)，此处应见真加速
    import numpy as np
    val = np.memmap("./data_long/val.bin", dtype=np.uint32, mode="r")
    p = torch.from_numpy(np.asarray(val[:300], dtype=np.int64)).unsqueeze(0).cuda()
    ok &= compare(m2, p, 120, "cuda-long 300tok-prefix")

print("ALL PASS" if ok else "FAILURES")
sys.exit(0 if ok else 1)
