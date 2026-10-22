"""V4.1 架构 ablation：S=512 真实语料，预训练 5000 步同预算，
各配置对比 best val loss（val 批次固定，跨配置可比）。

- base       默认配置（重跑以对齐 train_long 当前余弦调度）
- no_reuse   关跨层 KV 复用（每层自存 KV，顺带各层自持 indexer，参数 +~0.1M）
- idx4/32    稀疏度左/右移（默认 index_topk=8）
- dense      索引全选 ≈ 全注意力（稀疏注意力的上界参照）
- no_engram  关 engram 注入
- dct_r8_k2 / dct_r8_k4   v4.2 频域分块压缩 KV（model_v42.py 分支）：
  完整历史块投影到最低 2/4 个 DCT-II 低频系数再重构（几何压缩比 4×/2×），
  因果安全与均值池化同构

复用 train_long 的数据管道/训练循环。产出 ./ablate/<name>/ + ablate/results.json
（逐配置落盘，可删掉某项重跑续上）。

用法：.venv/bin/python ablate_long.py（6 配置 × ~19 分钟，GPU 串行）
"""

import json
from pathlib import Path

import numpy as np
import torch

import train_long as T
from model_v41 import MiniV41Config, MiniV41ForCausalLM
from model_v42 import MiniV42Config, MiniV42ForCausalLM

OUT = Path("./ablate")
OUT.mkdir(exist_ok=True)

CONFIGS = {
    "base":      dict(),
    "no_reuse":  dict(compress_ratios=(0,) * 8, kv_source_layer_ids=()),
    "idx4":      dict(index_topk=4),
    "idx32":     dict(index_topk=32),
    "dense":     dict(index_topk=4096, candidate_topk_blocks=4096),
    "no_engram": dict(engram_layer_ids=()),
}

# v4.2 频域分块压缩 KV（model_v42.py 原型分支）
V42_CONFIGS = {
    "dct_r8_k2": dict(kv_compress_mode="dct", dct_block_size=8,
                      dct_keep_freqs=2),
    "dct_r8_k4": dict(kv_compress_mode="dct", dct_block_size=8,
                      dct_keep_freqs=4),
}

if __name__ == "__main__":
    train_flat = T.load_bin("train.bin")
    val_flat = T.load_bin("val.bin")
    amp_dtype = (torch.bfloat16
                 if T.device == "cuda" and torch.cuda.is_bf16_supported()
                 else torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    res_path = OUT / "results.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else {}

    all_cfgs = ([(n, o, MiniV41Config, MiniV41ForCausalLM)
                 for n, o in CONFIGS.items()]
                + [(n, o, MiniV42Config, MiniV42ForCausalLM)
                   for n, o in V42_CONFIGS.items()])
    for name, over, cfg_cls, model_cls in all_cfgs:
        if name in results:
            print(f"⏭️ 跳过 {name}（已有结果 {results[name]['best_val']:.4f}）")
            continue
        print(f"\n{'='*60}\n🔬 配置 {name}: {over or '（默认）'}\n{'='*60}")
        cfg = cfg_cls(vocab_size=T.meta["vocab_size"], **over)
        model = model_cls(cfg).to(T.device)
        n_all = sum(p.numel() for p in model.parameters())
        best_val = T.run_stage(model, train_flat, val_flat, T.PRETRAIN_STEPS,
                               f"abl-{name}", T.PRETRAIN_LR, scaler, amp_dtype,
                               save_dir=str(OUT / name))

        # 收尾：验证批 loss + indexer 密度
        model.eval()
        model.reset_stats()
        x, y = T.get_batch(val_flat, torch.arange(T.BATCH))
        with torch.no_grad():
            with torch.amp.autocast(device_type=T.device, dtype=amp_dtype,
                                    enabled=(T.device == "cuda")):
                vloss = model(x, labels=y).loss.item()
        dens = [round(d, 4) for d in model.index_density()]
        results[name] = {"best_val": round(best_val, 4),
                         "params_m": round(n_all / 1e6, 2),
                         "config": over, "probe_loss": round(vloss, 4),
                         "index_density": dens}
        res_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(f"✅ {name}: best_val {best_val:.4f} params {n_all/1e6:.2f}M "
              f"probe_loss {vloss:.4f} dens {dens}")

    print("\n📊 汇总（best val loss 越低越好）:")
    for k, v in sorted(results.items(), key=lambda kv: kv[1]["best_val"]):
        print(f"  {k:10s} best_val {v['best_val']:.4f}  "
              f"params {v['params_m']:>6}M  密度均值 "
              f"{sum(v['index_density'])/len(v['index_density']):.3f}")
