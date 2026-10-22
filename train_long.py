"""V4.1 长上下文训练：真实语料（data_long.py 产出的打包 token 流），
随机起点切窗 → 全标签 LM。两阶段：预训练（pretrain val 门控 best 保存）→
SFT（sft val 门控，独立低 LR）。SFT_ONLY=True 时从 SAVE_DIR 加载只跑 SFT。

用法：.venv/bin/python train_long.py
产出：./my_deepseek_v41_long（HF 格式，含 tokenizer）
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from model_v41 import MiniV41Config, MiniV41ForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 设备: {device}")

DATA = Path("./data_long")
meta = json.loads((DATA / "meta.json").read_text())
SEQ_LEN = meta["seq_len"]
SAVE_DIR = "./my_deepseek_v41_long"

BATCH = 16             # 16 × 512 = 8192 token/步
PRETRAIN_STEPS = 5000  # ~2.2 epoch（35983 窗口/epoch）
SFT_STEPS = 3000       # 0 关闭 SFT 阶段
PRETRAIN_LR = 1.5e-3
SFT_LR = 1.5e-4        # SFT 用低 LR，避免冲掉预训练分布
GRAD_CLIP = 1.0
VAL_EVERY = 500
VAL_BATCHES = 16
SFT_ONLY = False       # True: 跳过预训练，从 SAVE_DIR 加载只跑 SFT


def load_bin(name):
    return np.memmap(DATA / name, dtype=np.uint32, mode="r")


def get_batch(flat, starts):
    x = torch.stack([torch.from_numpy(np.asarray(flat[s:s + SEQ_LEN],
                                                 dtype=np.int64))
                     for s in starts])
    y = torch.stack([torch.from_numpy(np.asarray(flat[s + 1:s + SEQ_LEN + 1],
                                                 dtype=np.int64))
                     for s in starts])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def evaluate(model, flat, tag="val"):
    model.eval()
    g = torch.Generator().manual_seed(1234)   # 固定批次，跨阶段可比
    n_tok = len(flat) - SEQ_LEN - 1
    starts = torch.randint(0, n_tok, (VAL_BATCHES, BATCH), generator=g)
    tot, n = 0.0, 0
    for bs in starts:
        x, y = get_batch(flat, bs)
        with torch.amp.autocast(device_type=device, dtype=torch.bfloat16,
                                enabled=(device == "cuda")):
            loss = model(x, labels=y).loss
        tot += loss.item() * BATCH
        n += BATCH
    model.train()
    val = tot / n
    print(f"  [{tag}] val loss {val:.4f} | ppl {math.exp(val):.2f}")
    return val


def run_stage(model, train_flat, val_flat, steps, tag, lr0, scaler, amp_dtype,
              save_dir=SAVE_DIR):
    opt = torch.optim.AdamW(model.parameters(), lr=lr0, weight_decay=1e-2,
                            fused=(device == "cuda"))
    warmup = 50 if tag == "sft" else 200

    def mul(step):                      # warmup 线性 + 余弦衰到 0.1 倍
        if step < warmup:
            return (step + 1) / warmup
        t = (step - warmup) / max(1, steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, mul)
    model.train()
    n_tok = len(train_flat) - SEQ_LEN - 1
    g = torch.Generator()
    best = float("inf")
    t0 = time.time()
    for step in range(steps):
        starts = torch.randint(0, n_tok, (BATCH,), generator=g)
        x, y = get_batch(train_flat, starts)
        with torch.amp.autocast(device_type=device, dtype=amp_dtype,
                                enabled=(device == "cuda")):
            loss = model(x, labels=y).loss
        if amp_dtype == torch.float16:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
        model.update_expert_balance(lr=0.01)
        if step % 100 == 0 or step == steps - 1:
            print(f"[{tag}] step {step:4d}/{steps} | loss {loss.item():.4f} "
                  f"| {BATCH*SEQ_LEN*(step+1)/(time.time()-t0):,.0f} tok/s "
                  f"| lr {sched.get_last_lr()[0]:.2e}")
        if (step + 1) % VAL_EVERY == 0 or step == steps - 1:
            v = evaluate(model, val_flat, tag)
            if v < best:
                best = v
                model.save_pretrained(save_dir)
                print(f"  💾 新 best {tag} val {v:.4f}，已存 {save_dir}")
    print(f"⚡ [{tag}] 完成 {time.time()-t0:.0f}s，best val {best:.4f}")
    return best


if __name__ == "__main__":
    train_flat = load_bin("train.bin")
    val_flat = load_bin("val.bin")
    sft_train = load_bin("sft_train.bin")
    sft_val = load_bin("sft_val.bin")
    print(f"📦 pretrain {len(train_flat)/1e6:.1f}M tok | sft "
          f"{len(sft_train)/1e6:.1f}M tok | S={SEQ_LEN}")

    tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
    amp_dtype = (torch.bfloat16
                 if device == "cuda" and torch.cuda.is_bf16_supported()
                 else torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    if SFT_ONLY:
        model = MiniV41ForCausalLM.from_pretrained(SAVE_DIR).to(device)
        print(f"🔁 SFT_ONLY：从 {SAVE_DIR} 加载")
    else:
        config = MiniV41Config(vocab_size=meta["vocab_size"])
        model = MiniV41ForCausalLM(config).to(device)
        n_all = sum(p.numel() for p in model.parameters())
        n_body = n_all - model.embed.weight.numel() - model.lm_head.weight.numel()
        print(f"🧱 参数: 总 {n_all/1e6:.1f}M（embed+head "
              f"{2*model.embed.weight.numel()/1e6:.1f}M，本体 {n_body/1e6:.2f}M）")
        run_stage(model, train_flat, val_flat, PRETRAIN_STEPS, "pretrain",
                  PRETRAIN_LR, scaler, amp_dtype)

    if SFT_STEPS > 0:
        run_stage(model, sft_train, sft_val, SFT_STEPS, "sft",
                  SFT_LR, scaler, amp_dtype)

    tokenizer.save_pretrained(SAVE_DIR)

    # 诊断（验证流上跑一批，看 S=512 下的路由/稀疏行为）
    model.eval()
    model.reset_stats()
    x, y = get_batch(val_flat, torch.arange(BATCH))
    with torch.no_grad():
        model(x, labels=y)
    usage = model.expert_usage().cpu()
    print("\n📊 专家使用（8 层 × 8 专家）:")
    for i, row in enumerate(usage):
        print(f"  L{i}: " + " ".join(f"{v:.2f}" for v in row))
    print("📊 indexer 密度 @S=512:",
          " ".join(f"{d:.3f}" for d in model.index_density()))

    # 生成抽样：预训练式续写 + SFT 式对话
    loaded_tok = AutoTokenizer.from_pretrained(SAVE_DIR)
    loaded = MiniV41ForCausalLM.from_pretrained(SAVE_DIR).to(device)
    loaded.eval()
    print("\n--- 生成抽样 ---")
    for prompt in ("人工智能是", "中国的首都是", "问：中国的首都是哪里？ 答："):
        ids = loaded_tok(prompt, return_tensors="pt").input_ids.to(device)
        out = loaded.generate(ids, max_new_tokens=40, do_sample=False,
                              pad_token_id=loaded_tok.eos_token_id)
        print(f"  {prompt!r} → "
              f"{loaded_tok.decode(out[0], skip_special_tokens=True)!r}")
