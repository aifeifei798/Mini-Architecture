"""V4.1 最小架构训练：20 条数据过拟合，验证架构可训练性。

用法：.venv/bin/python train_v41.py
产出：./my_deepseek_v41_mini（HF 格式，含 tokenizer）
"""

import time

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_v41 import qa_pairs
from model_v41 import MiniV41Config, MiniV41ForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 设备: {device}")

tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
VOCAB_SIZE = max(len(tokenizer), getattr(tokenizer, "vocab_size", 0))

SAVE_DIR = "./my_deepseek_v41_mini"
MAX_LEN = 48
BATCH = 20          # 20 条全量一步
EPOCHS = 300
LR = 3e-3


def build_dataset(texts, tokenizer, max_length=MAX_LEN):
    """SFT prompt masking：问题段 label=-100，只学答案。"""
    all_inputs, all_labels = [], []
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or 0

    for text in texts:
        q_text, _a = text.split(" 答：")
        prompt_ids = tokenizer.encode(q_text + " 答：")
        full_ids = tokenizer.encode(text) + [eos_id]
        if len(full_ids) > max_length:
            full_ids = full_ids[:max_length]

        pad_len = max_length - len(full_ids)
        input_ids = full_ids + [pad_id] * pad_len

        prompt_len = min(len(prompt_ids), max_length)
        target_ids = full_ids[1:] + [pad_id] + [-100] * pad_len
        target_ids = target_ids[:max_length]
        for i in range(prompt_len - 1):
            target_ids[i] = -100

        all_inputs.append(input_ids[:-1])
        all_labels.append(target_ids[:-1])

    x = torch.tensor(all_inputs, dtype=torch.long, device=device)
    y = torch.tensor(all_labels, dtype=torch.long, device=device)
    return x, y


if __name__ == "__main__":
    x, y = build_dataset(qa_pairs, tokenizer)
    print(f"📦 样本 {x.shape[0]} × 长度 {x.shape[1]}")

    config = MiniV41Config(vocab_size=VOCAB_SIZE)
    model = MiniV41ForCausalLM(config).to(device)
    n_all = sum(p.numel() for p in model.parameters())
    n_body = n_all - model.embed.weight.numel() - model.lm_head.weight.numel()
    print(f"🧱 参数: 总 {n_all/1e6:.2f}M（embed+head {2*model.embed.weight.numel()/1e6:.2f}M，"
          f"本体 {n_body/1e6:.2f}M）")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2,
                                  fused=(device == "cuda"))
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-4)
    amp_dtype = (torch.bfloat16
                 if device == "cuda" and torch.cuda.is_bf16_supported()
                 else torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    print(f"--- 1. 训练 {EPOCHS} epochs，batch={BATCH} ---")
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device, dtype=amp_dtype,
                                enabled=(device == "cuda")):
            loss = model(x, labels=y).loss
        if amp_dtype == torch.float16:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        scheduler.step()
        # noaux_tc 均衡：每步按被选频次修正偏置
        model.update_expert_balance(lr=0.01)

        if epoch % 50 == 0 or epoch == 1:
            print(f"epoch {epoch:03d} | loss {loss.item():.4f} | "
                  f"lr {scheduler.get_last_lr()[0]:.5f}")
    print(f"⚡ 训练完毕 {time.time()-t0:.1f}s，final loss {loss.item():.4f}")

    # 诊断：专家使用 / indexer 稀疏度
    model.eval()
    model.reset_stats()
    with torch.no_grad():
        _ = model(x, labels=y)
    usage = model.expert_usage().cpu()
    print("\n📊 各层专家使用频率 (行=层, 列=expert):")
    for i, row in enumerate(usage):
        print(f"  L{i}: " + " ".join(f"{v:.2f}" for v in row))
    print("📊 indexer 可见密度 (层):",
          " ".join(f"{d:.2f}" for d in model.index_density()))

    print("\n--- 2. 保存 ---")
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"✅ 已保存: {SAVE_DIR}")

    print("\n--- 3. 验证（重新加载 + greedy 生成）---")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    loaded_tok = AutoTokenizer.from_pretrained(SAVE_DIR)
    loaded = AutoModelForCausalLM.from_pretrained(SAVE_DIR).to(device)
    loaded.eval()

    test_cases = [qa_pairs[i].split(" 答：")[0] + " 答：" for i in (0, 5, 10, 15)]
    all_ok = True
    for prompt in test_cases:
        ids = loaded_tok(prompt, return_tensors="pt").input_ids.to(device)
        gen = []
        with torch.no_grad():
            for _ in range(30):
                out = loaded(ids)
                nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                if nxt.item() == loaded_tok.eos_token_id:
                    break
                gen.append(nxt.item())
                ids = torch.cat([ids, nxt], dim=-1)
        text = loaded_tok.decode(gen, skip_special_tokens=True)
        ref = [s for s in qa_pairs if s.startswith(prompt)]
        want = ref[0].split(" 答：")[1] if ref else ""
        ok = text.strip() == want.strip()
        all_ok &= ok
        print(f"{'✅' if ok else '❌'} 问 {prompt!r}")
        print(f"     得 {text.strip()!r}")
        if not ok:
            print(f"     期 {want.strip()!r}")
    print("\n验收:", "PASS — 全部命中" if all_ok else "部分未命中，可加 epochs 重训")
