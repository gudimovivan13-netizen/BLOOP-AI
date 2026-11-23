"""Обучение мини-GPT на текстовом корпусе.

Запуск:  python3 python/train.py --corpus data/corpus.txt --steps 3000
"""
import argparse
import math
import os
import random
import time

import torch

from minigpt import CharTokenizer, GPT, GPTConfig, save_checkpoint


def get_batch(data, block_size, batch_size, device):
    """Случайные куски текста длиной block_size: вход x и цель y (сдвиг на 1)."""
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def eval_loss(model, data, block_size, batch_size, device, iters=12):
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch(data, block_size, batch_size, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus.txt")
    ap.add_argument("--out", default="checkpoints/mini_gpt.pt")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--block-size", type=int, default=192)
    ap.add_argument("--n-embd", type=int, default=192)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(max(1, (os.cpu_count() or 2)))
    device = "cpu"

    with open(args.corpus, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"корпус: {len(text)} символов из {args.corpus}")

    tok = CharTokenizer.from_text(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    n_val = int(len(ids) * args.val_frac)
    val_data = ids[:n_val]
    train_data = ids[n_val:]
    print(f"словарь: {tok.vocab_size} символов, train {len(train_data)}, val {len(val_data)}")

    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(cfg).to(device)
    print(f"параметров (без эмбеддингов): {model.num_params():,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        p = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.1 + 0.45 * (1 + math.cos(math.pi * p)))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    model.train()
    t0 = time.time()
    best_val = float("inf")
    for step in range(1, args.steps + 1):
        lr = lr_at(step - 1)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = get_batch(train_data, cfg.block_size, args.batch_size, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 50 == 0 or step == 1:
            speed = step / (time.time() - t0)
            print(f"шаг {step}/{args.steps}  loss {loss.item():.4f}  lr {lr:.5f}  {speed:.2f} шаг/с", flush=True)
        if step % 250 == 0 or step == args.steps:
            vloss = eval_loss(model, val_data, cfg.block_size, args.batch_size, device)
            print(f"шаг {step}: val loss {vloss:.4f}", flush=True)
            if vloss < best_val:
                best_val = vloss
                save_checkpoint(args.out, model, tok)
                print(f"  сохранён чекпойнт: {args.out}", flush=True)

    save_checkpoint(args.out, model, tok)
    print(f"ГОТОВО. финальный val loss {best_val:.4f}, модель: {args.out}")


if __name__ == "__main__":
    main()
