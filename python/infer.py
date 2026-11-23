"""Генерация текста обученной мини-моделью.

Запуск:  python3 python/infer.py --prompt "Квантовая механика —" --max-new 300
Вывод: JSON {"ok": true, "text": "..."} в stdout.
"""
import argparse
import json
import os
import sys

import torch

from minigpt import load_checkpoint

DEFAULT_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "mini_gpt.pt"
)


def trim_text(text: str, min_len: int = 80) -> str:
    """Обрезаем продолжение на границе абзаца, чтобы не лезть в «другую статью»."""
    cut = text.find("\n\n")
    if cut != -1 and cut >= min_len:
        return text[:cut].strip()
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--top-k", type=int, default=40)
    args = ap.parse_args()

    if not os.path.exists(args.checkpoint):
        print(json.dumps({"ok": False, "error": "чекпойнт не найден: " + args.checkpoint}, ensure_ascii=False))
        sys.exit(1)

    device = "cpu"
    model, tok = load_checkpoint(args.checkpoint, device)
    prompt = args.prompt if args.prompt.strip() else "Наука"
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    if ids.shape[1] == 0:
        ids = torch.tensor([[tok.stoi[" "]]], dtype=torch.long)
    out = model.generate(ids, args.max_new, temperature=args.temperature, top_k=args.top_k)
    new_ids = out[0, ids.shape[1]:].tolist()
    text = trim_text(tok.decode(new_ids))
    print(json.dumps({"ok": True, "text": text}, ensure_ascii=False))


if __name__ == "__main__":
    main()
