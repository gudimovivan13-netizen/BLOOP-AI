"""Мини-GPT с нуля: символьный токенизатор + трансформер (архитектура GPT) на PyTorch.

Это учебная, но настоящая языковая модель: causal self-attention, многоголовое
внимание, MLP-блоки, residual connections и LayerNorm — всё то же, что и в
больших GPT-моделях, только размер поменьше.
"""
import json
import math
from dataclasses import dataclass, asdict
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharTokenizer:
    """Символьный токенизатор: каждый символ текста — один токен."""

    def __init__(self, chars):
        self.chars = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}
        self.unk = " "  # неизвестные символы заменяем пробелом

    @classmethod
    def from_text(cls, text, min_count=3, max_vocab=512):
        counts = Counter(text)
        pairs = [(c, n) for c, n in counts.items() if n >= min_count]
        pairs.sort(key=lambda cn: (-cn[1], cn[0]))
        chars = [c for c, _ in pairs[:max_vocab]]
        if " " not in chars:
            chars.append(" ")
        return cls(chars)

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text):
        return [self.stoi.get(c, self.stoi[self.unk]) for c in text]

    def decode(self, ids):
        return "".join(self.itos.get(int(i), "") for i in ids)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"chars": self.chars}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f)["chars"])


@dataclass
class GPTConfig:
    vocab_size: int = 512
    block_size: int = 192   # длина контекста (сколько символов модель видит)
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 192
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    """Многоголовое каузальное (само)внимание: токен смотрит только в прошлое."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, H, T, hs)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hs)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v                                   # (B, H, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    """Полносвязный блок: расширяем размер в 4 раза, GELU, сжимаем обратно."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Один блок трансформера: внимание + MLP с остаточными связями и LayerNorm."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """Сама модель: эмбеддинги + N блоков трансформера + линейный выход в словарь."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("proj.weight") or pn.endswith("attn.proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel() + self.pos_emb.weight.numel()
        return n

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.85, top_k=40):
        """Авторегрессионная генерация: по одному символу, с top-k сэмплированием."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None and 0 < top_k < logits.size(-1):
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx


def save_checkpoint(path, model: GPT, tok: CharTokenizer):
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(model.cfg),
            "chars": tok.chars,
        },
        path,
    )


def load_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=True)
    cfg = GPTConfig(**payload["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    tok = CharTokenizer(payload["chars"])
    return model, tok
