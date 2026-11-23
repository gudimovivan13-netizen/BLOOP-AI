"""Извлечение русских научно-энциклопедических статей из parquet-шарда Wikipedia.

Файл: https://huggingface.co/datasets/wikimedia/wikipedia (20231101.ru)
Берём статьи, чьи заголовки содержат научные ключевые слова, с фильтром длины,
и складываем в data/corpus.txt до целевого объёма.
"""
import json
import os
import re
import sys
import urllib.request

import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
SHARD = os.path.join(HERE, "..", "data", "shard.parquet")
OUT = os.path.join(HERE, "..", "data", "corpus.txt")
TARGET_CHARS = 4_500_000
MIN_ART, MAX_ART = 600, 12000

KEYWORDS = [
    "физик", "хим", "биолог", "астро", "математ", "геолог", "географ", "географ",
    "информ", "компьют", "программ", "медицин", "анатом", "физиолог", "генет",
    "эколог", "вирус", "бактер", "атом", "молекул", "энерг", "теория", "алгебр",
    "геометр", "статист", "вероятн", "логик", "киберн", "робот", "нано",
    "квант", "нейронн", "интеллект", "космос", "космонавт", "планет", "звезд",
    "эволюц", "клетк", "гормон", "мозг", "синтез", "реакц", "минерал",
    "климат", "океан", "вулкан", "тектон", "метеор", "спутник", "галакт",
    "вселенн", "частиц", "изотоп", "радиац", "рентген", "лазер", "оптик",
    "электр", "магнит", "термодин", "гидрод", "аэродин", "механик", "динамик",
    "число", "уравнен", "функция", "дифференц", "интеграл", "алгоритм",
    "структур данных", "база данных", "шифр", "криптогр", "блокчейн",
    "дрон", "двигател", "турбин", "насос", "материаловед", "сплав",
    "полимер", "кристалл", "металл", "керосин", "нефт", "газ", "уголь",
    "ветеран", "заболев", "иммун", "вакцин", "антибиот", "онколог", "кардио",
    "невролог", "психи", "фармац", "токсикол", "гигиен", "санитар",
    "агроном", "почвовед", "гидролог", "сейсмолог", "метеоролог",
    "энциклопедия", "наука", "учён", "учен", "нулев", "нобелевск",
    "эйнштейн", "ньютон", "менделеев", "ломоносов", "кури", "дарвин",
    "бор", "планк", "фейнман", "хокинг", "гагарин", "королёв", "целли",
    "технолог", "инженер", "станок", "производств", "индустри",
]


def clean(t: str) -> str:
    t = re.sub(r"\[\d+\]", "", t)          # сноски вида [1]
    t = re.sub(r"\[+[a-zа-я ]+\]", "", t)  # [en], [прояснить] и т.п.
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def main() -> None:
    table = pq.read_table(SHARD, columns=["title", "text"])
    titles = table.column("title").to_pylist()
    texts = table.column("text").to_pylist()
    print(f"статей в шарде: {len(titles)}")

    kw = [k for k in KEYWORDS if k]
    picked = []
    seen_titles = set()
    for title, text in zip(titles, texts):
        t = title.lower()
        if not any(k in t for k in kw):
            continue
        if title in seen_titles:
            continue
        n = len(text)
        if n < MIN_ART or n > MAX_ART:
            continue
        seen_titles.add(title)
        picked.append((title, clean(text)))
    print(f"подходящих статей: {len(picked)}")

    total = 0
    chunks = []
    for title, text in picked:
        art = f"{title}\n{text}\n\n"
        chunks.append(art)
        total += len(art)
        if total >= TARGET_CHARS:
            break

    corpus = "".join(chunks)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(corpus)
    print(f"ГОТОВО: {len(chunks)} статей, {len(corpus)} символов ({total/1e6:.2f} MB) -> {OUT}")


if __name__ == "__main__":
    main()
