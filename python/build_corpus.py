"""Build a Russian encyclopedic corpus from Wikipedia for training the mini-GPT.

Fetches plaintext extracts of articles across many sciences via the
MediaWiki API, cleans them, and writes data/corpus.txt.
"""
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

API = "https://ru.wikipedia.org/w/api.php"
HEADERS = {
    "User-Agent": "mini-gpt-corpus-builder/1.0 (educational project; contact: local)",
}
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUT_FILE = os.path.join(OUT_DIR, "corpus.txt")
MAX_CHARS_PER_ARTICLE = 18000
TARGET_BYTES = 4_000_000

TITLES = [
    # Физика
    "Физика", "Квантовая механика", "Теория относительности", "Электромагнетизм",
    "Термодинамика", "Оптика", "Акустика", "Ядерная физика", "Чёрная дыра",
    "Гравитация", "Атом", "Электрон", "Фотон", "Энергия", "Свет", "Магнетизм",
    "Суперпозиция", "Квантовая запутанность", "Теория струн", "Физика элементарных частиц",
    "Нейтрон", "Протон", "Волна", "Скорость света", "Законы Ньютона", "Дифракция",
    "Интерференция", "Плазма", "Сверхпроводимость", "Энтропия",
    # Астрономия
    "Астрономия", "Вселенная", "Солнечная система", "Солнце", "Земля", "Луна",
    "Марс", "Юпитер", "Сатурн", "Венера", "Меркурий", "Нептун", "Уран",
    "Звезда", "Галактика", "Млечный Путь", "Туманность", "Комета", "Астероид",
    "Большой взрыв", "Тёмная материя", "Тёмная энергия", "Экзопланета", "Сверхновая звезда",
    "Планетарная туманность", "Квазар", "Пульсар", "Метеорит", "Затмение", "Орбита",
    # Химия
    "Химия", "Периодическая таблица", "Химический элемент", "Химическая связь",
    "Молекула", "Кислород", "Водород", "Углерод", "Азот", "Железо", "Золото",
    "Органическая химия", "Неорганическая химия", "Катализ", "Кислота", "Основание (химия)",
    "Соль (химия)", "Полимер", "Бензол", "Спирты", "Реакция горения", "Электролиз",
    "Изотоп", "Радиоактивность", "Кристалл", "Алмаз", "Графит", "Гелий", "Литий", "Натрий",
    # Биология
    "Биология", "Клетка", "ДНК", "РНК", "Ген", "Генетика", "Эволюция",
    "Естественный отбор", "Фотосинтез", "Митоз", "Мейоз", "Белки", "Фермент",
    "Мембрана", "Митохондрия", "Рибосома", "Бактерии", "Вирус", "Гриб",
    "Растения", "Животные", "Млекопитающие", "Птицы", "Рыбы", "Насекомые",
    "Экосистема", "Экология", "Пищевая цепь", "Иммунная система", "Нервная система",
    "Кровеносная система", "Гормоны", "Мозг", "Сердце", "Печень", "Лёгкие",
    # Математика
    "Математика", "Алгебра", "Геометрия", "Математический анализ", "Производная",
    "Интеграл", "Теория вероятностей", "Статистика", "Теория чисел", "Простое число",
    "Логарифм", "Тригонометрия", "Матрица (математика)", "Вектор", "Комплексное число",
    "Дифференциальное уравнение", "Топология", "Теория множеств", "Пи (число)",
    "Число e", "Золотое сечение", "Фрактал", "Теорема Пифагора", "Логика", "Алгоритм",
    # Информатика
    "Информатика", "Компьютер", "Программирование", "Язык программирования",
    "Python", "JavaScript", "Алгоритм сортировки", "Структура данных",
    "База данных", "Интернет", "Всемирная паутина", "Операционная система",
    "Linux", "Процессор", "Оперативная память", "Жёсткий диск", "Нейронная сеть",
    "Машинное обучение", "Искусственный интеллект", "Глубокое обучение",
    "Компьютерное зрение", "Обработка естественного языка", "Криптография",
    "Шифрование", "Блокчейн", "Квантовый компьютер", "Трансформер (модель машинного обучения)",
    # Науки о Земле
    "Геология", "География", "Климат", "Погода", "Атмосфера Земли", "Океан",
    "Вулкан", "Землетрясение", "Тектоника плит", "Магнитное поле Земли",
    "Гидросфера", "Литосфера", "Биосфера", "Ледник", "Пустыня", "Лес",
    "Гора", "Река", "Озеро", "Минерал", "Горная порода", "Ископаемые",
    # Медицина
    "Медицина", "Анатомия человека", "Физиология", "Вакцина", "Антибиотик",
    "Вирус иммунодефицита человека", "Рак", "Диабет", "Инфаркт миокарда",
    "Инсульт", "Гигиена", "Хирургия", "Терапия", "Психология", "Психиатрия",
    # История науки и общие
    "Наука", "Научный метод", "Гипотеза", "Теория", "Эксперимент",
    "История науки", "Нобелевская премия", "Альберт Эйнштейн", "Исаак Ньютон",
    "Никола Тесла", "Мария Кюри", "Чарльз Дарвин", "Дмитрий Менделеев",
    "Ломоносов, Михаил Васильевич", "Софья Ковалевская", "Анри Пуанкаре",
    "Нильс Бор", "Макс Планк", "Ричард Фейнман", "Стивен Хокинг",
    "Философия науки", "Технология", "Инженерия", "Робототехника", "Дрон",
    "Электричество", "Двигатель внутреннего сгорания", "Электромобиль",
    "Возобновляемая энергия", "Солнечная панель", "Ветряная электростанция",
    "Атомная электростанция", "Термоядерный синтез", "Нанотехнологии",
    "Космонавтика", "Космический корабль", "Международная космическая станция",
    "Спутник", "Юрий Гагарин", "Выход человека в открытый космос",
]


def http_get(url: str, retries: int = 4) -> bytes:
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(4 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")


def fetch_batch(titles: list[str]) -> dict:
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "prop": "extracts",
        "explaintext": "1",
        "redirects": "1",
        "titles": "|".join(titles),
    }
    url = API + "?" + urllib.parse.urlencode(params)
    return json.loads(http_get(url))


def clean_text(text: str) -> str:
    # Drop reference groups and stray markup artifacts; collapse whitespace.
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    chunks: list[str] = []
    seen = set()
    missing = []
    total = 0
    batch_size = 6
    batches = [TITLES[i:i + batch_size] for i in range(0, len(TITLES), batch_size)]
    for batch in batches:
        try:
            data = fetch_batch(batch)
        except Exception as exc:  # noqa: BLE001
            print(f"batch failed: {exc}", file=sys.stderr)
            time.sleep(8)
            continue
        pages = data.get("query", {}).get("pages", [])
        for page in pages:
            title = page.get("title", "")
            extract = page.get("extract") or ""
            if page.get("missing") or len(extract) < 500:
                missing.append(title)
                continue
            if title in seen:
                continue
            seen.add(title)
            body = clean_text(extract)[:MAX_CHARS_PER_ARTICLE]
            article = f"{title}\n{body}\n\n"
            chunks.append(article)
            total += len(article.encode("utf-8"))
        print(f"fetched {len(seen)} articles, {total/1e6:.2f} MB", flush=True)
        if total >= TARGET_BYTES:
            break
        time.sleep(1.5)

    corpus = "".join(chunks)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write(corpus)
    print(f"DONE: {len(seen)} articles, {len(corpus)} chars, {total/1e6:.2f} MB -> {OUT_FILE}")
    if missing:
        print(f"missing/short: {missing}", file=sys.stderr)


if __name__ == "__main__":
    main()
