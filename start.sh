#!/usr/bin/env bash
# Запуск BLOOP AI. Ставит PyTorch, если его нет, и стартует сервер.
# Улучшение: если модель ещё не обучена (нет checkpoints/mini_gpt.pt),
# сначала запускай:  python3 python/train.py --corpus data/corpus.txt
set -e
cd "$(dirname "$0")"

python3 -c "import torch" 2>/dev/null || {
  echo "Ставлю PyTorch (CPU)..."
  pip install torch --index-url https://download.pytorch.org/whl/cpu
}

if [ ! -f "checkpoints/mini_gpt.pt" ]; then
  echo "Чекпойнт мини-модели не найден — мини-модель будет недоступна."
  echo "Обучи её: python3 python/train.py --corpus data/corpus.txt"
fi

echo "Запускаю сервер..."
exec node server.js
