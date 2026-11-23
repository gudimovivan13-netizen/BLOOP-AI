# Как превратить BLOOP AI в APK и EXE

Сейчас проект — локальный веб-сервер (Node) + браузерный интерфейс. Ниже
реалистичные пути к «настольному» и «мобильному» приложению, от простого к
сложному.

## 1. Телефон уже сейчас: PWA (5 минут, без кода)

Интерфейс уже адаптивный и с манифестом `public/manifest.webmanifest`.

1. Запусти сервер на компьютере или дешёвом VPS.
2. Открой адрес в мобильном Chrome/Safari → «Добавить на главный экран».
3. Приложение откроется на весь экран как обычное приложение.

Ограничение: сервер должен работать (мини-модель требует PyTorch, мощная —
API-ключ), поэтому это «приложение + сервер», а не полностью автономная
программа.

## 2. Windows EXE: Tauri или Electron

Логика: оборачиваем `server.js` и фронтенд в нативное окно.

**Tauri (рекомендую — exe получается ~10 МБ):**

1. Установи Rust и `cargo tauri` .
2. `tauri init` — укажи `devUrl: http://localhost:3000`.
3. В конфиг сборки добавь `beforeBuildCommand` и sidecar: собери сервер в
   один исполняемый файл через `pkg server.js` (или `nexe`), положи в
   `src-tauri/binaries/`.
4. `tauri build` → на выходе `.exe`-установщик.

**Electron (проще, но exe ~150 МБ):**

1. `npm i electron electron-builder`
2. В `package.json` добавь `"main": "server.js"` и скрипт electron-сборки.
3. `electron-builder --win` → установщик `.exe`.

Мини-модель в обоих случаях: PyTorch рядом с приложением (поставить Python +
`pip install torch`, положить в папку приложения и прописать путь в
переменную `PYTHON` для `server.js`) — либо вынеси режим «Моя нейросеть» в
облако, а в exe оставь только мощную модель.

## 3. Android APK

**Вариант A — WebView-обёртка (самый быстрый путь к настоящему APK):**

1. В Android Studio создай пустой проект (Empty Views Activity).
2. `MainActivity` запускает `server.js` через bundled Node runtime
   (например, библиотека nodejs-mobile) и загружает `http://127.0.0.1:3000`
   в WebView.
3. Мини-модель: вариант «на сервере» (телефон ходит в интернет) — PyTorch на
   Android ставить не нужно. Вариант «на телефоне»: экспортируй веса
   (`torch.save` уже есть) и подключи LibTorch для Android, переписав
   `infer.py` на Kotlin/C++ — это отдельная работа на пару вечеров.

**Вариант B — Termux (без сборки APK, работает сегодня):**

```bash
pkg install nodejs python
pip install torch --index-url https://download.pytorch.org/whl/cpu
git clone <адрес репозитория> && cd bloop-ai
node server.js
# открой http://localhost:3000 в браузере телефона
```

## 4. Что рекомендую

- Телефон: начни с PWA (шаг 1) — ноль затрат, выглядит как приложение.
- Windows: Electron — за один вечер получишь установщик; Tauri — когда
  захочется «по-взрослому».
- APK: WebView-обёртка с nodejs-mobile, когда базовая версия будет нравится.
