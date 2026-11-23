#!/usr/bin/env node
/* BLOOP AI — сервер чата.
 *
 * Ноль зависимостей: только встроенные модули Node.js.
 * Два бэкенда «мощной модели»:
 *   1) z-ai CLI (если установлен) — используется по умолчанию;
 *   2) любой OpenAI-совместимый API через переменные окружения
 *      OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL.
 * Мини-модель (обучена с нуля) вызывается через python/python/infer.py.
 *
 * Запуск: node server.js  (порт по умолчанию 3000, переменная PORT)
 */
const http = require('http');
const fs = require('fs');
const os = require('os');
const path = require('path');
const crypto = require('crypto');
const { execFile } = require('child_process');

const PORT = parseInt(process.env.PORT || '3000', 10);
const ROOT = __dirname;
const PUBLIC = path.join(ROOT, 'public');
const GENERATED = path.join(ROOT, 'generated');
const PY = path.join(ROOT, 'python');
const PYTHON = process.env.PYTHON || 'python3';
const ZAI = process.env.ZAI_CLI || 'z-ai';

const OPENAI_KEY = process.env.OPENAI_API_KEY || '';
const OPENAI_BASE = (process.env.OPENAI_BASE_URL || 'https://api.openai.com/v1').replace(/\/+$/, '');
const OPENAI_MODEL = process.env.OPENAI_MODEL || 'gpt-4o-mini';

const IMAGE_SIZES = ['1024x1024', '768x1344', '864x1152', '1344x768', '1152x864', '1440x720', '720x1440'];

const SYSTEM_PROMPT =
  'Ты — BLOOP AI, дружелюбный и очень умный ассистент. Ты знаешь точные науки, программирование, ' +
  'историю и литературу, объясняешь сложное простыми словами. Отвечай на том языке, на котором пишет ' +
  'пользователь. Будь конкретным: приводи примеры, формулы и шаги, где это уместно. Не выдумывай факты: ' +
  'если чего-то не знаешь, честно скажи об этом.';

/* ---------- маленькие утилиты ---------- */

function sendJson(res, code, obj) {
  const body = JSON.stringify(obj);
  res.writeHead(code, { 'Content-Type': 'application/json; charset=utf-8' });
  res.end(body);
}

function readBody(req, limit = 15 * 1024 * 1024) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    req.on('data', (c) => {
      size += c.length;
      if (size > limit) { reject(new Error('Слишком большой запрос')); req.destroy(); return; }
      chunks.push(c);
    });
    req.on('end', () => {
      try { resolve(chunks.length ? JSON.parse(Buffer.concat(chunks).toString('utf8')) : {}); }
      catch (e) { reject(new Error('Некорректный JSON в запросе')); }
    });
    req.on('error', reject);
  });
}

function run(cmd, args, timeoutMs = 120000) {
  return new Promise((resolve) => {
    execFile(cmd, args, { timeout: timeoutMs, maxBuffer: 8 * 1024 * 1024, windowsHide: true },
      (err, stdout, stderr) => {
        resolve({ code: err && err.code != null ? err.code : (err ? 1 : 0), stdout: stdout || '', stderr: stderr || '', killed: !!(err && err.killed) });
      });
  });
}

function tmpPath(ext) {
  return path.join(os.tmpdir(), 'bloop_' + crypto.randomBytes(8).toString('hex') + ext);
}

/* ---------- бэкенды мощной модели ---------- */

async function zaiChat(system, prompt, timeout = 180000) {
  const out = tmpPath('.json');
  try {
    const r = await run(ZAI, ['chat', '--system', system, '--prompt', prompt, '--output', out], timeout);
    if (r.code !== 0) {
      throw new Error('Мощная модель недоступна (z-ai CLI): ' + (r.stderr || 'ошибка запуска').trim().slice(0, 300));
    }
    const j = JSON.parse(fs.readFileSync(out, 'utf8'));
    const content = j && j.choices && j.choices[0] && j.choices[0].message && j.choices[0].message.content;
    if (!content) throw new Error('Пустой ответ мощной модели');
    return content;
  } finally {
    try { fs.unlinkSync(out); } catch (_) {}
  }
}

async function openaiChat(system, history, timeout = 180000) {
  const messages = [{ role: 'system', content: system }, ...history];
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeout);
  try {
    const resp = await fetch(OPENAI_BASE + '/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + OPENAI_KEY },
      body: JSON.stringify({ model: OPENAI_MODEL, messages }),
      signal: ctrl.signal,
    });
    if (!resp.ok) throw new Error('OpenAI-совместимый API вернул ' + resp.status);
    const j = await resp.json();
    const content = j && j.choices && j.choices[0] && j.choices[0].message && j.choices[0].message.content;
    if (!content) throw new Error('Пустой ответ модели');
    return content;
  } finally {
    clearTimeout(timer);
  }
}

function flattenHistory(items) {
  const lastMsg = items[items.length - 1];
  if (items.length <= 1) {
    return 'Сообщение пользователя:\n' + lastMsg.content +
      '\n\nОтветь как ассистент на это сообщение прямо сейчас. Пиши сразу сам ответ, без префиксов.';
  }
  const prior = items.slice(0, -1);
  return 'История диалога:\n' + prior.map((m) => (m.role === 'user' ? 'Пользователь: ' : 'Ассистент: ') + m.content).join('\n') +
    '\n\nНовое сообщение пользователя: ' + lastMsg.content +
    '\n\nОтветь как ассистент на новое сообщение, учитывая историю. Пиши сразу сам ответ, без префиксов.';
}

async function proChat(items, systemExtra = '') {
  const system = SYSTEM_PROMPT + (systemExtra ? '\n\n' + systemExtra : '');
  if (OPENAI_KEY) return openaiChat(system, items);
  return zaiChat(system, flattenHistory(items));
}

/* ---------- отдельные возможности ---------- */

async function webSearch(query) {
  const out = tmpPath('.json');
  try {
    const r = await run(ZAI, ['function', '-n', 'web_search', '-a', JSON.stringify({ query, num: 6 }), '-o', out], 90000);
    if (r.code !== 0) throw new Error('Поиск недоступен: ' + (r.stderr || 'ошибка').trim().slice(0, 200));
    const results = JSON.parse(fs.readFileSync(out, 'utf8'));
    if (!Array.isArray(results)) throw new Error('Неожиданный формат ответа поиска');
    return results.slice(0, 5).map((x) => ({ name: x.name || x.url, url: x.url || '', snippet: (x.snippet || '').replace(/\s+/g, ' ').slice(0, 400) }));
  } finally {
    try { fs.unlinkSync(out); } catch (_) {}
  }
}

async function miniChat(message) {
  const r = await run(PYTHON, [path.join(PY, 'infer.py'), '--prompt', message, '--max-new', '300', '--temperature', '0.85'], 90000);
  let j = null;
  try { j = JSON.parse(r.stdout); } catch (_) {}
  if (j && j.ok) return j.text;
  const stderr = (r.stderr || '').trim();
  const reason = j && j.error ? j.error : (stderr || 'ошибка инференса');
  throw new Error('Мини-модель недоступна: ' + reason);
}

async function generateImage(prompt, size) {
  if (!IMAGE_SIZES.includes(size)) size = '1024x1024';
  fs.mkdirSync(GENERATED, { recursive: true });
  const file = 'img_' + Date.now() + '_' + crypto.randomBytes(3).toString('hex') + '.png';
  const outPath = path.join(GENERATED, file);
  const r = await run(ZAI, ['image', '-p', prompt, '-o', outPath, '-s', size], 300000);
  if (r.code !== 0 || !fs.existsSync(outPath)) {
    throw new Error('Генерация картинок недоступна (нужен z-ai CLI): ' + (r.stderr || '').trim().slice(0, 200));
  }
  return '/generated/' + file;
}

async function vision(question, imageUrl, imageDataUrl) {
  let arg = imageUrl || '';
  let tmpFile = null;
  if (imageDataUrl) {
    const m = /^data:image\/[a-zA-Z+]+;base64,(.+)$/.exec(imageDataUrl);
    if (!m) throw new Error('Некорректный формат изображения');
    tmpFile = tmpPath('.png');
    fs.writeFileSync(tmpFile, Buffer.from(m[1], 'base64'));
    arg = tmpFile;
  }
  const out = tmpPath('.json');
  try {
    if (!arg) throw new Error('Нужно изображение: URL или файл');
    const r = await run(ZAI, ['vision', '--prompt', question, '--image', arg, '--output', out], 240000);
    if (r.code !== 0) throw new Error('Понимание картинок недоступно (нужен z-ai CLI): ' + (r.stderr || '').trim().slice(0, 200));
    const j = JSON.parse(fs.readFileSync(out, 'utf8'));
    const content = j && j.choices && j.choices[0] && j.choices[0].message && j.choices[0].message.content;
    if (!content) throw new Error('Пустой ответ модели зрения');
    return content;
  } finally {
    try { fs.unlinkSync(out); } catch (_) {}
    if (tmpFile) { try { fs.unlinkSync(tmpFile); } catch (_) {} }
  }
}

/* ---------- HTTP-сервер ---------- */

const MIME = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8', '.png': 'image/png', '.jpg': 'image/jpeg', '.svg': 'image/svg+xml', '.ico': 'image/x-icon' };

function serveStatic(res, filePath) {
  const ext = path.extname(filePath).toLowerCase();
  res.writeHead(200, { 'Content-Type': MIME[ext] || 'application/octet-stream' });
  fs.createReadStream(filePath).pipe(res);
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, 'http://localhost');
  try {
    if (req.method === 'POST' && url.pathname === '/api/chat') {
      const body = await readBody(req);
      const mode = body.mode === 'mini' ? 'mini' : 'pro';
      const message = String(body.message || '').trim();
      if (!message) return sendJson(res, 400, { error: 'Пустое сообщение' });
      const history = Array.isArray(body.history) ? body.history.slice(-16)
        .filter((m) => m && (m.role === 'user' || m.role === 'assistant') && typeof m.content === 'string' && m.content.length < 8000) : [];

      if (mode === 'mini') {
        const text = await miniChat(message + '\n');
        return sendJson(res, 200, { text, sources: null, mode });
      }

      let systemExtra = '';
      let sources = null;
      if (body.search) {
        sources = await webSearch(message);
        const ctx = sources.map((s, i) => `[${i + 1}] ${s.name} — ${s.url}\n${s.snippet}`).join('\n\n');
        systemExtra = 'Ниже свежие результаты веб-поиска по запросу пользователя. Используй их для точности ' +
          'и в конце ответа приведи список источников в формате [1], [2].\n\nРезультаты поиска:\n' + ctx;
      }
      // убираем дубликат: текущее сообщение всегда идёт последним элементом диалога
      const items = history.slice();
      const last = items[items.length - 1];
      if (!(last && last.role === 'user' && last.content === message)) {
        items.push({ role: 'user', content: message });
      }
      const text = await proChat(items, systemExtra);
      return sendJson(res, 200, { text, sources, mode });
    }

    if (req.method === 'POST' && url.pathname === '/api/image') {
      const body = await readBody(req);
      const prompt = String(body.prompt || '').trim();
      if (!prompt) return sendJson(res, 400, { error: 'Опиши, какую картинку нарисовать' });
      const imageUrl = await generateImage(prompt, String(body.size || '1024x1024'));
      return sendJson(res, 200, { url: imageUrl, prompt });
    }

    if (req.method === 'POST' && url.pathname === '/api/vision') {
      const body = await readBody(req);
      const question = String(body.question || 'Опиши это изображение подробно.').trim();
      const text = await vision(question, body.imageUrl ? String(body.imageUrl) : '', body.image ? String(body.image) : '');
      return sendJson(res, 200, { text });
    }

    if (req.method === 'GET' && (url.pathname === '/' || url.pathname === '/index.html')) {
      return serveStatic(res, path.join(PUBLIC, 'index.html'));
    }

    if (req.method === 'GET' && (url.pathname === '/manifest.webmanifest' || url.pathname === '/favicon.ico')) {
      const p = url.pathname === '/manifest.webmanifest' ? 'manifest.webmanifest' : 'favicon.ico';
      const fp = path.join(PUBLIC, p);
      if (fs.existsSync(fp)) return serveStatic(res, fp);
      res.writeHead(404); return res.end('Not found');
    }

    if (req.method === 'GET' && url.pathname.startsWith('/generated/')) {
      const safe = path.normalize(url.pathname).replace(/^(\.\.[/\\])+/, '');
      const filePath = path.join(GENERATED, path.basename(safe));
      if (fs.existsSync(filePath)) return serveStatic(res, filePath);
      res.writeHead(404); return res.end('Not found');
    }

    res.writeHead(404, { 'Content-Type': 'application/json; charset=utf-8' });
    res.end(JSON.stringify({ error: 'Not found' }));
  } catch (err) {
    const msg = err && err.message ? err.message : 'Внутренняя ошибка сервера';
    console.error('[bloop]', url.pathname, msg);
    sendJson(res, 500, { error: msg });
  }
});

fs.mkdirSync(GENERATED, { recursive: true });
server.listen(PORT, () => {
  const backend = OPENAI_KEY ? 'OpenAI-совместимый API (' + OPENAI_MODEL + ')' : 'z-ai CLI';
  console.log('BLOOP AI запущен: http://localhost:' + PORT + '  (мощная модель: ' + backend + ')');
});
