'use strict';

/* 高分子链构象计算器 —— 前端
 *
 * 这里**不含任何物理公式**。P(h)、⟨h²⟩、h* 等全部由后端 fjc_core.py 算好返回，
 * 前端只做格式化、画图和坐标变换。归一化视图也只是一个显示变换
 * （横轴 h/h_rms，纵轴 P·h_rms），不是另一套公式。
 */

const NS = 'http://www.w3.org/2000/svg';
const SERIES_VARS = ['--series-1', '--series-2', '--series-3', '--series-4', '--series-5'];

// 画布尺寸（viewBox 坐标，实际显示宽度由 CSS 决定并等比缩放）
// top 留得宽，是为了让阈值标签排在绘图区之上，不和曲线打架
const CH = { w: 860, h: 440, top: 48, right: 106, bottom: 58, left: 78 };
const PW = CH.w - CH.left - CH.right;
const PH = CH.h - CH.top - CH.bottom;

const $ = (id) => document.getElementById(id);

let state = {
  results: null,     // 后端返回的结果数组
  normalize: false,
  hoverIdx: null,    // 键盘/鼠标当前指向的采样点
  ctx: null,         // 当前的坐标映射，供 hover 复用
};

/* ---------------- 格式化 ---------------- */

/** 物理量数值格式化：4 位有效数字，极大/极小走科学计数法。 */
function fmt(x) {
  if (x === null || x === undefined || !isFinite(x)) return '–';
  if (x === 0) return '0';
  const a = Math.abs(x);
  if (a >= 1e6 || a < 1e-3) {
    return x.toExponential(2).replace('e', 'e');
  }
  const decimals = Math.max(0, 4 - Math.floor(Math.log10(a)) - 1);
  let s = x.toFixed(Math.min(decimals, 8));
  if (s.includes('.')) s = s.replace(/0+$/, '').replace(/\.$/, '');
  return s;
}

/** 坐标轴刻度：数值小的时候统一用科学计数法，避免和 0 混排时长短不一。 */
function fmtTick(v, useExp) {
  if (v === 0) return '0';
  if (useExp) return v.toExponential(1);
  const a = Math.abs(v);
  const decimals = Math.max(0, 3 - Math.floor(Math.log10(a)) - 1);
  let s = v.toFixed(Math.min(decimals, 6));
  if (s.includes('.')) s = s.replace(/0+$/, '').replace(/\.$/, '');
  return s;
}

function seriesVar(i) {
  return `var(${SERIES_VARS[i % SERIES_VARS.length]})`;
}

function seriesName(r, i) {
  return `n = ${r.n}`;
}

/* ---------------- 后端调用 ---------------- */

function parseNs(raw) {
  const parts = String(raw).split(/[,，;；\s]+/).filter((s) => s.length > 0);
  if (parts.length === 0) throw new Error('请输入链段数 n');
  if (parts.length > 5) throw new Error('最多同时对比 5 个 n');
  return parts.map((p) => {
    const v = Number(p);
    if (!isFinite(v)) throw new Error(`「${p}」不是数字`);
    return v;
  });
}

async function compute(ns, l) {
  const res = await fetch('/api/compute', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ n: ns.length === 1 ? ns[0] : ns, l, ...modelParams() }),
  });
  const data = await res.json().catch(() => ({ error: '服务器返回了无法解析的内容' }));
  if (!res.ok) throw new Error(data.error || `请求失败（HTTP ${res.status}）`);
  return data.results;
}

/* ---------------- 链模型 ----------------
 *
 * 模型目录（标签、公式、说明、参数范围、以及 analytic 那一位）从 GET /api/models 拉 ——
 * 唯一的那份在 fjc_core.model_catalog()，JS 里不抄第二份，所以界面上这个下拉永远和服务端
 * 一致。目录里有五项：四个解析模型 + 自回避行走。参数字段按模型显隐：自由旋转链要键角、
 * 受阻旋转链再多一个 ⟨cosφ⟩、蠕虫状链改用持久长度 p、自回避行走三个都不用。
 *
 * analytic 是「这个模型有没有闭式 ⟨h²⟩」。它决定页面上一批卡片与整个数值表分区是显示还是
 * 收起（见 syncModelFields 写进 body 的 data-model-kind，样式在 style.css）——
 * 收起来的那些全部出自 /api/compute、/api/force、/api/model-curves，都要求闭式解。
 */

let MODEL_BY_KEY = new Map();

/* 页面「哪些块显示」一变（切到自回避行走会让好几个分区与卡片收起、又放出另外几个），
   scrollspy 就得重量一次栏高、再重挑一次当前分区。钩子由 initScrollSpy() 装上，
   在那之前是个空函数。
   （分区本身被收起时不用靠这个钩子：sync() 每次都现筛可见分区再挑，因为
   display:none 的元素 top 恒为 0，会永远满足「top ≤ 偏移」而霸住高亮。这里补的是
   另外两件它管不着的：栏高与滚动条长度的变化。） */
let layoutSync = () => {};

/** 当前模型 + 参数，直接读界面 —— 计算、3D、扫描三处取数共用这一个。 */
function modelParams() {
  const key = $('in-model').value || 'fjc';
  const spec = MODEL_BY_KEY.get(key);
  const out = { model: key };
  if (!spec) return out;
  const num = (id, fallback) => {
    const v = Number($(id).value);
    return Number.isFinite(v) ? v : fallback;
  };
  if (spec.params.theta_deg.used) out.theta_deg = num('in-theta', spec.params.theta_deg.default);
  if (spec.params.cos_phi.used) out.cos_phi = num('in-cosphi', spec.params.cos_phi.default);
  if (spec.params.p.used) out.p = num('in-p', spec.params.p.default);
  return out;
}

/** 当前模型在目录里的那一项（目录 = `/api/models` 给的那份，见 loadModels）。
 *  chain3d.js 要问「现在这个模型有没有闭式解」，而目录在 app.js 手里 ——
 *  所以从这里出去，和 modelParams 同一个规矩。目录还没拉回来时返回 null。 */
function modelSpec() {
  return MODEL_BY_KEY.get($('in-model').value) || null;
}

/** 在某段说明下面挂一行模型注释（没有就建一个）。
 *  力–伸长与链长扫描那两张卡最容易误读 —— 前者的曲线**不随模型变**，
 *  后者的「理论线」在非 FJC 模型下也不是 n·l²，所以各挂一行说清楚。 */
function noteUnder(anchorId, noteId, text) {
  let note = document.getElementById(noteId);
  if (!note) {
    const anchor = document.getElementById(anchorId);
    if (!anchor) return;
    note = document.createElement('p');
    note.id = noteId;
    note.className = 'chart-note';
    anchor.after(note);
  }
  note.textContent = text;
  note.hidden = !text;
}

/** 按模型显隐参数字段，并把公式与说明换成这个模型的。 */
function syncModelFields() {
  const spec = MODEL_BY_KEY.get($('in-model').value);
  if (!spec) return;
  $('theta-wrap').hidden = !spec.params.theta_deg.used;
  $('cosphi-wrap').hidden = !spec.params.cos_phi.used;
  $('p-wrap').hidden = !spec.params.p.used;
  /* 模型分两类，这一位决定页面上**哪些块适用**（见 style.css 里按 data-only 的那几条）：
     有闭式解的四个模型，力–伸长 / h→n 反解 / 模型指纹 / 数值表 / P(h) / 主结果
     全都成立；自回避行走没有闭式解，那些「按公式算」的块一条都不适用，全部收起。
     写在 body 的属性上、由 CSS 去收，而不是逐个元素设 hidden —— `.card, .hero`
     是 display:flex，会盖掉 [hidden]（这个坑 style.css 里已经踩过一次，
     见那里的 .ai-drawer[hidden]）。 */
  const kind = spec.analytic ? 'analytic' : 'simulated';
  /* 除了上面那个二分类，还要把**具体是哪个模型**写到 body 上：能量图景那张卡只有
     受阻旋转链与蠕虫状链有内容（另外两个没有能量自由度），它是按模型而不是按类别
     显隐的 —— 于是它会在 kind 不变的时候出现或消失（fjc → wlc 两边都是 analytic）。
     所以两个属性任一变了都要重挑当前分区、重摆占位虚框。 */
  const changed = document.body.dataset.modelKind !== kind
    || document.body.dataset.model !== spec.key;
  document.body.dataset.modelKind = kind;
  document.body.dataset.model = spec.key;
  if (changed) {
    layoutSync();     // 显示的东西变了，scrollspy 得重新挑一次当前分区
    // 收起的卡片要把自己的占位虚框带走、放出来的要重新要一个（读 offsetParent 会
    // 强制一次样式重算，所以上面那行属性刚写完就能读到对的可见性）。
    syncGhosts();
  }
  // 说明里带默认参数（「键角固定为 109.47°」这种），所以换成当前输入框里的值
  const now = modelParams();
  $('model-formula').textContent = spec.formula;
  $('model-note').textContent = spec.note
    .replace(String(spec.params.theta_deg.default), String(now.theta_deg ?? spec.params.theta_deg.default))
    .replace(String(spec.params.cos_phi.default), String(now.cos_phi ?? spec.params.cos_phi.default))
    .replace(String(spec.params.p.default), String(now.p ?? spec.params.p.default));

  /* 下面四张卡片的抬头都带上模型名 —— 它们的内容**都跟着模型换**，
     不写清楚的话，切完模型只能靠猜这几张图是哪条链算的（数值表还要复制出去）。 */
  const tag = `　·　${spec.label}`;
  $('chart-title').textContent = `末端距分布 P(h)${tag}`;
  $('chain-title').textContent = `链构象${tag}`;
  $('sweep-title').textContent = `链长扫描：链变长时末端距怎么变${tag}`;
  const tableTitle = document.querySelector('#sec-table .card-subtitle');
  if (tableTitle) tableTitle.textContent = `统计量一览${tag}`;

  // 力–伸长是唯一**不跟着模型换**的那张：说清它只画 FJC，别让人以为切错了
  const isFjc = spec.key === 'fjc';
  $('force-title').textContent = isFjc
    ? '力–伸长曲线'
    : '力–伸长曲线（仅适用于自由连接链）';
  noteUnder('force-desc', 'force-model-note', isFjc ? '' :
    '注意：这条曲线是 FJC 的解，和上面选的模型无关 —— 另外三个模型没有同等地位的'
    + '解析力–伸长关系（蠕虫状链的 Marko–Siggia 是插值式近似），所以这里没跟着换。'
    + '要换成对应模型的曲线，得先有那条关系式。');
  noteUnder('sweep-desc', 'sweep-model-note', isFjc ? '' :
    `当前是${spec.label}：双对数下的理论线不再是斜率 1 的直线（短链端接近 L²，`
    + '长链端才过渡到 L¹）。汇总行里的「参考斜率」是按本模型自己的 ⟨h²⟩(n) 曲线算的，'
    + '拿 1 去比会天天误报。');
}

async function loadModels(preferred) {
  const res = await fetch('/api/models');
  const data = await res.json();
  const models = data.models || [];
  MODEL_BY_KEY = new Map(models.map((m) => [m.key, m]));
  const sel = $('in-model');
  sel.textContent = '';
  models.forEach((m) => {
    const o = document.createElement('option');
    o.value = m.key;
    o.textContent = m.label;
    sel.appendChild(o);
  });
  sel.value = MODEL_BY_KEY.has(preferred) ? preferred : 'fjc';
  syncModelFields();
  // 别的模块（3D、扫描、模型指纹）也要知道「目录到齐了、当前是哪个模型」
  document.dispatchEvent(new CustomEvent('fjc:models-ready'));
}

/** 模型的读数条：本 n 下的特征比、等价 Kuhn 长度、等效 Kuhn 段数。
 *  高斯近似能不能用就看最后那个 L/b —— 它比 10 小得多时 warnings 里会说明。 */
function renderModelReadout(r) {
  const box = $('model-readout');
  if (!r || !r.params) { box.hidden = true; return; }
  box.textContent = '';
  const item = (k, v) => {
    const s = document.createElement('span');
    s.className = 'legend-item';
    const a = document.createElement('span');
    a.textContent = k;
    const b = document.createElement('span');
    b.className = 'tt-val';
    b.textContent = v;
    s.append(a, b);
    box.appendChild(s);
  };
  item('Cn（本 n 下）', fmt(r.Cn));
  item('等价 Kuhn 长度 b', `${fmt(r.kuhn_length)} l`);
  item('等效 Kuhn 段数 L/b', fmt(r.n_kuhn));
  box.hidden = false;
}

/* ---------------- n 的预设快捷值 ---------------- */

// 快捷值。跨三个数量级，多选两三个就能看出分布随 n 变宽/变窄。
const N_PRESETS = [10, 30, 100, 300, 1000, 3000, 10000];

// n 输入框里的值 —— 和 parseNs 同一套分隔符，但不校验，只用来点亮 chip
function nValues() {
  return String($('in-n').value)
    .split(/[,，;；\s]+/)
    .filter((s) => s.length > 0)
    .map(Number);
}

/** 把当前 n 输入框里的值反映到 chip 的选中态上。 */
function syncNChips() {
  const cur = new Set(nValues().filter((v) => isFinite(v)));
  document.querySelectorAll('#n-chips .chip').forEach((b) => {
    const on = cur.has(Number(b.dataset.n));
    b.classList.toggle('on', on);
    b.setAttribute('aria-pressed', on ? 'true' : 'false');
  });
}

/** 点 chip：已在框里就移出，不在就加入。 */
function toggleNPreset(v) {
  const input = $('in-n');
  const cur = nValues().filter((x) => isFinite(x));
  let next = cur.some((x) => x === v) ? cur.filter((x) => x !== v) : [...cur, v];
  // 一个都不留，上面那张图就没东西可画了 —— 所以最后一个点不掉
  if (next.length === 0) return;
  // 最多 5 条曲线。超出了就把最早加进来的挤出去，好过弹一条错误
  if (next.length > 5) next = next.slice(next.length - 5);
  input.value = next.join(', ');
  // 走 input 事件，复用现成的防抖刷新
  input.dispatchEvent(new Event('input', { bubbles: true }));
}

function renderNChips() {
  const box = $('n-chips');
  if (!box) return;
  N_PRESETS.forEach((v) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'chip';
    b.dataset.n = String(v);
    b.textContent = String(v);
    b.setAttribute('aria-pressed', 'false');
    b.addEventListener('click', () => toggleNPreset(v));
    box.appendChild(b);
  });
}

/* ---------------- 提示条 ---------------- */

function setAlerts(errorMsg, warnings) {
  renderAlerts('alerts', errorMsg, warnings);
}

/** 往指定的提示条容器里写错误/警告。sweep.js 复用同一个（见 s-alerts）。 */
function renderAlerts(boxId, errorMsg, warnings) {
  const box = $(boxId);
  if (!box) return;
  box.textContent = '';

  const add = (kind, tagText, icon, text) => {
    const d = document.createElement('div');
    d.className = `alert alert-${kind}`;
    const i = document.createElement('span');
    i.className = 'icon';
    i.textContent = icon;
    i.setAttribute('aria-hidden', 'true');
    const t = document.createElement('span');
    t.className = 'tag';
    t.textContent = tagText;
    const m = document.createElement('span');
    m.textContent = text;
    d.append(i, t, m);
    box.appendChild(d);
  };

  if (errorMsg) add('error', '错误', '✕', errorMsg);
  (warnings || []).forEach((w) => add('warn', '注意', '⚠', w));
}

/* ---------------- 主结果 + 统计量卡片 ---------------- */

const TILES = [
  { key: 'h2', label: '均方末端距 ⟨h²⟩', formula: 'n·l²' },
  { key: 'h_mp', label: '最可几末端距 h*', formula: 'l·√(2n/3)' },
  { key: 'h_mean', label: '平均末端距 ⟨h⟩', formula: 'l·√(8n/(3π))' },
  { key: 'sigma', label: '分布标准差 σ', formula: 'l·√(n/3)' },
  { key: 'Rg_rms', label: '回转半径 Rg', formula: '√(n·l²/6)' },
  { key: 'h_max', label: '全伸展长度', formula: 'n·l' },
  { key: 'Cn', label: '特征比 Cn', formula: '⟨h²⟩/(n·l²) ≡ 1' },
];

function renderHero(r, multi) {
  $('hero-label').textContent = multi
    ? `根均方末端距 h —— 取第一个 n = ${r.n}`
    : `根均方末端距 h　·　${(MODEL_BY_KEY.get(r.model) || {}).label || ''}`;
  $('hero-value').textContent = fmt(r.h_rms);
  const tail = multi ? '　（对比模式下主结果取第一个 n，其余见下表）' : '';
  // FJC 保留那条最认得出的式子；其余模型把「等效 Kuhn 长度」摆出来 ——
  // 四个模型真正的差别就在这个数（以及 WLC 短链端那条过渡）
  $('hero-formula').textContent = r.model === 'fjc'
    ? `h = l·√n = ${fmt(r.l)} × √${r.n} = ${fmt(r.h_rms)}` + tail
    : `h = √⟨h²⟩ = ${fmt(r.h_rms)}　|　Cn = ${fmt(r.Cn)}，b = ${fmt(r.kuhn_length)} l` + tail;
}

function renderTiles(r, multi) {
  const box = $('tiles');
  box.textContent = '';

  if (multi) {
    const cap = document.createElement('p');
    cap.className = 'tiles-caption';
    cap.textContent = `以下为 n = ${r.n} 的统计量（对比模式下取第一个 n；各 n 的完整数值见下方表格）`;
    cap.style.gridColumn = '1 / -1';
    cap.style.margin = '0';
    cap.style.fontSize = '12.5px';
    cap.style.color = 'var(--text-muted)';
    box.appendChild(cap);
  }

  TILES.forEach((t) => {
    const d = document.createElement('div');
    d.className = 'tile';
    // 关键：文字不穿数据色，身份由旁边的色块承担
    d.innerHTML =
      `<div class="tile-label">${t.label}</div>` +
      `<div class="tile-value">${fmt(r[t.key])}</div>` +
      `<div class="tile-formula">${t.formula}</div>`;
    box.appendChild(d);
  });
}

/* ---------------- 图表 ---------------- */

function el(tag, attrs, parent) {
  const n = document.createElementNS(NS, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(n);
  return n;
}

/** 取一组「好看」的刻度值：步长落在 1/2/5×10^k 上，且最上一格必须盖住数据。 */
function niceTicks(max, count = 4) {
  if (!(max > 0)) return { ticks: [0], top: 1 };
  const raw = max / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10) * mag;
  const ticks = [];
  for (let i = 0; i * step <= max * 1.000001; i++) ticks.push(i * step);
  // 末格若落在数据之下，补一格 —— 否则曲线会顶破最高的那条网格线
  if (ticks[ticks.length - 1] < max) ticks.push(ticks[ticks.length - 1] + step);
  if (ticks.length < 2) ticks.push(step);
  return { ticks, top: ticks[ticks.length - 1] };
}

function drawChart(results, normalize) {
  const svg = $('chart');
  svg.textContent = '';

  const multi = results.length > 1;

  // 归一化：横轴按各自的 h_rms 缩放，纵轴乘回 h_rms。
  // 这样各 n 的曲线会精确重合（√n 标度律在整个分布上都成立）。
  const xy = results.map((r) => {
    if (!normalize) return { x: r.curve.h, y: r.curve.P, r };
    return {
      x: r.curve.h.map((h) => h / r.h_rms),
      y: r.curve.P.map((p) => p * r.h_rms),
      r,
    };
  });

  const xMax = Math.max(...xy.map((c) => c.x[c.x.length - 1]));
  const yMax = Math.max(...xy.map((c) => Math.max(...c.y)));

  const xt = niceTicks(xMax, 4);
  const yt = niceTicks(yMax, 4);
  const xTop = xt.top;
  const yTop = yt.top;

  const sx = (v) => CH.left + (v / xTop) * PW;
  const sy = (v) => CH.top + PH - (v / yTop) * PH;

  const useExpY = yTop < 0.01;

  // --- 网格（实线细发丝线）---
  const grid = el('g', {}, svg);
  yt.ticks.forEach((v) => {
    el('line', {
      class: 'grid-line', x1: CH.left, x2: CH.left + PW, y1: sy(v), y2: sy(v),
    }, grid);
    el('text', {
      class: 'tick-text', x: CH.left - 10, y: sy(v) + 4, 'text-anchor': 'end',
    }, grid).textContent = fmtTick(v, useExpY);
  });
  xt.ticks.forEach((v) => {
    el('text', {
      class: 'tick-text', x: sx(v), y: CH.top + PH + 20, 'text-anchor': 'middle',
    }, grid).textContent = fmtTick(v, false);
  });

  // --- 坐标轴 ---
  // 横轴 + 竖轴都画：只有底下那条线时，读数只能靠网格推，眼睛没有落点
  el('line', {
    class: 'axis-line', x1: CH.left, x2: CH.left + PW, y1: CH.top + PH, y2: CH.top + PH,
  }, svg);
  el('line', {
    class: 'axis-line', x1: CH.left, x2: CH.left, y1: CH.top, y2: CH.top + PH,
  }, svg);

  el('text', {
    class: 'axis-title', x: CH.left + PW / 2, y: CH.h - 14, 'text-anchor': 'middle',
  }, svg).textContent = normalize ? 'h / h_rms' : '末端距 h';

  el('text', {
    class: 'axis-title', x: 0, y: 0, 'text-anchor': 'middle',
    transform: `translate(18, ${CH.top + PH / 2}) rotate(-90)`,
  }, svg).textContent = normalize ? 'P(h) · h_rms' : 'P(h)';

  // --- 阈值参考线 ---
  // 单条曲线时按该 n 的实际值画；归一化后这些位置与 n 无关，多条也只需一组。
  let refs = [];
  if (normalize) {
    const r = results[0];
    refs = [
      { v: r.h_mp / r.h_rms, label: 'h*' },
      { v: r.h_mean / r.h_rms, label: '⟨h⟩' },
      { v: 1, label: 'h_rms' },
    ];
  } else if (!multi) {
    const r = results[0];
    refs = [
      { v: r.h_mp, label: 'h*' },
      { v: r.h_mean, label: '⟨h⟩' },
      { v: r.h_rms, label: 'h_rms' },
    ];
  }

  const refG = el('g', {}, svg);
  // 三个阈值（h*、⟨h⟩、h_rms）在横轴上本来就近，贪心分行保证标签互不重叠；
  // 标签一律排在绘图区上方，不压曲线。
  const lastXByRow = [-Infinity, -Infinity, -Infinity];
  refs.forEach((ref) => {
    const x = sx(ref.v);
    if (x > CH.left + PW + 1) return;
    el('line', {
      class: 'ref-line', x1: x, x2: x, y1: CH.top, y2: CH.top + PH,
    }, refG);
    let row = lastXByRow.findIndex((lx) => x - lx >= 48);
    if (row === -1) row = lastXByRow.length - 1;
    lastXByRow[row] = x;
    const anchor = x > CH.left + PW - 54 ? 'end' : 'start';
    el('text', {
      class: 'ref-text', x: anchor === 'end' ? x - 4 : x + 4,
      y: 15 + row * 14, 'text-anchor': anchor,
    }, refG).textContent = ref.label;
  });

  // --- 曲线（2px，圆角接头）---
  const lineG = el('g', {}, svg);
  xy.forEach((c, i) => {
    let d = '';
    for (let k = 0; k < c.x.length; k++) {
      d += (k === 0 ? 'M' : 'L') + sx(c.x[k]).toFixed(2) + ',' + sy(c.y[k]).toFixed(2);
    }
    el('path', { class: 'series-line', d, stroke: seriesVar(i) }, lineG);
    // 末端点：8px（r=4）+ 2px 表面色圆环
    const last = c.x.length - 1;
    el('circle', {
      class: 'series-dot', cx: sx(c.x[last]), cy: sy(c.y[last]), r: 4, fill: seriesVar(i),
    }, lineG);
  });

  // --- 悬停层 ---
  const hoverG = el('g', { id: 'hover-layer' }, svg);
  el('rect', {
    class: 'hover-surface', x: CH.left, y: CH.top, width: PW, height: PH,
    id: 'hover-surface',
  }, hoverG);

  state.ctx = { xy, sx, sy, xTop, yTop, multi, normalize };

  // 说明文字随视图切换：归一化视图里曲线重合本身就是结论，要点明
  $('chart-desc').textContent = normalize
    ? '归一化显示：横轴按各自的 h_rms 缩放、纵轴乘回 h_rms。各 n 的曲线在此精确重合，'
      + '说明 √n 标度律在整个分布上都成立，而不只是均方值 ⟨h²⟩=n·l² 成立。'
      + '纵向虚线落在 h/h_rms = √(2/3)、√(8/3π)、1 处，与 n 无关。'
      + '鼠标悬停或聚焦图表后按 ←/→ 可读数。'
    : 'P(h) 是末端距的径向分布（三维），曲线下面积为 1。纵向虚线标出最可几末端距 h*、'
      + '平均末端距 ⟨h⟩ 与根均方末端距 h_rms。对比多个 n 时不再画参考线（会互相压住），'
      + '改用图例与下方表格。鼠标悬停或聚焦图表后按 ←/→ 可读数。';

  if (state.hoverIdx !== null) updateHover(state.hoverIdx);
}

/** 把鼠标/键盘位置换算成数据横坐标，并对每条曲线各取最近采样点。 */
function sampleAt(dataX) {
  const { xy, normalize } = state.ctx;
  const rows = xy.map((c, i) => {
    let best = 0;
    let bestD = Infinity;
    for (let k = 0; k < c.x.length; k++) {
      const d = Math.abs(c.x[k] - dataX);
      if (d < bestD) { bestD = d; best = k; }
    }
    return {
      i, r: c.r, idx: best, x: c.x[best], y: c.y[best],
      hLabel: normalize ? c.r.curve.h[best] : c.x[best],
    };
  });
  return rows;
}

function updateHover(dataX) {
  const svg = $('chart');
  const layer = $('hover-layer');
  if (!layer || !state.ctx) return;

  const { sx, sy, xTop } = state.ctx;
  [...layer.querySelectorAll('.crosshair, .hover-dot')].forEach((n) => n.remove());

  if (dataX === null || dataX === undefined) {
    $('tooltip').hidden = true;
    return;
  }

  const rows = sampleAt(dataX);
  const x = sx(rows[0].x);

  el('line', {
    class: 'crosshair', x1: x, x2: x, y1: CH.top, y2: CH.top + PH,
  }, layer);
  rows.forEach((row) => {
    if (row.i >= 5) return;
    el('circle', {
      class: 'series-dot hover-dot', cx: sx(row.x), cy: sy(row.y), r: 4,
      fill: seriesVar(row.i),
    }, layer);
  });

  const tt = $('tooltip');
  tt.textContent = '';

  const head = document.createElement('div');
  head.className = 'tt-head';
  head.textContent = state.ctx.normalize
    ? `h / h_rms = ${fmt(rows[0].x)}`
    : `h = ${fmt(rows[0].x)}`;
  tt.appendChild(head);

  rows.forEach((row) => {
    const d = document.createElement('div');
    d.className = 'tt-row';
    const k = document.createElement('span');
    k.className = 'tt-key';
    k.style.background = seriesVar(row.i);
    const name = document.createElement('span');
    name.className = 'tt-name';
    name.textContent = `n = ${row.r.n}`;
    const val = document.createElement('span');
    val.className = 'tt-val';
    val.textContent = `P = ${fmt(row.y)}`;
    d.append(k, name, val);
    tt.appendChild(d);
  });

  // SVG 坐标 → 屏幕像素（图表按容器宽度等比缩放）
  const rect = svg.getBoundingClientRect();
  const scale = rect.width / CH.w;
  const px = x * scale;
  tt.hidden = false;
  const ttW = tt.offsetWidth;
  const left = px + 14 + ttW > rect.width ? px - 14 - ttW : px + 14;
  tt.style.left = `${Math.max(0, left)}px`;
  tt.style.top = `${Math.min(Math.max(0, sy(rows[0].y) * scale - 10), rect.height - tt.offsetHeight)}px`;
}

/* ---------------- 图例与表格 ---------------- */

function renderLegend(results) {
  const box = $('legend');
  box.textContent = '';
  // 单条曲线不需要图例 —— 标题已经说明画的是什么
  if (results.length < 2) { box.hidden = true; return; }
  box.hidden = false;
  results.forEach((r, i) => {
    const d = document.createElement('span');
    d.className = 'legend-item';
    const k = document.createElement('span');
    k.className = 'legend-key';
    k.style.background = seriesVar(i);
    const t = document.createElement('span');
    t.textContent = seriesName(r, i);
    d.append(k, t);
    box.appendChild(d);
  });
}

const COLS = [
  { head: 'n', get: (r) => String(r.n), key: true },
  { head: 'h_rms（h）', get: (r) => fmt(r.h_rms) },
  { head: '⟨h²⟩', get: (r) => fmt(r.h2) },
  { head: 'h*', get: (r) => fmt(r.h_mp) },
  { head: '⟨h⟩', get: (r) => fmt(r.h_mean) },
  { head: 'σ', get: (r) => fmt(r.sigma) },
  { head: 'Rg', get: (r) => fmt(r.Rg_rms) },
  { head: '全伸展 nl', get: (r) => fmt(r.h_max) },
  { head: 'Cn', get: (r) => fmt(r.Cn) },
];

function renderTable(results) {
  const table = $('table');
  [...table.children].forEach((c) => c.remove());

  const thead = document.createElement('thead');
  const hr = document.createElement('tr');
  COLS.forEach((c) => {
    const th = document.createElement('th');
    th.scope = 'col';
    th.textContent = c.head;
    hr.appendChild(th);
  });
  thead.appendChild(hr);
  table.appendChild(thead);

  const tbody = document.createElement('tbody');
  results.forEach((r, i) => {
    const tr = document.createElement('tr');
    COLS.forEach((c) => {
      const td = document.createElement('td');
      if (c.key) {
        if (results.length > 1) {
          const k = document.createElement('span');
          k.className = 'row-key';
          k.style.background = seriesVar(i);
          td.appendChild(k);
        }
        td.appendChild(document.createTextNode(String(r.n)));
      } else {
        td.textContent = c.get(r);
      }
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  state.results = results;
}

/* ---------------- 导出：下载 + SVG 转 PNG ----------------
 *
 * 这两个函数放在 app.js（它最先加载），并挂到 window.fjcExport 上给 sweep.js 用
 * —— 三张图各写一份序列化逻辑，改一处忘两处，最后三张图导出的样子还不一样。
 *
 * SVG → PNG 的唯一麻烦是：序列化出来的 SVG 是个**独立文件**，它看不到 style.css，
 * 也看不到 --series-* 这些自定义属性（自定义属性只在 CSS 里活，序列化成属性值
 * 会原样写成 `var(--series-1)`，图片里就是黑色）。所以必须先把每个元素**算出来的**
 * 样式抄成行内样式，再序列化。 */
const SVG_STYLE_PROPS = [
  'fill', 'stroke', 'stroke-width', 'stroke-dasharray', 'stroke-linecap',
  'stroke-linejoin', 'stroke-opacity', 'fill-opacity', 'opacity',
  'font-size', 'font-family', 'font-weight', 'font-style', 'text-anchor',
  'letter-spacing', 'visibility',
];

function download(blob, name) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** 导出 PNG 要垫的底色：**必须是不透明的**。
 *  面板是玻璃质感（半透明 + 背景模糊），照它自己的 background-color 垫底
 *  会得到一张半透明的图 —— 贴进 PPT 就透出下面的花纹，正是当初要垫底色的理由。
 *  所以读的是每个主题下专门给的 --png-bg；没有再退回面板自己的颜色。 */
function opaqueBg(el) {
  const root = document.querySelector('.viz-root');
  const v = root && getComputedStyle(root).getPropertyValue('--png-bg').trim();
  if (v) return v;
  const cs = el ? getComputedStyle(el).backgroundColor : '';
  return cs && cs !== 'rgba(0, 0, 0, 0)' ? cs : '#ffffff';
}

async function svgToPng(svg, name, scale = 2) {
  const vb = svg.viewBox && svg.viewBox.baseVal;
  const w = (vb && vb.width) || svg.clientWidth || 860;
  const h = (vb && vb.height) || svg.clientHeight || 440;

  const clone = svg.cloneNode(true);
  clone.setAttribute('xmlns', NS);
  clone.setAttribute('width', w);
  clone.setAttribute('height', h);

  // 1) 抄样式：原树和克隆树的节点是逐一对应的，按序号配对即可
  const src = [svg, ...svg.querySelectorAll('*')];
  const dst = [clone, ...clone.querySelectorAll('*')];
  src.forEach((node, i) => {
    const d = dst[i];
    if (!d || d.nodeType !== 1) return;
    const cs = getComputedStyle(node);
    let inline = '';
    SVG_STYLE_PROPS.forEach((p) => {
      const v = cs.getPropertyValue(p);
      if (v) inline += `${p}:${v};`;
    });
    d.setAttribute('style', inline);
  });

  // 2) 垫一张底色，否则导出的 PNG 是透明的 —— 贴进 PPT 会露出下面的花纹
  const holder = svg.closest('.card') || svg.parentElement;
  const bgcs = opaqueBg(holder);
  const bg = document.createElementNS(NS, 'rect');
  bg.setAttribute('x', 0); bg.setAttribute('y', 0);
  bg.setAttribute('width', w); bg.setAttribute('height', h);
  bg.setAttribute('fill', bgcs && bgcs !== 'rgba(0, 0, 0, 0)' ? bgcs : '#ffffff');
  clone.insertBefore(bg, clone.firstChild);

  const xml = new XMLSerializer().serializeToString(clone);
  const url = URL.createObjectURL(
    new Blob(['<?xml version="1.0" encoding="UTF-8"?>\n' + xml],
      { type: 'image/svg+xml;charset=utf-8' })
  );

  const img = new Image();
  // 返回 Promise：成功时带 blob、失败时 reject —— 按钮不在乎，
  // 但静默失败（onerror 里只 revoke 不吭声）排查起来太贵。
  return new Promise((resolve, reject) => {
    img.onload = () => {
      URL.revokeObjectURL(url);
      const cv = document.createElement('canvas');
      cv.width = Math.round(w * scale);
      cv.height = Math.round(h * scale);
      const ctx = cv.getContext('2d');
      ctx.drawImage(img, 0, 0, cv.width, cv.height);
      cv.toBlob((b) => {
        if (b) { download(b, name); resolve(b); }
        else reject(new Error('canvas.toBlob 返回空'));
      }, 'image/png');
    };
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error('SVG 光栅化失败')); };
    img.src = url;
  });
}

window.fjcExport = { download, svgToPng, pngFail };

/** 导出失败得说出来 —— 按钮点了没反应是最难查的一类问题。
 *  追加而不是复用 renderAlerts：那条会清空整块，不该把刚算出来的警告冲掉。 */
function pngFail(boxId, err) {
  const box = $(boxId);
  if (!box) return;
  const d = document.createElement('div');
  d.className = 'alert alert-error';
  const i = document.createElement('span');
  i.className = 'icon'; i.textContent = '✕'; i.setAttribute('aria-hidden', 'true');
  const t = document.createElement('span');
  t.className = 'tag'; t.textContent = '错误';
  const m = document.createElement('span');
  m.textContent = `导出 PNG 失败：${err && err.message ? err.message : err}`;
  d.append(i, t, m);
  box.appendChild(d);
}

/** 给一串文件名用的 n 段：`100` 或 `100_300`（文件名里不能有逗号/空格）。 */
function nStem() {
  return nValues().filter((v) => isFinite(v)).join('_') || 'x';
}

/* ---------------- 水波纹 ---------------- */

const RIPPLE_SEL = '.btn-primary, .tonal-btn, .outlined-btn, .ghost-btn, .chip, .section-tab, .ai-x';

function initRipple() {
  // 委托到 document：按钮是动态生成的（n 的预设、Kuhn 预置），逐个绑会漏
  document.addEventListener('pointerdown', (ev) => {
    const host = ev.target && ev.target.closest ? ev.target.closest(RIPPLE_SEL) : null;
    if (!host || host.disabled) return;
    const r = host.getBoundingClientRect();
    const size = Math.max(r.width, r.height) * 1.6;
    const span = document.createElement('span');
    span.className = 'ripple';
    span.style.width = `${size}px`;
    span.style.height = `${size}px`;
    span.style.left = `${ev.clientX - r.left - size / 2}px`;
    span.style.top = `${ev.clientY - r.top - size / 2}px`;
    host.appendChild(span);
    setTimeout(() => span.remove(), 520);
  }, { passive: true });
}

/* ---------------- 吸顶高度 + 分区 scrollspy ---------------- */

function initScrollSpy() {
  const bar = $('app-bar');
  const nav = $('section-tabs');
  const tabs = [...document.querySelectorAll('.section-tab')];
  const secs = tabs.map((t) => $(t.dataset.target)).filter(Boolean);
  if (!bar || !secs.length) return;

  // 分区导航有两种形态（见 style.css 的 .side-nav）：
  //   宽屏 —— 左侧竖排 rail，不占竖直空间，吸顶的只有 App Bar；
  //   窄屏 —— 顶部横条，压在 App Bar 上面，两者叠起来才是遮挡高度。
  // 所以 --nav-h 在宽屏记 0、窄屏记条高；scroll-margin 和点击偏移都用它 + --bar-h。
  // 用 flexDirection 判形态，比比坐标稳：sticky 状态下 getBoundingClientRect()
  // 会随滚动变，flex 方向不会。
  let stickyTop = bar.offsetHeight;
  const measure = () => {
    const railOnSide = !!nav && getComputedStyle(nav).flexDirection === 'column';
    const navH = nav && !railOnSide ? nav.offsetHeight : 0;
    stickyTop = bar.offsetHeight + navH;
    document.documentElement.style.setProperty('--bar-h', `${bar.offsetHeight}px`);
    document.documentElement.style.setProperty('--nav-h', `${navH}px`);
  };
  measure();

  let ticking = false;
  const sync = () => {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(() => {
      ticking = false;
      const offset = stickyTop + 24;
      /* 只数**看得见**的分区。被收起的分区（自回避行走下没有数值表那节）
         display:none，它的 getBoundingClientRect().top 恒为 0，于是永远满足
         `top <= offset`，会把高亮一直霸在自己身上。现算一遍可见列表最省事 ——
         比再挂一个「模型换了要重挑」的钩子稳。 */
      const vis = secs.filter((s) => s.offsetParent !== null);
      if (!vis.length) return;
      let cur = vis[0];
      vis.forEach((s) => { if (s.getBoundingClientRect().top <= offset) cur = s; });
      // 滚到底时最后一节可能永远没越过阈值，直接钉到最后一个**看得见的**
      if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 6) {
        cur = vis[vis.length - 1];
      }
      tabs.forEach((t) => {
        const on = t.dataset.target === cur.id;
        t.classList.toggle('on', on);
        if (on) t.setAttribute('aria-current', 'true');
        else t.removeAttribute('aria-current');
      });
      bar.classList.toggle('stuck', window.scrollY > 4);
      if (nav) nav.classList.toggle('stuck', window.scrollY > 4);
    });
  };

  window.addEventListener('scroll', sync, { passive: true });
  window.addEventListener('resize', () => { measure(); sync(); });
  // 只听 window.resize 会漏：抽屉开合、标题换行、以及**面板从隐藏变可见**都会改栏高，
  // 而这些一个 resize 事件都不发。ResizeObserver 直接盯盒子本身，补上这条。
  // 往回写 --bar-h / --nav-h 不改变这两个盒子的尺寸（一个只影响 scroll-margin、
  // 一个只影响 .app-bar 的 top），所以不会形成回调环。
  if (typeof ResizeObserver === 'function') {
    const ro = new ResizeObserver(() => { measure(); sync(); });
    ro.observe(bar);
    if (nav) ro.observe(nav);
  }
  window.addEventListener('load', () => { measure(); sync(); });
  // 换模型会让一批分区/卡片收起或出现（见 syncModelFields 里的 data-model-kind）——
  // 那时栏高和「哪些分区看得见」都变了，得重量一次再重挑一次。
  layoutSync = () => { measure(); sync(); };
  sync();

  tabs.forEach((t) => t.addEventListener('click', (ev) => {
    ev.preventDefault();
    const el = $(t.dataset.target);
    if (!el) return;
    // 抽屉开合会改正文宽度、标题跟着换行，栏高不一定还是上次量的值
    measure();
    window.scrollTo({
      top: el.getBoundingClientRect().top + window.scrollY - (stickyTop + 12),
      behavior: 'smooth',
    });
  }));
}

/* ---------------- 模块尺寸与位置（拖右下角改大小、抓标题栏换位置） ----------------
 *
 * 尺寸用原生 CSS 的 resize（取舍写在 style.css 那一段里），位置用指针事件自己拖。
 * 这里补三件浏览器不管的事：
 *
 *  1. **拖动之后放开宽度上限**。卡片本来是「跟着内容定宽」的（图多宽卡片就多宽，
 *     免得模块里出现大片空白）。那个上限会挡住用户拖宽，所以第一次真正拖动之后就撤掉它。
 *  2. **记住**。浏览器不会为一次拖拽发事件，所以用 ResizeObserver 看尺寸变化、防抖后
 *     写进 localStorage；下次打开按同样的大小摆回去。
 *  3. **换位置**。抓 `.card-head`（主结果那条没有标题栏，用它的标签行）拖动，落点是
 *     **同一个父容器里的兄弟卡片** —— 「只在所属分区内挪」是靠这条约束保证的：`.pair` /
 *     `.layout` 这类成组容器各自算一格，卡片拖不出自己的组，并排的两张也就不会被拆散。
 *     顺序同样记进 localStorage。
 *
 * 键：init 时给每张卡挂一个 `data-card-key`（有 id 用 id，没有的用 anon#N 按文档顺序编号）。
 * 之所以不临时算序号：用户拖动之后卡片的序号会变，键必须**在第一次拖动之前就钉死**。
 */
const CARD_SIZE_KEY = 'fjc-card-sizes';

function cardEls() {
  return Array.from(document.querySelectorAll('.card, .hero'));
}

/** 当前**真的显示着**的卡片。
 *
 *  切到自回避行走时会有一批卡片被收起来（display:none），而它们并没有从 DOM 里消失。
 *  要读隐藏卡的**布局**就得避开它：隐藏元素的 getBoundingClientRect() 全是 0，
 *  拿它算出来的偏移量是错的。目前只有 reclampCards() 需要这一份。
 *
 *  **多数地方仍要用全量的 cardEls()**，而且理由各不相同：
 *  saveCardPos / applyCardPos 是从零重建整份记录，按可见性过滤会把隐藏卡已存的
 *  偏移悄悄删掉；syncGhosts 正相反，它得遍历到隐藏卡才能把那张卡留下的虚框**扫掉**
 *  （见那里的注释）；拖动的命中与取尺寸都是 pointerdown 打在卡片自己身上，
 *  能点到就一定是显示着的。
 */
function visibleCards() {
  return cardEls().filter((c) => c.offsetParent !== null);
}

/* 键：init 时钉死（有 id 用 id，其余按文档顺序编 anon#N）。
   不能临时算序号 —— 用户拖过之后卡片序号就变了，键必须在第一次拖动之前固定。 */
function initCardKeys() {
  let anon = 0;
  cardEls().forEach((el) => { el.dataset.cardKey = el.id || `anon#${anon++}`; });
}

/** 同一个父容器里的卡片兄弟 —— 换位只发生在这些兄弟之间。 */
function cardSiblings(el) {
  const parent = el.parentElement;
  if (!parent) return [el];
  return Array.from(parent.children)
    .filter((c) => c.classList.contains('card') || c.classList.contains('hero'));
}

function applyCardSizes() {
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem(CARD_SIZE_KEY) || '{}') || {}; } catch { saved = {}; }
  cardEls().forEach((el) => {
    const s = saved[el.dataset.cardKey];
    if (!s) return;
    if (s.w) { el.style.width = `${s.w}px`; el.style.maxWidth = 'none'; }
    if (s.h) el.style.height = `${s.h}px`;
  });
}

/** 抓标题栏拖动换位：拖动 = 把卡片在当前分区里挪个位置。
 *  • 只挪卡片自己，布局不动 —— 下面的卡片**不会**自动补位挤上来（见下面的「自由摆放」）。
 *  • 双击卡片标题，或者页脚的「恢复默认布局」，让卡片回到自己那一格。 */
function initCardDrag() {
  cardEls().forEach((card) => {
    const handle = card.querySelector(':scope > .card-head')
      || card.querySelector(':scope > .hero-label');
    if (!handle) return;
    handle.title = '拖动可挪位置（只在当前分区内），双击回到原位';
    let drag = null;

    const onMove = (ev) => {
      if (!drag) return;
      ev.preventDefault();
      freeCard(drag.card);
      // 活动范围每次现算：拖动当中窗口、字号变了也不会跑出界
      const b = dragBounds(drag.card);
      const rawX = ev.clientX - drag.grabX - b.flowLeft;
      const rawY = ev.clientY - drag.grabY - b.flowTop;
      drag.card.style.left = `${Math.round(Math.min(b.maxX, Math.max(b.minX, rawX)))}px`;
      drag.card.style.top = `${Math.round(Math.min(b.maxY, Math.max(b.minY, rawY)))}px`;
      drag.moved = true;
      slotGhost(drag.card);            // 原地留个虚框：这一格还是它的
    };

    const onUp = () => {
      if (!drag) return;
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      const card0 = drag.card;
      const { moved } = drag;
      drag = null;
      card0.classList.remove('dragging');
      // 只是点了一下（没挪）就别留下 relative + 抬起的痕迹
      if (!moved) homeCard(card0, { tidy: false });
      syncGhosts();
      if (moved) saveCardPos();
    };

    handle.addEventListener('pointerdown', (ev) => {
      if (ev.button !== 0) return;
      // 标题栏里的按钮、下拉、链接各自有用，别把它们也拖走
      if (ev.target.closest('button, a, input, select, [role="button"]')) return;
      const r = card.getBoundingClientRect();
      drag = {
        card,
        moved: false,
        grabX: ev.clientX - r.left,   // 手指/指针相对卡片左上角的偏移：抓住哪儿就从哪儿拖
        grabY: ev.clientY - r.top,
      };
      card.classList.add('dragging');
      window.addEventListener('pointermove', onMove);
      window.addEventListener('pointerup', onUp);
    });

    // 双击标题 = 收回原位（拖歪了不用拿尺子找像素）
    handle.addEventListener('dblclick', (ev) => {
      if (ev.target.closest('button, a, input, select, [role="button"]')) return;
      if (!card.style.left && !card.style.top) return;
      homeCard(card);
      saveCardPos();
    });
  });
}

/* ---------- 在分区内自由摆放 ----------
 * 位置 = position: relative + left/top 的**相对偏移**。
 *  ① relative 的元素仍然占着自己原来那一格 —— 卡片被拖走的时候，下面的卡片
 *    不会自动补位挤上来。（上一版用 position: absolute，卡片一脱离文档流，
 *    下面的内容立刻上移，看着就像在「跳」，用户反馈的奇怪就是这么来的。）
 *  ② 拖走的卡片原地留一个虚线空框（.card-ghost）：这一格还是它的。
 *    双击卡片标题、点「恢复默认布局」都能把它收回去。
 *  ③ 只存 { x, y } 两个相对偏移，都是 0 就不存 —— 默认布局下 localStorage 里是空的。
 */
const CARD_POS_KEY = 'fjc-card-pos2';   // 键换代：上一版存的是绝对坐标，语义不同

function sectionOf(card) {
  return card.closest('.page-section') || document.body;
}

function cardOffset(card) {
  return { x: parseFloat(card.style.left) || 0, y: parseFloat(card.style.top) || 0 };
}

/** 让卡片进入「自由摆放」状态：照旧占着文档流里那一格，只是自己挪开一点。 */
function freeCard(card) {
  card.style.position = 'relative';
  card.style.zIndex = '3';
  return sectionOf(card);
}

/** 活动范围：以卡片**没有偏移时的那一格**为原点，四周不能超出分区的内容框。
 *  卡片自己把分区撑到该有的高度，所以这个范围在拖动过程里是稳定的。
 *  （顺带把边框算进去，1px 的偏差在贴边的时候看得出来。） */
function dragBounds(card) {
  const sec = sectionOf(card);
  const sr = sec.getBoundingClientRect();
  const cs = getComputedStyle(sec);
  const bl = parseFloat(cs.borderLeftWidth) || 0;
  const bt = parseFloat(cs.borderTopWidth) || 0;
  const br = parseFloat(cs.borderRightWidth) || 0;
  const bb = parseFloat(cs.borderBottomWidth) || 0;
  const padL = parseFloat(cs.paddingLeft) || 0;
  const padT = parseFloat(cs.paddingTop) || 0;
  const padR = parseFloat(cs.paddingRight) || 0;
  const padB = parseFloat(cs.paddingBottom) || 0;
  const cr = card.getBoundingClientRect();
  const { x, y } = cardOffset(card);
  const flowLeft = cr.left - x;      // 流式位置（视口坐标；拖动期间不会变）
  const flowTop = cr.top - y;
  const minX = sr.left + bl + padL - flowLeft;
  const maxX = sr.right - br - padR - cr.width - flowLeft;
  const minY = sr.top + bt + padT - flowTop;
  const maxY = sr.bottom - bb - padB - cr.height - flowTop;
  return {
    flowLeft, flowTop,
    minX, maxX: Math.max(minX, maxX), minY, maxY: Math.max(minY, maxY),
  };
}

/** 分区里的占位虚框：一张卡片一个，用 data-key 认领。
 *  它**绝对定位、不参与排版**（所以不会把别的卡片挤走），插在分区最前面
 *  ——免得抢走 `.page-section > :last-child` 那条「最后一块不留外边距」的样式。 */
function ghostOf(sec, key) {
  if (!sec || !sec.children) return null;
  return Array.from(sec.children).find(
    (c) => c.classList && c.classList.contains('card-ghost') && c.dataset.key === key,
  ) || null;
}

function slotGhost(card) {
  const sec = sectionOf(card);
  if (!sec.classList || !sec.classList.contains('page-section')) return null;
  let g = ghostOf(sec, card.dataset.cardKey);
  if (!g) {
    g = document.createElement('div');
    g.className = 'card-ghost';
    g.dataset.key = card.dataset.cardKey;
    g.setAttribute('aria-hidden', 'true');
    sec.insertBefore(g, sec.firstChild);
  }
  const sr = sec.getBoundingClientRect();
  const cs = getComputedStyle(sec);
  const cr = card.getBoundingClientRect();
  const b = dragBounds(card);
  g.style.left = `${Math.round(b.flowLeft - sr.left - (parseFloat(cs.borderLeftWidth) || 0))}px`;
  g.style.top = `${Math.round(b.flowTop - sr.top - (parseFloat(cs.borderTopWidth) || 0))}px`;
  g.style.width = `${Math.round(cr.width)}px`;
  g.style.height = `${Math.round(cr.height)}px`;
  return g;
}

/** 占位虚框和实际位置对齐一遍：松手、窗口变化、字体就位、换模型之后各叫一次。 */
function syncGhosts() {
  /* 遍历**全量**卡片，但只给看得见的那些留虚框。被收起的卡（换到自回避行走时那几张）
     必须把虚框**去掉**：分区里会留一个谁也认不出来的空框，而卡片的 inline offset
     还原封不动地存着（显示回来时会自己再要一个）。这里用全量而不是 visibleCards()，
     正是为了把收起那张的框扫掉 —— 只遍历可见的话它会一直挂在那儿。 */
  cardEls().forEach((card) => {
    const sec = sectionOf(card);
    if (!sec.classList || !sec.classList.contains('page-section')) return;
    const g = ghostOf(sec, card.dataset.cardKey);
    if (card.offsetParent === null) { if (g) g.remove(); return; }   // 收起了：不留框
    const { x, y } = cardOffset(card);
    if (!x && !y) { if (g) g.remove(); return; }
    slotGhost(card);
  });
}

/** 把卡片收回自己那一格：抹掉偏移、虚框、抬起状态。 */
function homeCard(card, opts = {}) {
  card.style.left = '';
  card.style.top = '';
  card.style.zIndex = '';
  card.style.position = '';
  const g = ghostOf(sectionOf(card), card.dataset.cardKey);
  if (g) g.remove();
  if (opts.tidy !== false) syncGhosts();
}

/** 窗口变窄变高之后，把已经挪出去的卡片拉回分区里（否则会挂在分区外头）。
 *  只看看得见的卡：dragBounds() 读的是 getBoundingClientRect()，
 *  对 display:none 的卡会拿到 width/height = 0、left/top = 0，
 *  于是算出一个错位的范围、再把错的偏移量写回去 —— 用户下次显示这张卡时
 *  它就跑偏了。收起期间不动它，原样存着。 */
function reclampCards() {
  visibleCards().forEach((card) => {
    const { x, y } = cardOffset(card);
    if (!x && !y) return;
    const b = dragBounds(card);
    const nx = Math.round(Math.min(b.maxX, Math.max(b.minX, x)));
    const ny = Math.round(Math.min(b.maxY, Math.max(b.minY, y)));
    if (nx !== x) card.style.left = `${nx}px`;
    if (ny !== y) card.style.top = `${ny}px`;
  });
}

let tidyTimer = 0;
/** 窗口尺寸变了：先拉回边界，再重画虚框（防抖，省得 resize 期间疯狂算）。 */
function tidyLayout() {
  clearTimeout(tidyTimer);
  tidyTimer = setTimeout(() => { reclampCards(); syncGhosts(); }, 180);
}

/** 只记挪过的卡片：{ 分区: { 卡片: {x, y} } }。 */
function saveCardPos() {
  const out = {};
  cardEls().forEach((c) => {
    const { x, y } = cardOffset(c);
    if (!x && !y) return;
    const sec = sectionOf(c);
    const sk = sec.id || 'body';
    out[sk] = out[sk] || {};
    out[sk][c.dataset.cardKey] = { x: Math.round(x), y: Math.round(y) };
  });
  try { localStorage.setItem(CARD_POS_KEY, JSON.stringify(out)); } catch { /* 隐私模式，忽略 */ }
}

function applyCardPos() {
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem(CARD_POS_KEY) || '{}') || {}; } catch { saved = {}; }
  cardEls().forEach((c) => {
    const sec = sectionOf(c);
    const rec = (saved[sec.id || 'body'] || {})[c.dataset.cardKey];
    if (!rec) return;
    freeCard(c);
    c.style.left = `${Math.round(rec.x) || 0}px`;
    c.style.top = `${Math.round(rec.y) || 0}px`;
  });
}

/* ---------- 八向改大小 ----------
 * 原生 `resize` 只给右下角，而且改不了「往左/往上」。这里八个手柄自己算：
 * 指针位移换算成 width/height 加 margin —— 卡片在布局里是居中的（margin: auto），
 * 所以往左拉时同时把 margin-left 加上同样的位移，左边框才跟着指针走。
 */
const CARD_MIN_W = 260;
const CARD_MIN_H = 140;

function initCardHandles() {
  const DIRS = ['n', 's', 'e', 'w', 'ne', 'nw', 'se', 'sw'];
  cardEls().forEach((card) => {
    DIRS.forEach((dir) => {
      const g = document.createElement('div');
      g.className = `card-grip card-grip-${dir}`;
      g.dataset.dir = dir;
      g.title = '拖动调整大小';
      card.appendChild(g);
    });
  });
}

/** 这张卡此刻最多能拉到多宽：成组容器里的卡片不能超出自己那一格。 */
function cardMaxWidth(card) {
  const parent = card.parentElement;
  if (getComputedStyle(parent).display.includes('grid')) {
    // 同一格里的兄弟还有「没有手动宽度」的就借它的宽度当格宽；
    // 都被拖过就退回整组的宽度（宁可给宽一点，也不要把用户卡死在半格）
    const sib = cardSiblings(card).find((c) => c !== card && !c.style.width);
    return Math.round((sib || parent).getBoundingClientRect().width);
  }
  return Math.round(parent.clientWidth);
}

function initCardResizeDrag() {
  document.addEventListener('pointerdown', (ev) => {
    const grip = ev.target.closest && ev.target.closest('.card-grip');
    if (!grip || ev.button !== 0) return;
    const card = grip.closest('.card, .hero');
    if (!card) return;
    ev.preventDefault();

    const dir = grip.dataset.dir;
    const r = card.getBoundingClientRect();
    const cs = getComputedStyle(card);
    const start = {
      x: ev.clientX, y: ev.clientY, w: r.width, h: r.height,
      ml: parseFloat(cs.marginLeft) || 0, mt: parseFloat(cs.marginTop) || 0,
    };
    const maxW = Math.max(CARD_MIN_W, cardMaxWidth(card));
    card.classList.add('resizing');

    const move = (e) => {
      const dx = e.clientX - start.x;
      const dy = e.clientY - start.y;
      let w = start.w;
      let h = start.h;
      let ml = start.ml;
      let mt = start.mt;
      if (dir.includes('e')) w = Math.min(maxW, start.w + dx);
      if (dir.includes('w')) { w = Math.min(maxW, start.w - dx); ml = start.ml + (start.w - w); }
      if (dir.includes('s')) h = Math.max(CARD_MIN_H, start.h + dy);
      if (dir.includes('n')) { h = Math.max(CARD_MIN_H, start.h - dy); mt = start.mt + (start.h - h); }
      card.style.width = `${Math.round(w)}px`;
      card.style.height = `${Math.round(h)}px`;
      card.style.maxWidth = 'none';
      card.style.marginLeft = `${Math.round(ml)}px`;
      card.style.marginTop = `${Math.round(mt)}px`;
      // 3D 画布是 canvas，让它彻底跟随盒子（去掉「4:3 + 620px」那套限制），
      // 于是这一格的比例就是拖出来的比例；SVG 图仍然按自己的比例等比铺满。
      const cv = card.querySelector('.chain-body > canvas');
      if (cv) {
        cv.style.maxWidth = 'none';
        cv.style.aspectRatio = 'auto';
        cv.style.height = '100%';
        cv.style.width = '100%';
      }
    };
    const up = () => {
      document.removeEventListener('pointermove', move);
      document.removeEventListener('pointerup', up);
      card.classList.remove('resizing');
      // 存盘交给已有的 ResizeObserver —— 它看见行内 width/height 就会记下来
    };
    document.addEventListener('pointermove', move);
    document.addEventListener('pointerup', up);
  });
}

function initCardResize() {
  initCardKeys();
  applyCardSizes();
  applyCardPos();
  initCardDrag();
  initCardHandles();
  initCardResizeDrag();
  // 布局稳定下来（字体、图都就位）再对齐一次虚框和活动边界；窗口变化时同理
  requestAnimationFrame(() => { reclampCards(); syncGhosts(); });
  window.addEventListener('resize', tidyLayout);
  window.addEventListener('load', () => { reclampCards(); syncGhosts(); });
  // 图、3D 画布要等一两秒才把高度撑到位，光靠 rAF 那一次会算早 ——
  // 再补几拍（很便宜，只是把虚框挪回该在的地方）
  [400, 1000, 2000].forEach((ms) => setTimeout(() => { reclampCards(); syncGhosts(); }, ms));
  if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(() => { reclampCards(); syncGhosts(); });
  }
  // 「恢复默认布局」：清掉尺寸和位置，回到出厂排布
  const reset = $('reset-layout');
  if (reset) {
    reset.addEventListener('click', () => {
      try {
        localStorage.removeItem(CARD_SIZE_KEY);
        localStorage.removeItem(CARD_POS_KEY);
      } catch { /* 隐私模式，忽略 */ }
      location.reload();
    });
  }
  if (!window.ResizeObserver) return;      // 老浏览器：能拖，只是不记住
  const pending = new Map();
  const save = () => {
    const out = {};
    cardEls().forEach((el) => {
      // 只记**用户拖过**的：没拖过的保持「跟着内容定宽」，别把自动宽度存成死值
      if (!el.style.width && !el.style.height) return;
      out[el.dataset.cardKey] = {
        w: el.style.width ? Math.round(el.getBoundingClientRect().width) : null,
        h: el.style.height ? Math.round(el.getBoundingClientRect().height) : null,
      };
    });
    try { localStorage.setItem(CARD_SIZE_KEY, JSON.stringify(out)); } catch { /* 隐私模式，忽略 */ }
  };
  const ro = new ResizeObserver((entries) => {
    entries.forEach((e) => {
      const el = e.target;
      // 拖动时浏览器会写行内 width/height —— 有它才说明是用户拉的，不是我们排的
      if (el.style.width || el.style.height) el.style.maxWidth = 'none';
      clearTimeout(pending.get(el));
      pending.set(el, setTimeout(save, 400));
    });
  });
  cardEls().forEach((el) => ro.observe(el));
  // 分区是「随内容长」的：它一变形，虚框和活动边界就重新算一遍
  const roSec = new ResizeObserver(() => tidyLayout());
  document.querySelectorAll('.page-section').forEach((s) => roSec.observe(s));
}

/* ---------------- URL 状态 + 上次参数 ----------------
 *
 * 优先级：URL > localStorage > 默认值。
 * URL 在前是因为「把链接发给别人」必须以链接为准 —— 否则对方打开看到的
 * 是他自己浏览器里存的旧参数，分享就白分享了。 */
// 链模型那几项的默认值要和 fjc_core.ModelParams 保持一致。对不上也不会算错 ——
// 每个值都会送到服务端再校验一遍（校验只有一处），顶多是链接里多带一个参数。
const PARAM_DEFAULTS = {
  n: '100', l: '1', norm: false,
  model: 'fjc', theta: '109.47', cosphi: '0.5', p: '10',
};
const PARAM_LS_KEY = 'fjc-params';

function applyStoredParams() {
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem(PARAM_LS_KEY) || '{}') || {}; } catch { saved = {}; }
  const q = new URLSearchParams(location.search);
  const pick = {
    n: q.has('n') ? q.get('n') : (saved.n !== undefined ? saved.n : PARAM_DEFAULTS.n),
    l: q.has('l') ? q.get('l') : (saved.l !== undefined ? saved.l : PARAM_DEFAULTS.l),
    model: q.has('model') ? q.get('model')
      : (saved.model !== undefined ? saved.model : PARAM_DEFAULTS.model),
    theta: q.has('theta') ? q.get('theta')
      : (saved.theta !== undefined ? saved.theta : PARAM_DEFAULTS.theta),
    cosphi: q.has('cosphi') ? q.get('cosphi')
      : (saved.cosphi !== undefined ? saved.cosphi : PARAM_DEFAULTS.cosphi),
    p: q.has('p') ? q.get('p') : (saved.p !== undefined ? saved.p : PARAM_DEFAULTS.p),
    norm: q.has('norm')
      ? (q.get('norm') === '1' || q.get('norm') === 'true')
      : (saved.norm !== undefined ? !!saved.norm : PARAM_DEFAULTS.norm),
  };
  $('in-n').value = String(pick.n);
  $('in-l').value = String(pick.l);
  $('in-normalize').checked = !!pick.norm;
  // 模型下拉这时还是「一个占位选项」，真正的选项要等 /api/models 回来；
  // 所以这里先填参数输入框，模型键由 loadModels(…return这个pick…) 落地。
  $('in-theta').value = String(pick.theta);
  $('in-cosphi').value = String(pick.cosphi);
  $('in-p').value = String(pick.p);
  return pick;
}

function persistParams() {
  const st = {
    n: $('in-n').value,
    l: $('in-l').value,
    norm: $('in-normalize').checked,
    model: $('in-model').value,
    theta: $('in-theta').value,
    cosphi: $('in-cosphi').value,
    p: $('in-p').value,
  };
  try { localStorage.setItem(PARAM_LS_KEY, JSON.stringify(st)); } catch { /* 隐私模式会抛，忽略 */ }

  // 和默认值一样的参数不写进 URL —— 链接越短越有人愿意点
  const q = new URLSearchParams();
  if (st.n !== PARAM_DEFAULTS.n) q.set('n', st.n);
  if (st.l !== PARAM_DEFAULTS.l) q.set('l', st.l);
  if (st.model !== PARAM_DEFAULTS.model) q.set('model', st.model);
  if (st.theta !== PARAM_DEFAULTS.theta) q.set('theta', st.theta);
  if (st.cosphi !== PARAM_DEFAULTS.cosphi) q.set('cosphi', st.cosphi);
  if (st.p !== PARAM_DEFAULTS.p) q.set('p', st.p);
  if (st.norm) q.set('norm', '1');
  const qs = q.toString();
  try {
    history.replaceState(null, '', qs ? `${location.pathname}?${qs}` : location.pathname);
  } catch { /* file:// 下 replaceState 会抛，忽略 —— 记忆仍然存在 localStorage */ }
}

function resetParams() {
  $('in-n').value = PARAM_DEFAULTS.n;
  $('in-l').value = PARAM_DEFAULTS.l;
  $('in-normalize').checked = PARAM_DEFAULTS.norm;
  $('in-model').value = PARAM_DEFAULTS.model;
  $('in-theta').value = PARAM_DEFAULTS.theta;
  $('in-cosphi').value = PARAM_DEFAULTS.cosphi;
  $('in-p').value = PARAM_DEFAULTS.p;
  syncModelFields();
  // Kuhn 预置和反解的读数都是跟着 l/n 走的，归位时一并清掉
  document.querySelectorAll('#kuhn-chips .chip').forEach((c) => {
    c.classList.remove('on');
    c.setAttribute('aria-pressed', 'false');
  });
  $('kuhn-readout').hidden = true;
  $('solve-readout').hidden = true;
  $('sol-use').disabled = true;
  ['in-n', 'in-l', 'in-theta', 'in-cosphi', 'in-p']
    .forEach((id) => $(id).dispatchEvent(new Event('input', { bubbles: true })));
  clearTimeout(debounceTimer);
  refresh();
}

/* ---------------- Kuhn 长度预置 ----------------
 *
 * 表只有 fjc_core.KUHN_PRESETS 一份，这里从 /api/kuhn 拉 —— JS 里不抄第二份。 */
let KUHN_ROWS = [];

function syncKuhnChips() {
  const cur = Number($('in-l').value);
  let hit = null;
  KUHN_ROWS.forEach((p) => { if (isFinite(cur) && Number(p.b_nm) === cur) hit = p; });
  document.querySelectorAll('#kuhn-chips .chip').forEach((c) => {
    const on = !!hit && Number(c.dataset.b) === Number(hit.b_nm);
    c.classList.toggle('on', on);
    c.setAttribute('aria-pressed', on ? 'true' : 'false');
  });
}

function showKuhn(p) {
  const box = $('kuhn-readout');
  box.textContent = '';
  if (!p) { box.hidden = true; return; }
  box.hidden = false;
  const add = (k, v, cls) => {
    const kk = document.createElement('span'); kk.className = 'k'; kk.textContent = k;
    const vv = document.createElement('span'); vv.className = `v${cls ? ' ' + cls : ''}`; vv.textContent = v;
    box.append(kk, vv);
  };
  add('Kuhn 长度 b', `${fmt(p.b_nm)} nm`, 'big');
  add('持久长度 p', `${fmt(p.p_nm)} nm`, '');
  add('可信度', p.conf_label, 'dim');
  const note = document.createElement('span');
  note.className = 'note';
  note.textContent = `${p.name}：${p.note}　现在 l = ${fmt(Number(p.b_nm))}，所以结果的单位是 nm。`;
  box.appendChild(note);
}

async function loadKuhn() {
  const box = $('kuhn-chips');
  try {
    const res = await fetch('/api/kuhn');
    const data = await res.json().catch(() => null);
    if (!res.ok || !data || !Array.isArray(data.presets)) {
      throw new Error((data && data.error) || `HTTP ${res.status}`);
    }
    KUHN_ROWS = data.presets;
    KUHN_ROWS.forEach((p) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'chip';
      b.dataset.b = String(p.b_nm);
      b.setAttribute('aria-pressed', 'false');
      b.title = `${p.name}：b = ${p.b_nm} nm，p = ${p.p_nm} nm（${p.conf_label}）。${p.note}`;
      const name = document.createElement('span');
      name.textContent = p.name;
      const val = document.createElement('span');
      val.className = 'b';
      val.textContent = fmt(p.b_nm);
      b.append(name, val);
      b.addEventListener('click', () => {
        $('in-l').value = String(p.b_nm);
        $('in-l').dispatchEvent(new Event('input', { bubbles: true }));
        clearTimeout(debounceTimer);
        refresh();
        showKuhn(p);
      });
      box.appendChild(b);
    });
    box.hidden = KUHN_ROWS.length === 0;
    if (data.note) $('kuhn-note').textContent = data.note;
    syncKuhnChips();
  } catch (e) {
    // 拉不到不是致命的：手动填 l 照样能用，所以只改提示、不弹错误条
    $('kuhn-note').textContent = `预置表没能加载（${e.message}）—— 手动填 l 是一样的。`;
  }
}

/* ---------------- h → n 反解 ---------------- */

function renderSolveReadout(d) {
  const box = $('solve-readout');
  box.textContent = '';
  box.hidden = false;

  const add = (k, v, cls) => {
    const kk = document.createElement('span'); kk.className = 'k'; kk.textContent = k;
    const vv = document.createElement('span'); vv.className = `v${cls ? ' ' + cls : ''}`; vv.textContent = v;
    box.append(kk, vv);
  };

  add('精确解 n', fmt(d.n_exact), 'big');
  add('⌊n⌋ → h', `${d.n_floor} → ${fmt(d.h_floor)}`, '');
  add('⌈n⌉ → h', `${d.n_ceil} → ${fmt(d.h_ceil)}`, '');

  const note = document.createElement('span');
  note.className = 'note';
  note.textContent =
    `这是「${d.kind_label}」的反解（l = ${fmt(d.l)}）。` +
    'n_exact 是连续解 —— 链段数本来该是整数，这里没替你取整；' +
    '⌊n⌋ 给出的 h 不超过目标、⌈n⌉ 给出的不低于目标，推荐值取的是 ⌈n⌉。' +
    `（n ∝ (h/l)²：l 翻倍，同一个 h 只要 1/4 的链段数。）`;
  box.appendChild(note);

  const btn = $('sol-use');
  btn.disabled = false;
  btn.textContent = `用 n = ${d.n_ceil}`;
  btn.dataset.n = String(d.n_ceil);
}

async function runSolve() {
  const box = $('solve-readout');
  const raw = $('sol-h').value.trim();
  if (!raw) { renderAlerts('solve-alerts', '先填一个目标末端距 h。', []); return; }

  const h = Number(raw);
  if (!isFinite(h)) { renderAlerts('solve-alerts', `「${raw}」不是数字。`, []); return; }
  const l = Number($('in-l').value);
  if (!isFinite(l)) { renderAlerts('solve-alerts', '链段长度 l 必须是数字。', []); return; }

  let data;
  try {
    const res = await fetch('/api/solve_n', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ h, l, kind: $('sol-kind').value }),
    });
    data = await res.json().catch(() => ({ error: '服务器返回了无法解析的内容' }));
    if (!res.ok) throw new Error(data.error || `请求失败（HTTP ${res.status}）`);
  } catch (e) {
    renderAlerts('solve-alerts', e.message, []);
    box.hidden = true;
    $('sol-use').disabled = true;
    return;
  }

  renderAlerts('solve-alerts', null, data.warnings || []);
  renderSolveReadout(data);
}

/* ---------------- 力–伸长曲线 ---------------- */

// viewBox 860×420，和 index.html 里 #force-chart 的 viewBox 对齐
const FCH = { w: 860, h: 420, top: 36, right: 40, bottom: 56, left: 74 };
const FPW = FCH.w - FCH.left - FCH.right;
const FPH = FCH.h - FCH.top - FCH.bottom;

const forceState = { data: null, idx: null, hoverX: null, ctx: null };

/** λ 目标 → 曲线上 λ 最接近的采样点下标。认不出 / 越界返回 null。 */
function forceIdxFor(lamRaw) {
  const d = forceState.data;
  const s = String(lamRaw).trim();
  if (!d || s === '') return null;
  const lam = Number(s);
  if (!isFinite(lam) || lam < 0) return null;
  let best = Infinity;
  let at = null;
  d.curve.lambda.forEach((v, i) => {
    const diff = Math.abs(v - lam);
    if (diff < best) { best = diff; at = i; }
  });
  return at;
}

function forceReadoutValid() {
  const s = $('f-lam').value.trim();
  if (s === '') return false;
  const lam = Number(s);
  return isFinite(lam) && lam >= 0 && lam < 1;
}

function renderForceReadout() {
  const d = forceState.data;
  const box = $('force-readout');
  box.textContent = '';

  if (!d) { box.hidden = true; return; }

  const s = $('f-lam').value.trim();
  if (s !== '' && !forceReadoutValid()) {
    box.hidden = false;
    const note = document.createElement('span');
    note.className = 'note';
    note.textContent = 'λ 要是 0 到 1 之间的数（λ = 1 是全伸展，永远到不了）。';
    box.appendChild(note);
    forceState.idx = null;
    return;
  }

  const i = forceState.idx;
  if (i === null || i === undefined) { box.hidden = true; return; }

  box.hidden = false;
  const add = (k, v, cls) => {
    const kk = document.createElement('span'); kk.className = 'k'; kk.textContent = k;
    const vv = document.createElement('span'); vv.className = `v${cls ? ' ' + cls : ''}`; vv.textContent = v;
    box.append(kk, vv);
  };
  add('λ', fmt(d.curve.lambda[i]), 'big');
  add('x = f·l/(k_BT)', fmt(d.curve.x[i]), '');
  add('力 f', `${fmt(d.curve.force_pn[i])} pN`, '');
  add('绝对伸长', fmt(d.curve.ext[i]), '');

  const note = document.createElement('span');
  note.className = 'note';
  note.textContent =
    `按曲线上 λ 最接近的采样点读数（共 ${d.curve.x.length} 点，x_max = ${fmt(d.x_max)}）。` +
    `绝对伸长按 n = ${d.n}、l = ${fmt(d.l)} 算；pN 换算假设 l 以 nm、温度以 K 计`
    + `（T = ${fmt(d.temperature)} K）。曲线本身与 n、l 无关。`;
  box.appendChild(note);
}

function drawForceChart() {
  const d = forceState.data;
  const svg = $('force-chart');
  if (!d) return;
  svg.textContent = '';

  const xt = niceTicks(d.x_max, 5);
  const xTop = xt.top;
  const yTop = 1;                       // λ 的渐近线就是 1，固定住才看得出「离全伸展还差多少」
  const sx = (v) => FCH.left + (v / xTop) * FPW;
  const sy = (v) => FCH.top + FPH - (v / yTop) * FPH;

  // --- 网格与刻度 ---
  const grid = el('g', {}, svg);
  [0, 0.25, 0.5, 0.75, 1].forEach((v) => {
    el('line', { class: 'grid-line', x1: FCH.left, x2: FCH.left + FPW, y1: sy(v), y2: sy(v) }, grid);
    el('text', { class: 'tick-text', x: FCH.left - 10, y: sy(v) + 4, 'text-anchor': 'end' }, grid)
      .textContent = fmtTick(v, false);
  });
  xt.ticks.forEach((v) => {
    el('text', { class: 'tick-text', x: sx(v), y: FCH.top + FPH + 20, 'text-anchor': 'middle' }, grid)
      .textContent = fmtTick(v, false);
  });

  el('line', {
    class: 'axis-line', x1: FCH.left, x2: FCH.left + FPW, y1: FCH.top + FPH, y2: FCH.top + FPH,
  }, svg);
  el('line', {
    class: 'axis-line', x1: FCH.left, x2: FCH.left, y1: FCH.top, y2: FCH.top + FPH,
  }, svg);
  el('text', {
    class: 'axis-title', x: FCH.left + FPW / 2, y: FCH.h - 14, 'text-anchor': 'middle',
  }, svg).textContent = '无量纲力 x = f·l / (k_B T)';
  el('text', {
    class: 'axis-title', x: 0, y: 0, 'text-anchor': 'middle',
    transform: `translate(18, ${FCH.top + FPH / 2}) rotate(-90)`,
  }, svg).textContent = '归一化伸长 λ';

  // --- 高斯链 λ = x/3：参考线，所以走虚线灰（不占数据色）---
  // 只截到 λ = 1（x = 3）为止 —— 再往右它会预测出超过全伸展的伸长，那不是「近似得不好」。
  if (d.curve_linear && d.curve_linear.x.length) {
    let dl = '';
    d.curve_linear.x.forEach((xv, i) => {
      dl += (i === 0 ? 'M' : 'L') + sx(xv).toFixed(2) + ',' + sy(d.curve_linear.lambda[i]).toFixed(2);
    });
    el('path', { class: 'fit-line', d: dl }, svg);
  }

  // --- Langevin：数据，所以穿 series-1 ---
  let path = '';
  d.curve.x.forEach((xv, i) => {
    path += (i === 0 ? 'M' : 'L') + sx(xv).toFixed(2) + ',' + sy(d.curve.lambda[i]).toFixed(2);
  });
  el('path', { class: 'series-line', d: path, stroke: seriesVar(0) }, svg);

  // --- 「高斯近似差 2%」的位置：后端算好给的，标出来就是这张图的结论 ---
  if (isFinite(d.x_linear_end) && d.x_linear_end > 0 && d.x_linear_end < xTop) {
    const x = sx(d.x_linear_end);
    el('line', { class: 'ref-line', x1: x, x2: x, y1: FCH.top, y2: FCH.top + FPH }, svg);
    el('text', {
      class: 'ref-text', x: x + 4, y: FCH.top + FPH - 6, 'text-anchor': 'start',
    }, svg).textContent = `差 2%：x = ${fmt(d.x_linear_end)}`;
  }

  // --- λ 目标标线 ---
  const i = forceState.idx;
  if (i !== null && i !== undefined && i >= 0) {
    const mx = sx(d.curve.x[i]);
    const my = sy(d.curve.lambda[i]);
    el('line', { class: 'mark-line', x1: mx, x2: mx, y1: FCH.top, y2: FCH.top + FPH }, svg);
    el('circle', { class: 'series-dot', cx: mx, cy: my, r: 4.5, fill: seriesVar(0) }, svg);
    el('text', {
      class: 'ref-text', x: mx > FCH.left + FPW - 90 ? mx - 6 : mx + 6,
      y: FCH.top + 12, 'text-anchor': mx > FCH.left + FPW - 90 ? 'end' : 'start',
    }, svg).textContent = `λ = ${fmt(d.curve.lambda[i])}`;
  }

  // --- 悬停层 ---
  const hoverG = el('g', { id: 'force-hover-layer' }, svg);
  el('rect', {
    class: 'hover-surface', x: FCH.left, y: FCH.top, width: FPW, height: FPH,
    id: 'force-hover-surface',
  }, hoverG);

  forceState.ctx = { sx, sy, xTop, yTop };

  if (forceState.hoverX !== null) updateForceHover(forceState.hoverX);
}

/** 把数据横坐标换算到最近的采样点，刷新十字线和 tooltip。 */
function updateForceHover(dataX) {
  const d = forceState.data;
  const layer = $('force-hover-layer');
  const tt = $('force-tooltip');
  if (!d || !layer || !forceState.ctx) return;
  [...layer.querySelectorAll('.crosshair, .hover-dot')].forEach((n) => n.remove());

  if (dataX === null || dataX === undefined) { tt.hidden = true; return; }

  let best = 0;
  let bestD = Infinity;
  d.curve.x.forEach((xv, i) => {
    const diff = Math.abs(xv - dataX);
    if (diff < bestD) { bestD = diff; best = i; }
  });

  const { sx, sy } = forceState.ctx;
  const x = sx(d.curve.x[best]);
  el('line', { class: 'crosshair', x1: x, x2: x, y1: FCH.top, y2: FCH.top + FPH }, layer);
  el('circle', {
    class: 'series-dot hover-dot', cx: x, cy: sy(d.curve.lambda[best]), r: 4.5,
    fill: seriesVar(0),
  }, layer);

  const row = (k, v) => {
    const div = document.createElement('div');
    div.className = 'tt-row';
    const kk = document.createElement('span'); kk.className = 'tt-name'; kk.textContent = k;
    const vv = document.createElement('span'); vv.className = 'tt-val'; vv.textContent = v;
    div.append(kk, vv);
    return div;
  };

  tt.textContent = '';
  const head = document.createElement('div');
  head.className = 'tt-head';
  head.textContent = `x = ${fmt(d.curve.x[best])}`;
  tt.appendChild(head);
  tt.appendChild(row('λ', fmt(d.curve.lambda[best])));
  tt.appendChild(row('f', `${fmt(d.curve.force_pn[best])} pN`));
  tt.appendChild(row('⟨x⟩', fmt(d.curve.ext[best])));

  const svg = $('force-chart');
  const rect = svg.getBoundingClientRect();
  const scale = rect.width / FCH.w;
  const px = x * scale;
  tt.hidden = false;
  const ttW = tt.offsetWidth;
  const left = px + 14 + ttW > rect.width ? px - 14 - ttW : px + 14;
  tt.style.left = `${Math.max(0, left)}px`;
  tt.style.top = `${Math.min(Math.max(0, sy(d.curve.lambda[best]) * scale - 10),
    rect.height - tt.offsetHeight)}px`;
}

async function refreshForce(n, l) {
  // 力–伸长是 FJC 的解析解，自回避行走没有同等地位的关系式（卡片也收起来了）。
  const spec = modelSpec();
  if (spec && !spec.analytic) return;

  const body = $('force-body');
  body.classList.add('busy');
  $('force-png').disabled = true;

  let data;
  try {
    const res = await fetch('/api/force', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ n, l }),
    });
    data = await res.json().catch(() => ({ error: '服务器返回了无法解析的内容' }));
    if (!res.ok) throw new Error(data.error || `请求失败（HTTP ${res.status}）`);
  } catch (e) {
    renderAlerts('force-alerts', e.message, []);
    forceState.data = null;
    body.classList.remove('busy');
    return;
  }

  forceState.data = data;
  forceState.hoverX = null;
  forceState.idx = forceReadoutValid() ? forceIdxFor($('f-lam').value) : null;
  renderForceReadout();
  drawForceChart();
  renderAlerts('force-alerts', null, data.warnings || []);
  $('force-png').disabled = false;
  body.classList.remove('busy');
}

function refreshForceReadoutOnly() {
  forceState.idx = forceReadoutValid() ? forceIdxFor($('f-lam').value) : null;
  renderForceReadout();
  if (forceState.data) drawForceChart();
}

/* ---------------- 主流程 ---------------- */

let debounceTimer = null;

async function refresh() {
  syncNChips();      // 放在最前面：输入不合法提前返回时，chip 的选中态也已经跟上了
  syncKuhnChips();   // Kuhn 选中态同理（l 被手改掉就该灭掉）

  /* 自回避行走：这一页「按公式算」的东西一件都不适用（没有闭式解），
     所以**不去打 /api/compute** —— 打了服务端会返 400，顶部告警条上就挂着一条
     假报错，而下面那些卡片本来也都要收起来。收起由 CSS 按 body 的 data-model-kind
     做，这里只负责把上一次解析模型留下的告警清掉、别留个过期的数字在屏幕上。 */
  const spec = modelSpec();
  if (spec && !spec.analytic) {
    setAlerts(null, []);
    $('chart-body').classList.remove('busy');
    return;
  }

  let ns;
  let l;
  try {
    ns = parseNs($('in-n').value);
    l = Number($('in-l').value);
    if (!isFinite(l)) throw new Error('链段长度 l 必须是数字');
  } catch (e) {
    setAlerts(e.message, []);
    return;
  }

  state.normalize = $('in-normalize').checked;
  $('chart-body').classList.add('busy');

  let results;
  try {
    results = await compute(ns, l);
  } catch (e) {
    setAlerts(e.message, []);
    $('chart-body').classList.remove('busy');
    return;
  }

  const warnings = [];
  results.forEach((r) => {
    r.warnings.forEach((w) => {
      const tagged = results.length > 1 ? `n = ${r.n}：${w}` : w;
      if (!warnings.includes(tagged)) warnings.push(tagged);
    });
  });
  setAlerts(null, warnings);

  state.hoverIdx = null;
  const multi = results.length > 1;
  renderHero(results[0], multi);
  renderTiles(results[0], multi);
  renderModelReadout(results[0]);
  renderLegend(results);
  renderTable(results);
  drawChart(results, state.normalize);
  $('chart-body').classList.remove('busy');
  $('chart-png').disabled = false;

  // 只在算成功以后才记忆：把一个输错的中间态写进 URL，刷新回来就卡在错值上
  persistParams();

  // 力–伸长那张图自己有 busy 与报错条，所以不 await —— 主图先出，它随后补上
  refreshForce(ns[0], l);

  // l 变了反解的结论就变了（n ∝ (h/l)²）。读数条还开着就顺手重算一遍，
  // 免得留在屏幕上的是个过期的 n。runSolve 不会再调 refresh，不会循环。
  if (!$('solve-readout').hidden && $('sol-h').value.trim()) runSolve();
}

function scheduleRefresh() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(refresh, 180);
}

function initEvents() {
  renderNChips();
  $('in-n').addEventListener('input', scheduleRefresh);
  $('in-l').addEventListener('input', scheduleRefresh);
  $('in-normalize').addEventListener('change', refresh);
  // 链模型：换模型要连字段显隐、公式说明一起换；改参数只重算
  $('in-model').addEventListener('change', () => {
    syncModelFields();
    persistParams();
    refresh();
  });
  ['in-theta', 'in-cosphi', 'in-p'].forEach((id) => {
    $(id).addEventListener('input', () => { syncModelFields(); scheduleRefresh(); });
  });
  $('model-default').addEventListener('click', () => {
    const spec = MODEL_BY_KEY.get($('in-model').value);
    if (!spec) return;
    $('in-theta').value = String(spec.params.theta_deg.default);
    $('in-cosphi').value = String(spec.params.cos_phi.default);
    $('in-p').value = String(spec.params.p.default);
    syncModelFields();
    persistParams();
    refresh();
  });

  // --- 新增的三件：Kuhn 预置、h → n 反解、力–伸长 ---

  $('reset-btn').addEventListener('click', resetParams);

  $('kuhn-clear').addEventListener('click', () => {
    $('in-l').value = PARAM_DEFAULTS.l;
    $('in-l').dispatchEvent(new Event('input', { bubbles: true }));
    clearTimeout(debounceTimer);
    refresh();
    showKuhn(null);
  });

  $('sol-run').addEventListener('click', runSolve);
  // 目标 h 的框里按回车就该反解 —— 这是个单值输入，等鼠标去点按钮是多余的一步
  $('sol-h').addEventListener('keydown', (ev) => {
    if (ev.key === 'Enter') { ev.preventDefault(); runSolve(); }
  });
  $('sol-kind').addEventListener('change', () => {
    if ($('sol-h').value.trim()) runSolve();
  });
  $('sol-use').addEventListener('click', () => {
    const n = $('sol-use').dataset.n;
    if (!n) return;
    $('in-n').value = n;
    $('in-n').dispatchEvent(new Event('input', { bubbles: true }));
    clearTimeout(debounceTimer);
    refresh();
  });

  // λ 输入只改标线和读数，不打后端 —— 曲线本来就已经在手上了
  $('f-lam').addEventListener('input', refreshForceReadoutOnly);

  $('chart-png').addEventListener('click', () => {
    if (!$('chart').textContent) return;
    const tag = state.normalize ? 'norm' : 'ph';
    window.fjcExport.svgToPng($('chart'), `fjc-${tag}-n${nStem()}.png`)
      .catch((e) => window.fjcExport.pngFail('alerts', e));
  });
  $('force-png').addEventListener('click', () => {
    if (!forceState.data) return;
    window.fjcExport.svgToPng($('force-chart'),
      `fjc-force-n${forceState.data.n}-T${forceState.data.temperature}.png`)
      .catch((e) => window.fjcExport.pngFail('force-alerts', e));
  });

  const svg = $('chart');

  // 鼠标：按最近的采样点读数
  svg.addEventListener('mousemove', (ev) => {
    if (!state.ctx) return;
    const rect = svg.getBoundingClientRect();
    const scale = CH.w / rect.width;
    const vx = (ev.clientX - rect.left) * scale;
    if (vx < CH.left || vx > CH.left + PW) { state.hoverIdx = null; updateHover(null); return; }
    const dataX = ((vx - CH.left) / PW) * state.ctx.xTop;
    state.hoverIdx = dataX;
    updateHover(dataX);
  });
  svg.addEventListener('mouseleave', () => { state.hoverIdx = null; updateHover(null); });

  // 键盘：聚焦后用 ←/→ 移动，读到的和悬停一样
  svg.addEventListener('keydown', (ev) => {
    if (!state.ctx) return;
    if (ev.key !== 'ArrowLeft' && ev.key !== 'ArrowRight') return;
    ev.preventDefault();
    const step = state.ctx.xTop / 60;
    let v = state.hoverIdx === null ? state.ctx.xTop / 2 : state.hoverIdx;
    v = Math.min(state.ctx.xTop, Math.max(0, v + (ev.key === 'ArrowRight' ? step : -step)));
    state.hoverIdx = v;
    updateHover(v);
  });
  svg.addEventListener('blur', () => { state.hoverIdx = null; updateHover(null); });

  // 力–伸长那张图：同一套悬停 + 键盘读数，只是坐标系是自己的
  const fsvg = $('force-chart');
  const forceXFromEvent = (ev) => {
    if (!forceState.ctx) return null;
    const rect = fsvg.getBoundingClientRect();
    const scale = FCH.w / rect.width;
    const vx = (ev.clientX - rect.left) * scale;
    if (vx < FCH.left || vx > FCH.left + FPW) return null;
    return ((vx - FCH.left) / FPW) * forceState.ctx.xTop;
  };
  fsvg.addEventListener('mousemove', (ev) => {
    forceState.hoverX = forceXFromEvent(ev);
    updateForceHover(forceState.hoverX);
  });
  fsvg.addEventListener('mouseleave', () => { forceState.hoverX = null; updateForceHover(null); });
  fsvg.addEventListener('keydown', (ev) => {
    if (!forceState.ctx) return;
    if (ev.key !== 'ArrowLeft' && ev.key !== 'ArrowRight') return;
    ev.preventDefault();
    const step = forceState.ctx.xTop / 60;
    let v = forceState.hoverX === null ? forceState.ctx.xTop / 2 : forceState.hoverX;
    v = Math.min(forceState.ctx.xTop, Math.max(0, v + (ev.key === 'ArrowRight' ? step : -step)));
    forceState.hoverX = v;
    updateForceHover(v);
  });
  fsvg.addEventListener('blur', () => { forceState.hoverX = null; updateForceHover(null); });

  // 复制数值表（制表符分隔，可直接粘进 Excel）
  $('copy-table').addEventListener('click', () => {
    if (!state.results) return;
    const lines = [COLS.map((c) => c.head).join('\t')];
    state.results.forEach((r) => lines.push(COLS.map((c) => c.get(r)).join('\t')));
    const text = lines.join('\n');
    const btn = $('copy-table');

    // navigator.clipboard 在某些上下文（iframe、缺权限）会直接抛错，
    // 用隐藏 textarea + execCommand 兜底，再不行就明确告诉用户去手动选。
    const legacyCopy = () => {
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.setAttribute('readonly', '');
      ta.style.position = 'fixed';
      ta.style.top = '-1000px';
      document.body.appendChild(ta);
      ta.select();
      let ok = false;
      try { ok = document.execCommand('copy'); } catch { ok = false; }
      ta.remove();
      return ok;
    };

    const done = (label) => {
      btn.textContent = label;
      setTimeout(() => { btn.textContent = '复制'; }, 1600);
    };

    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text)
        .then(() => done('已复制'))
        .catch(() => done(legacyCopy() ? '已复制' : '请手动选择表格'));
    } else {
      done(legacyCopy() ? '已复制' : '请手动选择表格');
    }
  });

    // 深浅色：默认深色（默认值写在 <html data-theme="dark"> 上，见 index.html），
    // 用户手动选过的记在 localStorage 里。这里只负责恢复那个选择 + 切换按钮。
  const btn = $('theme-toggle');
  const root = document.documentElement;
  const saved = localStorage.getItem('fjc-theme');
  if (saved) root.dataset.theme = saved;

  const isDark = () => {
    const t = root.dataset.theme;
    if (t) return t === 'dark';
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  };
  const syncBtn = () => { btn.textContent = isDark() ? '浅色' : '深色'; };
  syncBtn();

  btn.addEventListener('click', () => {
    const next = isDark() ? 'light' : 'dark';
    root.dataset.theme = next;
    localStorage.setItem('fjc-theme', next);
    syncBtn();
    if (state.results) drawChart(state.results, state.normalize);
    if (forceState.data) drawForceChart();
  });

  window.addEventListener('resize', () => {
    if (state.ctx !== null) updateHover(state.hoverIdx);
    if (forceState.ctx !== null) updateForceHover(forceState.hoverX);
  });

  // 这三件收尾：栏高/高亮要量 DOM，水波纹走委托，Kuhn 表是异步拉的
  initScrollSpy();
  initRipple();
  loadKuhn();
}

const storedParams = applyStoredParams();   // 必须在 initEvents / refresh 之前 —— chip 与首次请求都读它
initEvents();
// 模块尺寸：先把上次存的大小摆回去，再挂监听（顺序反了会把「恢复」当成一次拖动）
initCardResize();
// 模型目录要先到（第一次请求才知道该带哪些模型参数）。拉不到就退回默认的 FJC：
// 页面照常能用，只是「链模型」那张卡只有一个选项。
loadModels(storedParams.model).catch(() => {}).then(refresh);

/* ---------------- 给 AI 助手的接口 ----------------
 *
 * 助手（ai.js）不直接摸 DOM，只调这里的四个函数。定这一层的唯一理由是那条
 * 反复踩过的坑：**setter 必须走界面自己的那条路**。
 * 直接写 `$('in-n').value = 400` 是没用的 —— 图和表都不会重画，因为没有任何
 * 事件被派发；而 `#in-l` 更麻烦，chain3d.js 和 sweep.js 各自挂着监听器，
 * 只改 value 会让 3D 视图和扫描提示停在旧值上，看起来像「助手改了个假参数」。
 *
 * 所以规则是：改哪个输入框，就派发它自己的 input/change 事件，让每一个监听器
 * 都按平时那样醒一次；然后只补一次 await refresh()（并把防抖里排队的那次撤掉，
 * 免得对 /api/compute 打两个请求）。
 *
 * 暴露的是一组动作，不是一个状态对象 —— 写回去以后读 getState() 才是真值。
 */
window.fjcApp = {
  // 当前链模型与它的参数（计算 / 3D / 扫描 / 助手共用这一份读法）
  modelParams,
  // 当前模型在目录里的那一项 —— 3D 卡靠它知道这个模型有没有闭式解
  modelSpec,

  getState() {
    return {
      n: $('in-n').value,
      l: $('in-l').value,
      normalize: $('in-normalize').checked,
      model: $('in-model').value,
      ...modelParams(),
      curves: state.results ? state.results.map((r) => r.n) : [],
    };
  },

  /** 改顶部参数行（含链模型与它的参数）。n / l 至少给一个。 */
  async setParams(args) {
    const { n, l, model, theta_deg, cos_phi, p } = args || {};
    const touched = [];
    if (n !== undefined) { $('in-n').value = String(n); touched.push('in-n'); }
    if (l !== undefined) { $('in-l').value = String(l); touched.push('in-l'); }
    if (model !== undefined) {
      const want = MODEL_BY_KEY.get(String(model));
      if (!want) {
        return { ok: false, error: `没有这个链模型：${model}` };
      }
      // 自回避行走在目录里，但它**没有闭式解** —— 切过去会把整页「按公式算」的
      // 卡片全收起来。助手（或任何脚本）静默把用户面前的页面换掉，比报个错糟得多，
      // 所以这里只放四个解析模型过去，SAW 由用户自己在下拉里选。
      if (!want.analytic) {
        return {
          ok: false,
          error: `${want.label}没有闭式解，只做实空间采样，页面上多数卡片不适用；`
            + '要切到它请在模型下拉里手动选。',
        };
      }
      $('in-model').value = String(model);
      syncModelFields();
      touched.push('in-model');
    }
    if (theta_deg !== undefined) { $('in-theta').value = String(theta_deg); touched.push('in-theta'); }
    if (cos_phi !== undefined) { $('in-cosphi').value = String(cos_phi); touched.push('in-cosphi'); }
    if (p !== undefined) { $('in-p').value = String(p); touched.push('in-p'); }
    if (!touched.length) return { ok: false, error: '没给 n 也没给 l' };

    touched.forEach((id) => $(id).dispatchEvent(new Event('input', { bubbles: true })));
    clearTimeout(debounceTimer);      // 撤掉刚被排队的那次防抖刷新
    await refresh();
    return { ok: true, data: this.getState() };
  },

  /** 曲线集合：写进 n 栏（界面本来就是逗号分隔的多值），走同一条路。 */
  async setCurves(args) {
    const ns = (args || {}).ns;
    if (!Array.isArray(ns) || ns.length === 0) {
      return { ok: false, error: 'ns 必须是非空数组' };
    }
    $('in-n').value = ns.join(', ');
    $('in-n').dispatchEvent(new Event('input', { bubbles: true }));
    clearTimeout(debounceTimer);
    await refresh();
    return { ok: true, data: this.getState() };
  },

  /**
   * 归一化开关。这个不用派发 change：`#in-normalize` 的监听器就是 refresh
   * 本身，而 refresh 每次都从复选框现读 state.normalize —— 派发只会让它多跑一遍。
   */
  async setNormalize(args) {
    const box = $('in-normalize');
    box.checked = !!(args || {}).on;
    await refresh();
    return { ok: true, data: this.getState() };
  },
};
