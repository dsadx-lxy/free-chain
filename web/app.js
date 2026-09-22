'use strict';

/* 自由连接链计算器 —— 前端
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
    body: JSON.stringify({ n: ns.length === 1 ? ns[0] : ns, l }),
  });
  const data = await res.json().catch(() => ({ error: '服务器返回了无法解析的内容' }));
  if (!res.ok) throw new Error(data.error || `请求失败（HTTP ${res.status}）`);
  return data.results;
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
    : '根均方末端距 h';
  $('hero-value').textContent = fmt(r.h_rms);
  $('hero-formula').textContent =
    `h = l·√n = ${fmt(r.l)} × √${r.n} = ${fmt(r.h_rms)}` +
    (multi ? '　（对比模式下主结果取第一个 n，其余见下表）' : '');
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
  el('line', {
    class: 'axis-line', x1: CH.left, x2: CH.left + PW, y1: CH.top + PH, y2: CH.top + PH,
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
  const bgcs = holder ? getComputedStyle(holder).backgroundColor : '';
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
      let cur = secs[0];
      secs.forEach((s) => { if (s.getBoundingClientRect().top <= offset) cur = s; });
      // 滚到底时最后一节可能永远没越过阈值，直接钉到最后一个
      if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 6) {
        cur = secs[secs.length - 1];
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

/* ---------------- URL 状态 + 上次参数 ----------------
 *
 * 优先级：URL > localStorage > 默认值。
 * URL 在前是因为「把链接发给别人」必须以链接为准 —— 否则对方打开看到的
 * 是他自己浏览器里存的旧参数，分享就白分享了。 */
const PARAM_DEFAULTS = { n: '100', l: '1', norm: false };
const PARAM_LS_KEY = 'fjc-params';

function applyStoredParams() {
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem(PARAM_LS_KEY) || '{}') || {}; } catch { saved = {}; }
  const q = new URLSearchParams(location.search);
  const pick = {
    n: q.has('n') ? q.get('n') : (saved.n !== undefined ? saved.n : PARAM_DEFAULTS.n),
    l: q.has('l') ? q.get('l') : (saved.l !== undefined ? saved.l : PARAM_DEFAULTS.l),
    norm: q.has('norm')
      ? (q.get('norm') === '1' || q.get('norm') === 'true')
      : (saved.norm !== undefined ? !!saved.norm : PARAM_DEFAULTS.norm),
  };
  $('in-n').value = String(pick.n);
  $('in-l').value = String(pick.l);
  $('in-normalize').checked = !!pick.norm;
}

function persistParams() {
  const st = {
    n: $('in-n').value,
    l: $('in-l').value,
    norm: $('in-normalize').checked,
  };
  try { localStorage.setItem(PARAM_LS_KEY, JSON.stringify(st)); } catch { /* 隐私模式会抛，忽略 */ }

  // 和默认值一样的参数不写进 URL —— 链接越短越有人愿意点
  const q = new URLSearchParams();
  if (st.n !== PARAM_DEFAULTS.n) q.set('n', st.n);
  if (st.l !== PARAM_DEFAULTS.l) q.set('l', st.l);
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
  // Kuhn 预置和反解的读数都是跟着 l/n 走的，归位时一并清掉
  document.querySelectorAll('#kuhn-chips .chip').forEach((c) => {
    c.classList.remove('on');
    c.setAttribute('aria-pressed', 'false');
  });
  $('kuhn-readout').hidden = true;
  $('solve-readout').hidden = true;
  $('sol-use').disabled = true;
  ['in-n', 'in-l'].forEach((id) => $(id).dispatchEvent(new Event('input', { bubbles: true })));
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

  // 深浅色：跟着系统，但可以手动覆盖
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

applyStoredParams();   // 必须在 initEvents / refresh 之前 —— chip 与首次请求都读它
initEvents();
refresh();

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
  getState() {
    return {
      n: $('in-n').value,
      l: $('in-l').value,
      normalize: $('in-normalize').checked,
      curves: state.results ? state.results.map((r) => r.n) : [],
    };
  },

  /** 改顶部参数行。n / l 至少给一个。 */
  async setParams(args) {
    const { n, l } = args || {};
    const touched = [];
    if (n !== undefined) { $('in-n').value = String(n); touched.push('in-n'); }
    if (l !== undefined) { $('in-l').value = String(l); touched.push('in-l'); }
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
