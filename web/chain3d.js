'use strict';

/* 自由连接链计算器 —— 3D 单链构象视图
 *
 * 手写 canvas 2D 投影，不引入任何 3D 库：一条链就是一条折线，用不上 WebGL，
 * 自己算投影反而能把线宽、纵深、坐标框全捏在手里，也保住了 README 承诺的
 * 「无框架、无 CDN、离线可用」。
 *
 * 这里**不含任何物理公式**。顶点坐标、R、⟨R²⟩ 全部由后端 fjc_core.py 算好返回，
 * 本文件只做坐标变换、画图和导出。
 *
 * 复用 app.js 里的 fmt() / fmtTick()（同为传统 script，chain3d.js 在其后加载）。
 */

(function () {
  const BANDS = 8;                        // 纵深分带数（画家算法的粒度）
  const MARGIN = 34;                      // 画布内边距（CSS 像素）
  const DRAG_MAX_SEGMENTS = 20000;        // 拖拽时最多画这么多段，超了就抽稀
  const ANIM_MS = 2500;                   // 生长动画总时长，与 n 无关
  const DEF = { azim: -Math.PI / 3, elev: (20 * Math.PI) / 180, zoom: 1 };
  const FONT = '12px system-ui, "Segoe UI", "Microsoft YaHei", sans-serif';

  const $ = (id) => document.getElementById(id);

  const view = { azim: DEF.azim, elev: DEF.elev, zoom: DEF.zoom };
  const state = {
    data: null,
    grid: true,
    frac: 1,          // 生长动画进度：1 = 完整链
    animId: null,
    drag: null,
    seq: 0,           // 请求序号，用来丢弃过期响应
    colors: null,
  };

  // fetchChain 的防抖句柄。提到模块作用域，是为了让下面的 window.fjc3d
  // 能在自己触发取数时撤掉排队中的那一次 —— 否则助手改一次参数会打两个 /api/chain。
  let fetchTimer = null;

  /* ---------------- 取色 ----------------
   * canvas 不参与 CSS 变量解析 —— var(--series-1) 在这里是无效值。
   * 必须每次重绘时用 getComputedStyle 现取，否则切换深浅色后画布颜色不会变。
   */
  function readColors() {
    const cs = getComputedStyle(document.querySelector('.viz-root'));
    const g = (n) => cs.getPropertyValue(n).trim();
    state.colors = {
      surface: g('--surface-1'),
      grid: g('--grid'),
      axis: g('--axis'),
      ink: g('--text-primary'),
      secondary: g('--text-secondary'),
      muted: g('--text-muted'),
      series: [1, 2, 3, 4, 5].map((i) => g(`--series-${i}`)),
    };
  }

  /* ---------------- 相机与投影 ---------------- */

  /** 由方位角/仰角构造一组正交基。z 轴朝上（沿用 matplotlib 的约定）。 */
  function basis() {
    const ca = Math.cos(view.azim);
    const sa = Math.sin(view.azim);
    const ce = Math.cos(view.elev);
    const se = Math.sin(view.elev);
    return {
      right: [-sa, ca, 0],                            // 屏幕向右
      up: [-se * ca, -se * sa, ce],                   // 屏幕向上
      forward: [ce * ca, ce * sa, se],                // 指向相机，用于判深浅
    };
  }

  /**
   * 比例尺用**包围球**定，不用投影包围盒。
   * 包围球是旋转不变量，所以转视角时比例不会每帧变 —— 用投影包围盒画面会「呼吸」。
   */
  function makeCam(box, w, h) {
    const half = [0, 1, 2].map((k) => (box.max[k] - box.min[k]) / 2);
    const center = [0, 1, 2].map((k) => (box.min[k] + box.max[k]) / 2);
    const radius = Math.hypot(half[0], half[1], half[2]) || 1;
    return {
      b: basis(),
      center,
      half,
      scale: (Math.min(w, h) / 2 - MARGIN) / radius,
      cx: w / 2,
      cy: h / 2,
    };
  }

  function proj(cam, x, y, z) {
    const b = cam.b;
    const dx = x - cam.center[0];
    const dy = y - cam.center[1];
    const dz = z - cam.center[2];
    const sx = dx * b.right[0] + dy * b.right[1] + dz * b.right[2];
    const sy = dx * b.up[0] + dy * b.up[1] + dz * b.up[2];
    const d = dx * b.forward[0] + dy * b.forward[1] + dz * b.forward[2];
    const s = cam.scale * view.zoom;
    return { x: cam.cx + sx * s, y: cam.cy - sy * s, d };
  }

  /* ---------------- 坐标框 ---------------- */

  function boundsOf(chains) {
    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];
    for (const flat of chains) {
      for (let i = 0; i < flat.length; i += 3) {
        for (let k = 0; k < 3; k++) {
          const v = flat[i + k];
          if (v < min[k]) min[k] = v;
          if (v > max[k]) max[k] = v;
        }
      }
    }
    // 退化轴（比如 n=1 且某轴恰好没跨度）不能是零长度，否则刻度步长会除零
    let span = 0;
    for (let k = 0; k < 3; k++) span = Math.max(span, max[k] - min[k]);
    const floor = Math.max(span * 0.12, 1e-6);
    for (let k = 0; k < 3; k++) {
      if (max[k] - min[k] < floor) {
        const mid = (min[k] + max[k]) / 2;
        min[k] = mid - floor / 2;
        max[k] = mid + floor / 2;
      }
    }
    return { min, max };
  }

  /** 落在 [lo, hi] 内的「好看」刻度：步长取 1/2/5×10^k。 */
  function axisTicks(lo, hi, count = 4) {
    const span = hi - lo;
    if (!(span > 0)) return [lo];
    const raw = span / count;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const norm = raw / mag;
    const step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10) * mag;
    const out = [];
    for (let i = Math.ceil(lo / step - 1e-9); i * step <= hi + step * 1e-9; i++) {
      out.push(Math.abs(i) < 1e-12 ? 0 : i * step);
    }
    return out;
  }

  /** 盒子上离相机最近的那个底角 —— 三条主轴从这里出发（matplotlib 的做法）。 */
  function floorCorner(cam, box) {
    let best = null;
    for (const x of [box.min[0], box.max[0]]) {
      for (const y of [box.min[1], box.max[1]]) {
        const p = proj(cam, x, y, box.min[2]);
        if (!best || p.d > best.d) best = { x, y, z: box.min[2], d: p.d };
      }
    }
    return best;
  }

  const other = (v, lo, hi) => (v === lo ? hi : lo);

  function drawFrame(ctx, cam, box, ticks) {
    const c = state.colors;
    const [x0, y0, z0] = box.min;
    const [x1, y1, z1] = box.max;

    // --- 三个背面上的网格（floor + 两个远墙）---
    if (state.grid) {
      ctx.strokeStyle = c.grid;
      ctx.lineWidth = 1;
      ctx.beginPath();
      const seg = (a, b) => {
        const p = proj(cam, a[0], a[1], a[2]);
        const q = proj(cam, b[0], b[1], b[2]);
        ctx.moveTo(p.x, p.y);
        ctx.lineTo(q.x, q.y);
      };
      const fc = floorCorner(cam, box);
      const farX = other(fc.x, x0, x1);
      const farY = other(fc.y, y0, y1);
      ticks[0].forEach((v) => {
        seg([v, y0, z0], [v, y1, z0]);           // 地板
        seg([farX, v, z0], [farX, v, z1]);       // 远 x 墙
      });
      ticks[1].forEach((v) => {
        seg([x0, v, z0], [x1, v, z0]);
        seg([v, farY, z0], [v, farY, z1]);       // 远 y 墙
      });
      ticks[2].forEach((v) => {
        seg([farX, y0, v], [farX, y1, v]);
        seg([x0, farY, v], [x1, farY, v]);
      });
      ctx.stroke();
    }

    // --- 12 条棱的线框 ---
    ctx.strokeStyle = c.grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    const corners = [];
    for (const x of [x0, x1]) for (const y of [y0, y1]) for (const z of [z0, z1]) {
      corners.push([x, y, z]);
    }
    corners.forEach((a, i) => {
      corners.forEach((b, j) => {
        if (j <= i) return;
        // 只连相差一个坐标的顶点，正好 12 条棱
        const diff = (a[0] !== b[0]) + (a[1] !== b[1]) + (a[2] !== b[2]);
        if (diff !== 1) return;
        const p = proj(cam, a[0], a[1], a[2]);
        const q = proj(cam, b[0], b[1], b[2]);
        ctx.moveTo(p.x, p.y);
        ctx.lineTo(q.x, q.y);
      });
    });
    ctx.stroke();

    // --- 三条主轴：从最近的底角出发，带刻度与数字 ---
    const fc = floorCorner(cam, box);
    const axes = [
      { end: [other(fc.x, x0, x1), fc.y, fc.z], name: 'X', vals: ticks[0],
        at: (v) => [v, fc.y, fc.z] },
      { end: [fc.x, other(fc.y, y0, y1), fc.z], name: 'Y', vals: ticks[1],
        at: (v) => [fc.x, v, fc.z] },
      { end: [fc.x, fc.y, z1], name: 'Z', vals: ticks[2],
        at: (v) => [fc.x, fc.y, v] },
    ];

    const centerP = proj(cam, cam.center[0], cam.center[1], cam.center[2]);

    ctx.font = FONT;
    axes.forEach((ax) => {
      const a = proj(cam, fc.x, fc.y, fc.z);
      const b = proj(cam, ax.end[0], ax.end[1], ax.end[2]);

      ctx.strokeStyle = c.axis;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();

      // 刻度数字一律排在背离盒心的那一侧（整条轴只定一次方向，免得数字来回跳）
      const len = Math.hypot(b.x - a.x, b.y - a.y) || 1;
      let px = -(b.y - a.y) / len;
      let py = (b.x - a.x) / len;
      const mx = (a.x + b.x) / 2 - centerP.x;
      const my = (a.y + b.y) / 2 - centerP.y;
      if (px * mx + py * my < 0) { px = -px; py = -py; }

      ctx.strokeStyle = c.axis;
      ctx.fillStyle = c.muted;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ax.vals.forEach((v) => {
        const w = ax.at(v);
        const p = proj(cam, w[0], w[1], w[2]);
        ctx.beginPath();
        ctx.moveTo(p.x, p.y);
        ctx.lineTo(p.x + px * 5, p.y + py * 5);
        ctx.stroke();
        ctx.fillText(fmtTick(v, false), p.x + px * 15, p.y + py * 15);
      });

      ctx.fillStyle = c.secondary;
      ctx.font = '13px system-ui, "Segoe UI", "Microsoft YaHei", sans-serif';
      ctx.fillText(ax.name, b.x + px * 20, b.y + py * 20);
      ctx.font = FONT;
    });
  }

  /* ---------------- 链 ---------------- */

  /** 投影整条链，返回屏幕坐标与深度（动画中只画前缀，但深度域用整条链，避免分带跳动）。 */
  function projectChain(cam, flat, nseg) {
    const m = nseg + 1;
    const px = new Float64Array(m);
    const py = new Float64Array(m);
    const pd = new Float64Array(m);
    for (let i = 0; i < m; i++) {
      const p = proj(cam, flat[i * 3], flat[i * 3 + 1], flat[i * 3 + 2]);
      px[i] = p.x;
      py[i] = p.y;
      pd[i] = p.d;
    }
    return { px, py, pd };
  }

  /** 所有链合起来的深度域。只用某一条链的域会把别的链压进单个带，粗细就失真的。 */
  function depthRange(cam, chains) {
    let lo = Infinity;
    let hi = -Infinity;
    const f = cam.b.forward;
    for (const flat of chains) {
      for (let i = 0; i < flat.length; i += 3) {
        const d = (flat[i] - cam.center[0]) * f[0]
                + (flat[i + 1] - cam.center[1]) * f[1]
                + (flat[i + 2] - cam.center[2]) * f[2];
        if (d < lo) lo = d;
        if (d > hi) hi = d;
      }
    }
    return [lo, hi];
  }

  /**
   * 按深度分带、由远及近描边：8 次 stroke() 换来真实的纵深读感，代价可以忽略。
   * 远的细而淡、近的粗而实，和 matplotlib 那种一根匀线的观感差别很大。
   */
  function drawChain(ctx, cam, flat, nseg, color, dr, step, thin) {
    const { px, py, pd } = projectChain(cam, flat, nseg);
    const [lo, hi] = dr;
    const span = hi - lo || 1;

    const buckets = [];
    for (let b = 0; b < BANDS; b++) buckets.push([]);
    for (let i = 0; i < nseg; i += step) {
      const dm = (pd[i] + pd[Math.min(i + step, nseg)]) / 2;
      let b = Math.floor(((dm - lo) / span) * BANDS);
      if (b < 0) b = 0;
      if (b >= BANDS) b = BANDS - 1;
      buckets[b].push(i);
    }

    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = color;
    for (let b = 0; b < BANDS; b++) {
      const segs = buckets[b];
      if (!segs.length) continue;
      const f = (b + 0.5) / BANDS;
      // 纵深主要靠**线宽**承担，透明度只做微调。
      // 一开始用 0.45→1.0 的 alpha，结果最远的那条链淡到几乎看不见，
      // 而且 alpha 混合会把已验证的系列色对比度拉到 1.8:1 —— 低于可读下限。
      // 线宽不改变颜色，对比度恒定，纵深照样读得出来。
      ctx.globalAlpha = 0.82 + 0.18 * f;
      ctx.lineWidth = thin ? 1.1 : 1.1 + 1.1 * f;
      ctx.beginPath();
      for (const i of segs) {
        const j = Math.min(i + step, nseg);
        ctx.moveTo(px[i], py[i]);
        ctx.lineTo(px[j], py[j]);
      }
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    return { px, py };
  }

  /* ---------------- 主绘制 ---------------- */

  function draw() {
    const cv = $('chain-canvas');
    const d = state.data;
    if (!cv || !d) return;

    const ctx = cv.getContext('2d');
    const w = cv.clientWidth;
    const h = cv.clientHeight;
    if (!w || !h) return;

    if (!state.colors) readColors();
    const c = state.colors;

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, cv.width, cv.height);
    const dpr = cv.width / w;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    // 铺底色：否则导出的 PNG 是透明的
    ctx.fillStyle = c.surface;
    ctx.fillRect(0, 0, w, h);

    const box = boundsOf(d.points);
    const cam = makeCam(box, w, h);
    const ticks = [0, 1, 2].map((k) => axisTicks(box.min[k], box.max[k], 4));

    drawFrame(ctx, cam, box, ticks);

    const nseg = state.frac >= 1
      ? d.n
      : Math.max(1, Math.round(state.frac * d.n));

    // R 矢量：虚线是「参考量」的语义，和分布图里的阈值线一致
    const end = { x: d.R[0][0], y: d.R[0][1], z: d.R[0][2] };
    const p0 = proj(cam, 0, 0, 0);
    const p1 = proj(cam, end.x, end.y, end.z);
    if (state.frac >= 1) {
      ctx.save();
      ctx.setLineDash([4, 3]);
      ctx.strokeStyle = c.muted;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(p0.x, p0.y);
      ctx.lineTo(p1.x, p1.y);
      ctx.stroke();
      ctx.restore();
    }

    // 拖拽中抽稀以保证跟手，松手后按原分辨率重绘一次（endDrag 会再调一次 draw）
    const total = d.n * d.chains;
    const step = state.drag && total > DRAG_MAX_SEGMENTS
      ? Math.ceil(total / DRAG_MAX_SEGMENTS)
      : 1;
    const thin = step > 1;
    const dr = depthRange(cam, d.points);

    const screen = [];
    for (let ci = 0; ci < d.chains; ci++) {
      screen.push(drawChain(ctx, cam, d.points[ci], nseg, c.series[ci % 5], dr, step, thin));
    }

    // --- 起止标记 ---
    // 单链时用已验证的 series 槽位（3=起点、2=终点）；多链时颜色槽被各条链占用，
    // 退回中性墨色，身份改由**形状 + 文字**承担。两种情况都不靠颜色单独区分。
    const multi = d.chains > 1;
    const startColor = multi ? c.ink : c.series[2];
    const endColor = multi ? c.ink : c.series[1];

    ctx.font = FONT;
    ctx.textBaseline = 'middle';

    const markStart = (p, color) => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
      ctx.fillStyle = c.surface;
      ctx.fill();
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.stroke();
    };
    const markEnd = (p, color) => {
      ctx.beginPath();
      ctx.rect(p.x - 4.5, p.y - 4.5, 9, 9);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = c.surface;
      ctx.lineWidth = 2;
      ctx.stroke();
    };
    const label = (p, text, color, dx, dy) => {
      ctx.fillStyle = color;
      ctx.textAlign = dx < 0 ? 'right' : 'left';
      ctx.fillText(text, p.x + dx, p.y + dy);
    };

    // 多链时只标第一条起止，否则 10 个标签会互相压住
    const labelCount = multi ? 1 : d.chains;
    for (let ci = 0; ci < labelCount; ci++) {
      const s = screen[ci];
      markStart({ x: s.px[0], y: s.py[0] }, startColor);
      label({ x: s.px[0], y: s.py[0] }, '起点', c.secondary, 9, -11);
      if (state.frac >= 1) {
        const p = proj(cam, d.R[ci][0], d.R[ci][1], d.R[ci][2]);
        markEnd(p, endColor);
        label(p, '终点', c.secondary, 9, -11);
      }
    }

    // --- R 的标注 ---
    if (state.frac >= 1) {
      ctx.font = FONT;
      ctx.fillStyle = c.secondary;
      ctx.textAlign = 'center';
      const mx = (p0.x + p1.x) / 2;
      const my = (p0.y + p1.y) / 2;
      ctx.fillText(`R${multi ? '₁' : ''} = ${fmt(d.R_mag[0])}`, mx, my - 9);
    }
    ctx.textAlign = 'left';
  }

  /* ---------------- 对照面板 ---------------- */

  function compareItem(key, val, cls) {
    const d = document.createElement('span');
    d.className = 'compare-item';
    const k = document.createElement('span');
    k.className = 'k';
    k.textContent = key;
    const v = document.createElement('span');
    v.className = 'v' + (cls ? ` ${cls}` : '');
    v.textContent = val;
    d.append(k, v);
    return d;
  }

  function renderCompare(d) {
    const box = $('chain-compare');
    box.textContent = '';

    const row1 = document.createElement('div');
    row1.className = 'compare-row';
    row1.append(
      compareItem('实测 R' + (d.chains > 1 ? '₁' : ''), fmt(d.R_mag[0])),
      compareItem('理论 h_rms = l√n', fmt(d.h_rms)),
      compareItem('最可几 h*', fmt(d.h_mp)),
      compareItem('R / h_rms', fmt(d.R_mag[0] / d.h_rms)),
    );
    box.appendChild(row1);

    const note = document.createElement('p');
    note.className = 'compare-note';

    if (d.chains > 1) {
      const dev = ((d.R2_mean - d.R2_theory) / d.R2_theory) * 100;
      const row2 = document.createElement('div');
      row2.className = 'compare-row';
      const devCls = Math.abs(dev) < 10 ? 'ok' : '';
      row2.append(
        compareItem(`${d.chains} 条链的 ⟨R²⟩`, fmt(d.R2_mean)),
        compareItem('理论 n·l²', fmt(d.R2_theory)),
        compareItem('相对偏差', `${dev >= 0 ? '+' : ''}${dev.toFixed(1)}%`, devCls),
      );
      box.appendChild(row2);

      // 说「收敛了」之前先看偏差有没有超出这个条数**本来就该有**的涨落。
      // 只画 5 条时固有涨落就有 ±36%，拿 −20% 说「已经很接近」是不诚实的。
      const se = d.R2_rel_se * 100;
      const within = Math.abs(dev) <= 2 * se;
      note.textContent =
        `${d.chains} 条链的 ⟨R²⟩ 与理论相差 ${dev >= 0 ? '+' : ''}${dev.toFixed(1)}%，`
        + `而只画 ${d.chains} 条时的固有涨落就有 ±${se.toFixed(1)}%`
        + `（√(2/3)/√${d.chains}）。这个偏差${within ? '落在正常范围内' : '偏大，多画几条会更稳'}。`;
    } else {
      const se = d.R2_rel_se * 100;
      note.textContent =
        `单条链的 R 是随机变量：它的 ⟨R²⟩ 相对标准误高达 ±${se.toFixed(0)}%，`
        + `所以 R 偏离 h_rms 是正常涨落，不是算错了。`
        + `把链数调到 5，看 ⟨R²⟩ 如何收敛到 n·l²。`;
    }
    box.appendChild(note);

    const seed = document.createElement('p');
    seed.className = 'compare-seed';
    seed.textContent = `随机种子 ${d.seed} —— 同一个种子画出同一条链。`;
    box.appendChild(seed);
  }

  /* ---------------- 图例 ---------------- */

  function renderLegend(d) {
    const box = $('chain-legend');
    box.textContent = '';
    if (d.chains < 2) { box.hidden = true; return; }
    box.hidden = false;
    for (let i = 0; i < d.chains; i++) {
      const item = document.createElement('span');
      item.className = 'legend-item';
      const k = document.createElement('span');
      k.className = 'legend-key';
      k.style.background = `var(--series-${(i % 5) + 1})`;
      const t = document.createElement('span');
      t.textContent = `链 ${i + 1}　R = ${fmt(d.R_mag[i])}`;
      item.append(k, t);
      box.appendChild(item);
    }
  }

  /* ---------------- 数据 ---------------- */

  async function fetchChain() {
    const cv = $('chain-canvas');
    const nRaw = $('c-n').value;
    const chainsRaw = $('c-chains').value;
    const seedRaw = $('c-seed').value.trim();
    const l = Number($('in-l').value);

    if (!isFinite(l) || l <= 0) {
      setError('链段长度 l 必须是大于 0 的数字');
      return;
    }

    const my = ++state.seq;
    $('chain-body').classList.add('busy');
    let data;
    try {
      const res = await fetch('/api/chain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          n: nRaw === '' ? null : Number(nRaw),
          l,
          chains: chainsRaw === '' ? 1 : Number(chainsRaw),
          seed: seedRaw === '' ? null : Number(seedRaw),
        }),
      });
      data = await res.json().catch(() => ({ error: '服务器返回了无法解析的内容' }));
      if (!res.ok) throw new Error(data.error || `请求失败（HTTP ${res.status}）`);
    } catch (e) {
      if (my === state.seq) {
        setError(e.message);
        $('chain-body').classList.remove('busy');
      }
      return;
    }
    if (my !== state.seq) return;   // 已经有更新的请求了，丢弃这次

    // 没给种子时把后端现取的那个填回输入框，用户才看得到、才能复现
    if (seedRaw === '') $('c-seed').value = String(data.seed);

    state.data = data;
    state.frac = 1;
    setError(null, data.warnings);
    renderCompare(data);
    renderLegend(data);
    $('chain-body').classList.remove('busy');
    if (cv) draw();
  }

  function setError(msg, warnings) {
    const box = $('chain-alerts');
    box.textContent = '';
    const add = (kind, tag, text) => {
      const d = document.createElement('div');
      d.className = `alert alert-${kind}`;
      const t = document.createElement('span');
      t.className = 'tag';
      t.textContent = tag;
      const m = document.createElement('span');
      m.textContent = text;
      d.append(t, m);
      box.appendChild(d);
    };
    if (msg) add('error', '错误', msg);
    (warnings || []).forEach((w) => add('warn', '注意', w));
  }

  /* ---------------- 尺寸 ---------------- */

  function resize() {
    const cv = $('chain-canvas');
    if (!cv) return;
    const w = cv.clientWidth;
    const h = cv.clientHeight;
    if (!w || !h) return;
    const dpr = window.devicePixelRatio || 1;
    const bw = Math.round(w * dpr);
    const bh = Math.round(h * dpr);
    if (cv.width !== bw || cv.height !== bh) {
      cv.width = bw;
      cv.height = bh;
    }
    draw();
  }

  /* ---------------- 生长动画 ---------------- */

  function reduceMotion() {
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  }

  function play() {
    if (!state.data) return;
    if (state.animId !== null) cancelAnimationFrame(state.animId);
    if (reduceMotion()) { state.frac = 1; draw(); return; }
    const t0 = performance.now();
    const tick = (now) => {
      const t = Math.min(1, (now - t0) / ANIM_MS);
      state.frac = t;
      draw();
      state.animId = t < 1 ? requestAnimationFrame(tick) : null;
    };
    state.frac = 0;
    state.animId = requestAnimationFrame(tick);
  }

  /* ---------------- 导出 ---------------- */

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

  const stem = (d) => `fjc-chain-n${d.n}-seed${d.seed}`;

  function exportPNG() {
    const cv = $('chain-canvas');
    if (!cv || !state.data) return;
    // 导出前确保画的是完整构象（动画中途导出会得到半条链）
    if (state.frac < 1) { state.frac = 1; draw(); }
    cv.toBlob((blob) => {
      if (blob) download(blob, `${stem(state.data)}.png`);
    }, 'image/png');
  }

  function exportCSV() {
    const d = state.data;
    if (!d) return;
    // 纯数字 + ASCII 表头，不加 BOM（加了反而会在某些解析器里多出一个字符）
    const lines = ['chain,segment,x,y,z'];
    d.points.forEach((flat, ci) => {
      for (let i = 0; i <= d.n; i++) {
        lines.push(`${ci},${i},${flat[i * 3]},${flat[i * 3 + 1]},${flat[i * 3 + 2]}`);
      }
    });
    download(new Blob([lines.join('\r\n')], { type: 'text/csv;charset=utf-8' }),
      `${stem(d)}.csv`);
  }

  /* ---------------- 交互 ---------------- */

  function resetView() {
    view.azim = DEF.azim;
    view.elev = DEF.elev;
    view.zoom = DEF.zoom;
    draw();
  }

  function initEvents() {
    const cv = $('chain-canvas');

    const schedule = () => {
      clearTimeout(fetchTimer);
      fetchTimer = setTimeout(fetchChain, 180);
    };

    $('c-n').addEventListener('input', schedule);
    $('c-chains').addEventListener('input', schedule);
    $('c-seed').addEventListener('input', schedule);
    // l 与上方分布图共用，改 l 时这张图也要跟着变
    $('in-l').addEventListener('input', schedule);

    $('c-grid').addEventListener('change', () => {
      state.grid = $('c-grid').checked;
      draw();
    });

    $('c-new').addEventListener('click', () => {
      $('c-seed').value = String(Math.floor(Math.random() * 2147483647));
      fetchChain();
    });

    $('c-replay').addEventListener('click', play);
    $('c-png').addEventListener('click', exportPNG);
    $('c-csv').addEventListener('click', exportCSV);

    // --- 拖动旋转 ---
    cv.addEventListener('pointerdown', (ev) => {
      if (state.animId !== null) cancelAnimationFrame(state.animId);
      state.animId = null;
      state.frac = 1;
      state.drag = { x: ev.clientX, y: ev.clientY };
      // 某些环境（合成事件、无活动指针）下 setPointerCapture 会抛 NotFoundError，
      // 捕获失败只是拖出画布后不再跟随，不该让整个拖动失效
      try { cv.setPointerCapture(ev.pointerId); } catch (e) { /* 忽略 */ }
      cv.style.cursor = 'grabbing';
    });
    cv.addEventListener('pointermove', (ev) => {
      if (!state.drag) return;
      const dx = ev.clientX - state.drag.x;
      const dy = ev.clientY - state.drag.y;
      state.drag = { x: ev.clientX, y: ev.clientY };
      view.azim -= dx * 0.008;
      view.elev = Math.max(-1.5533, Math.min(1.5533, view.elev + dy * 0.008));  // ±89°
      draw();
    });
    const endDrag = (ev) => {
      if (!state.drag) return;
      state.drag = null;
      cv.style.cursor = 'grab';
      try {
        if (cv.hasPointerCapture(ev.pointerId)) cv.releasePointerCapture(ev.pointerId);
      } catch (e) { /* 忽略 */ }
      draw();   // 松手后按原分辨率重绘一次
    };
    cv.addEventListener('pointerup', endDrag);
    cv.addEventListener('pointercancel', endDrag);

    cv.addEventListener('wheel', (ev) => {
      ev.preventDefault();
      view.zoom = Math.max(0.4, Math.min(4, view.zoom * Math.exp(-ev.deltaY * 0.0012)));
      draw();
    }, { passive: false });

    cv.addEventListener('dblclick', resetView);

    cv.addEventListener('keydown', (ev) => {
      const step = ev.shiftKey ? 0.15 : 0.05;
      let used = true;
      switch (ev.key) {
        case 'ArrowLeft': view.azim -= step; break;
        case 'ArrowRight': view.azim += step; break;
        case 'ArrowUp': view.elev = Math.min(1.5533, view.elev + step); break;
        case 'ArrowDown': view.elev = Math.max(-1.5533, view.elev - step); break;
        case '+': case '=': view.zoom = Math.min(4, view.zoom * 1.15); break;
        case '-': case '_': view.zoom = Math.max(0.4, view.zoom / 1.15); break;
        case '0': resetView(); return;
        default: used = false;
      }
      if (used) { ev.preventDefault(); draw(); }
    });

    cv.style.cursor = 'grab';

    // 尺寸变化（含宽屏/窄屏切换）时同步 backing store
    if (window.ResizeObserver) {
      new ResizeObserver(resize).observe($('chain-body'));
    } else {
      window.addEventListener('resize', resize);
    }

    // 深浅色：canvas 取不到 CSS 变量的自动更新，必须自己监听后重绘
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    if (mq.addEventListener) mq.addEventListener('change', () => { readColors(); draw(); });
    new MutationObserver(() => { readColors(); draw(); })
      .observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  }

  /* ---------------- 启动 ---------------- */

  readColors();
  initEvents();
  resize();
  fetchChain();
  /* ---------------- 给 AI 助手的接口 ----------------
   * 和 app.js 末尾那份同一个规矩：**setter 走界面自己的路** ——
   * 改输入框就派发它自己的 input 事件（好让 initEvents 里那个防抖重新取数），
   * 改视角就直接改 view 再 draw()。暴露动作，不暴露状态对象。
   */
  window.fjc3d = {
    getState() {
      const d = state.data;
      return {
        n: $('c-n').value,
        chains: $('c-chains').value,
        seed: $('c-seed').value,
        grid: $('c-grid').checked,
        azim_deg: view.azim * 180 / Math.PI,
        elev_deg: view.elev * 180 / Math.PI,
        zoom: view.zoom,
        drawn: !!d,
        // 这几个是这一帧真正画出来的那张链的读数（不是输入框里的值）
        R: d ? d.R_mag[0] : null,
        R2_mean: d ? d.R2_mean : null,
        R2_theory: d ? d.R2_theory : null,
      };
    },

    /** 改 3D 卡片的参数。这里的 l 没有：3D 卡片和上方分布图共用主界面的 l。 */
    async setChainParams(args) {
      const a = args || {};
      const touched = [];
      if (a.n !== undefined) { $('c-n').value = String(a.n); touched.push('c-n'); }
      if (a.chains !== undefined) { $('c-chains').value = String(a.chains); touched.push('c-chains'); }
      if (a.seed !== undefined) { $('c-seed').value = String(a.seed); touched.push('c-seed'); }
      if (!touched.length) return { ok: false, error: '没给任何 3D 参数' };

      // 派发 input 只是为了把 initEvents 的监听器叫醒；紧接着撤掉它排的那次防抖，
      // 自己 await 一次 fetchChain —— 否则这里没法知道它什么时候画完。
      touched.forEach((id) => $(id).dispatchEvent(new Event('input', { bubbles: true })));
      clearTimeout(fetchTimer);
      await fetchChain();
      if (!state.data) {
        return { ok: false, error: '这条链没画出来，请看 3D 卡片上的提示' };
      }
      return { ok: true, data: this.getState() };
    },

    /** 视角。azim / elev 用**角度**（界面里是弧度），zoom 夹到 0.4–4。 */
    setView(args) {
      const a = args || {};
      const deg = Math.PI / 180;
      if (typeof a.azim === 'number' && isFinite(a.azim)) view.azim = a.azim * deg;
      if (typeof a.elev === 'number' && isFinite(a.elev)) {
        view.elev = Math.max(-1.5533, Math.min(1.5533, a.elev * deg));
      }
      if (typeof a.zoom === 'number' && isFinite(a.zoom) && a.zoom > 0) {
        view.zoom = Math.max(0.4, Math.min(4, a.zoom));
      }
      draw();
      return { ok: true, data: this.getState() };
    },

    /** 链生长动画。开了「减少动态效果」时 play() 会直接画完整条链，这是它的既定行为。 */
    play() {
      if (!state.data) return { ok: false, error: '还没画出链来' };
      play();
      return { ok: true, data: { playing: !reduceMotion() } };
    },
  };
})();
