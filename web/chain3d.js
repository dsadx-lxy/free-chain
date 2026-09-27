'use strict';

/* 高分子链构象计算器 —— 3D 单链构象视图
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
  /* 画布内边距。刻度数字挂在轴外侧 15px 处、再要占掉半行字（十来像素），
     所以边上至少得留 40px 出头；34 在画布变矮变宽之后会把最外面那排数字顶出画布。 */
  const MARGIN = 48;
  const DRAG_MAX_SEGMENTS = 20000;        // 拖拽时最多画这么多段，超了就抽稀
  const ANIM_MS = 2500;                   // 生长动画总时长，与 n 无关
  const DEF = { azim: -Math.PI / 3, elev: (20 * Math.PI) / 180, zoom: 1 };
  const FONT = '12px system-ui, "Segoe UI", "Microsoft YaHei", sans-serif';

  const $ = (id) => document.getElementById(id);
  /** 读一个显示开关。控件不在页面上时按「开」处理 —— 宁可多画一点，
   *  也不要在缺一个复选框时整张图变得什么都没有。 */
  const opt = (id) => { const e = $(id); return e ? e.checked : true; };

  /** 把任意 CSS 颜色规范化成 [r,g,b]：先写进 canvas 的 fillStyle 再读回来，
   *  hex / rgb() / 命名色都能吃，不用自己写一堆解析分支。 */
  function toRGB(ctx, css) {
    ctx.fillStyle = '#000000';
    ctx.fillStyle = css;
    const s = ctx.fillStyle;
    if (s[0] === '#') {
      return [parseInt(s.slice(1, 3), 16), parseInt(s.slice(3, 5), 16), parseInt(s.slice(5, 7), 16)];
    }
    const m = (s.match(/\d+/g) || [0, 0, 0]).map(Number);
    return [m[0], m[1], m[2]];
  }

  function mix(a, b, t) {
    return `rgb(${Math.round(a[0] + (b[0] - a[0]) * t)},`
      + `${Math.round(a[1] + (b[1] - a[1]) * t)},${Math.round(a[2] + (b[2] - a[2]) * t)})`;
  }

  const view = { azim: DEF.azim, elev: DEF.elev, zoom: DEF.zoom };
  const state = {
    data: null,
    grid: true,
    sel: 0,        // 当前查看的是第几条链（0 起）—— 对照面板、图例高亮、加粗都看它
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
      accent: g('--accent'),
      accent2: g('--accent-2'),
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
  function drawChain(ctx, cam, flat, nseg, color, color2, dr, step, thin, bold) {
    const { px, py, pd } = projectChain(cam, flat, nseg);
    const [lo, hi] = dr;
    const span = hi - lo || 1;
    /* 单链时沿链做一次颜色过渡（起点色 → 终点色，和起止标记同一套语义）；
       多链时**保持纯色** —— 那几条链的颜色是用来区分「哪条是哪条」的，
       再叠一层渐变就把身份信息搅浑了。 */
    const NB = color2 ? 6 : 1;
    const rgb1 = color2 ? toRGB(ctx, color) : null;
    const rgb2 = color2 ? toRGB(ctx, color2) : null;

    /* 一层「光晕」：同一根线先用几倍线宽、很低的不透明度铺一遍，再把实线压上去。
       它不参与任何数据表达（颜色和线宽仍然只由深度决定），纯粹让线看起来是发光的。
       实线那一遍完全不碰透明度 —— 原因见下面那段注释（alpha 会把已校验的
       系列色对比度拉到可读下限以下）。 */
    ctx.save();
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = color;
    ctx.globalAlpha = bold ? 0.22 : (thin ? 0.07 : 0.12);
    ctx.lineWidth = bold ? 6 : (thin ? 3.5 : 4.5);
    ctx.beginPath();
    for (let i = 0; i < nseg; i += step) {
      const j = Math.min(i + step, nseg);
      ctx.moveTo(px[i], py[i]);
      ctx.lineTo(px[j], py[j]);
    }
    ctx.stroke();
    ctx.restore();

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
    for (let b = 0; b < BANDS; b++) {
      const segs = buckets[b];
      if (!segs.length) continue;
      const f = (b + 0.5) / BANDS;
      // 纵深主要靠**线宽**承担，透明度只做微调。
      // 一开始用 0.45→1.0 的 alpha，结果最远的那条链淡到几乎看不见，
      // 而且 alpha 混合会把已验证的系列色对比度拉到 1.8:1 —— 低于可读下限。
      // 线宽不改变颜色，对比度恒定，纵深照样读得出来。
      ctx.globalAlpha = 0.82 + 0.18 * f;
      // 选中的那条靠**线宽与光晕**区分，不去调淡别的链 ——
      // 系列色是验过对比度的，把它们压暗会掉到可读下限以下。
      ctx.lineWidth = bold ? 1.9 + 1.6 * f : (thin ? 1.1 : 1.1 + 1.1 * f);
      for (let q = 0; q < NB; q++) {
        const from = (q * nseg) / NB;
        const to = ((q + 1) * nseg) / NB;
        let drew = false;
        ctx.strokeStyle = color2 ? mix(rgb1, rgb2, (q + 0.5) / NB) : color;
        ctx.beginPath();
        for (const i of segs) {
          if (i < from || i >= to) continue;
          const j = Math.min(i + step, nseg);
          ctx.moveTo(px[i], py[i]);
          ctx.lineTo(px[j], py[j]);
          drew = true;
        }
        if (drew) ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
    return { px, py };
  }

  /* ---------------- 参考几何：h_rms 球、Kuhn 刻度、三轴 ---------------- */

  /** h_rms 参考球 —— 三条大圆（xy / xz / yz）拼出来的球骨架。
   *
   *  这是这张图最值得加的一笔：单条链的 R 偏离 h_rms 是**正常涨落**，
   *  可「偏多少算正常」光看数字没有几何感。把半径 h_rms 的球画出来，一眼就能
   *  看到终点落在球面附近；多条链时更能看出这一簇是围着球面散开的。
   *  虚线的语义和别的图里的阈值线一致：参考量、不是数据。
   */
  function drawRefSphere(ctx, cam, radius) {
    const c = state.colors;
    const N = 96;
    ctx.save();
    ctx.setLineDash([3, 4]);
    ctx.lineWidth = 1;
    ctx.strokeStyle = c.accent;
    ctx.globalAlpha = 0.45;
    for (const plane of [0, 1, 2]) {
      ctx.beginPath();
      for (let i = 0; i <= N; i++) {
        const a = (i / N) * Math.PI * 2;
        const u = radius * Math.cos(a);
        const v = radius * Math.sin(a);
        const p = plane === 0 ? proj(cam, u, v, 0)
          : plane === 1 ? proj(cam, u, 0, v)
            : proj(cam, 0, u, v);
        if (i === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y);
      }
      ctx.stroke();
    }
    ctx.restore();

    const at = proj(cam, radius * 0.7071, -radius * 0.7071, 0);
    ctx.save();
    ctx.font = FONT;
    ctx.fillStyle = c.accent;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(`h_rms = ${fmt(radius)}`, at.x + 6, at.y);
    ctx.restore();
  }

  /** 每走过一个 Kuhn 长度 b 就在链上点一个小点 —— 让「等价 Kuhn 长度」看得见。
   *  b/l < 2 时不画：FJC 的 b 就是一个链段，每段都点等于什么都没说。 */
  function drawKuhnMarks(ctx, cam, flat, nseg, every) {
    const c = state.colors;
    ctx.save();
    ctx.fillStyle = c.accent2;
    ctx.globalAlpha = 0.9;
    for (let i = every; i <= nseg; i += every) {
      const p = proj(cam, flat[i * 3], flat[i * 3 + 1], flat[i * 3 + 2]);
      ctx.beginPath();
      ctx.arc(p.x, p.y, 2.2, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }

  /** 右下角的三轴指示器（随视角旋转）。3D 图最容易迷路的就是「现在转到哪了」。 */
  function drawGizmo(ctx, w, h) {
    const c = state.colors;
    const b = basis();
    const ox = w - 46;
    const oy = h - 38;
    const L = 21;
    const axes = [
      ['x', [1, 0, 0], c.series[0]],
      ['y', [0, 1, 0], c.series[2]],
      ['z', [0, 0, 1], c.series[3]],
    ];
    ctx.save();
    ctx.font = FONT;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.lineWidth = 1.5;
    ctx.lineCap = 'round';
    for (const [name, v, color] of axes) {
      const dx = (v[0] * b.right[0] + v[1] * b.right[1] + v[2] * b.right[2]) * L;
      const dy = -(v[0] * b.up[0] + v[1] * b.up[1] + v[2] * b.up[2]) * L;
      ctx.globalAlpha = 0.85;
      ctx.strokeStyle = color;
      ctx.beginPath();
      ctx.moveTo(ox, oy);
      ctx.lineTo(ox + dx, oy + dy);
      ctx.stroke();
      ctx.fillStyle = color;
      ctx.fillText(name, ox + dx * 1.3, oy + dy * 1.3);
    }
    ctx.restore();
  }

  /** 左上角的 HUD 读数：模型、规模、本次实现的 R 与它和理论值的比。
   *  数字直接写在图上，看图就不用回头翻对照面板（导出的 PNG 也一样带着）。 */
  function drawHud(ctx, d) {
    const c = state.colors;
    const label = (d.params && d.params.label) || '';
    const rows = [ `${label}　n = ${d.n}　l = ${fmt(d.l)}` ];
    if (d.chains > 1) {
      const r2 = d.R2_mean / d.R2_theory;
      rows.push(`R₁ = ${fmt(d.R_mag[0])}　h_rms = ${fmt(d.h_rms)}`);
      rows.push(`⟨R²⟩ / ⟨h²⟩ = ${fmt(r2)}（${d.chains} 条链）`);
    } else {
      rows.push(`R = ${fmt(d.R_mag[0])}　h_rms = ${fmt(d.h_rms)}`);
      rows.push(`R / h_rms = ${fmt(d.R_mag[0] / d.h_rms)}`);
    }
    ctx.save();
    ctx.font = FONT;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    /* 先垫一层半透明底再写字。坐标轴的数字是挂在盒角外侧的，投影一转就可能
       正好转到左上角来 —— 没有这层底，两段文字会叠在一起谁也看不清。
       底衬比文字大一圈，所以它同时也把 HUD 框成了一块「仪表读数」。 */
    const padX = 7;
    const padY = 5;
    const widest = Math.max(...rows.map((t) => ctx.measureText(t).width));
    ctx.fillStyle = c.surface;
    ctx.globalAlpha = 0.72;
    ctx.beginPath();
    ctx.rect(MARGIN - 28 - padX, MARGIN - 34 - padY,
             widest + padX * 2, rows.length * 15 + padY * 2);
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.fillStyle = c.secondary;
    rows.forEach((t, i) => ctx.fillText(t, MARGIN - 28, MARGIN - 34 + i * 15));
    ctx.restore();
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
    // 参考球的半径也要算进取景框，否则球会被画到画布外面（n 小的时候尤其明显）
    const hr = Number(d.h_rms) || 0;
    const showSphere = opt('c-sphere');
    if (showSphere) {
      for (let k = 0; k < 3; k++) {
        box.min[k] = Math.min(box.min[k], -hr);
        box.max[k] = Math.max(box.max[k], hr);
      }
    }
    const cam = makeCam(box, w, h);
    const ticks = [0, 1, 2].map((k) => axisTicks(box.min[k], box.max[k], 4));

    // 坐标框（含刻度、网格底板、三轴指示器）可以整体关掉 —— 只想看构象时干净
    const showAxis = opt('c-axis');
    if (showAxis) drawFrame(ctx, cam, box, ticks);
    if (showSphere) drawRefSphere(ctx, cam, Math.max(hr, 1e-9));

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
      // 只有一条链时才做沿链渐变（多链时颜色要留给「哪条链」）：
      // 起点用 series[2]、终点用 series[1]，和下面的起止标记同一套色
      const single = d.chains === 1;
      screen.push(drawChain(ctx, cam, d.points[ci], nseg, c.series[ci % 5],
                            single ? c.series[1] : null, dr, step, thin,
                            ci === state.sel));
    }
    state.cam = cam;        // 画布点击要用它做命中判断（找离点击处最近的那条链）

    // 生长动画的头部：一个会发光的点，动画走着的时候一眼看得到「头」在哪
    if (state.frac < 1) {
      const s = screen[0];
      const tip = Math.min(d.n, Math.max(1, Math.round(state.frac * d.n)));
      const g = ctx.createRadialGradient(s.px[tip], s.py[tip], 0, s.px[tip], s.py[tip], 15);
      g.addColorStop(0, 'rgba(255,255,255,0.9)');
      g.addColorStop(0.4, c.accent);
      g.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.save();
      ctx.globalAlpha = 0.85;
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(s.px[tip], s.py[tip], 15, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }

    // Kuhn 刻度：b/l ≥ 2 才有意义（FJC 的 b 就是一个链段）
    const kuhnEvery = Math.round((Number(d.kuhn_length) || 0) / (Number(d.l) || 1));
    if (opt('c-kuhn') && kuhnEvery >= 2 && state.frac >= 1) {
      for (let ci = 0; ci < d.chains; ci++) {
        drawKuhnMarks(ctx, cam, d.points[ci], nseg, kuhnEvery);
      }
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

    // 多链时只标**当前查看的那条**的起止，否则 10 个标签会互相压住
    const marked = multi ? [state.sel] : screen.map((_, i) => i);
    for (const ci of marked) {
      if (!screen[ci]) continue;
      const s = screen[ci];
      markStart({ x: s.px[0], y: s.py[0] }, startColor);
      label({ x: s.px[0], y: s.py[0] }, '起点', c.secondary, 9, -11);
      if (state.frac >= 1) {
        const p = proj(cam, d.R[ci][0], d.R[ci][1], d.R[ci][2]);
        // 终点外面再套一圈很淡的光环：一眼能找到「这条链走到哪了」
        ctx.save();
        ctx.globalAlpha = 0.28;
        ctx.strokeStyle = endColor;
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 10, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
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
      ctx.fillText(`R${multi ? state.sel + 1 : ''} = ${fmt(d.R_mag[state.sel])}`, mx, my - 9);
    }
    ctx.textAlign = 'left';
    if (opt('c-hud')) drawHud(ctx, d);
    if (showAxis) drawGizmo(ctx, w, h);
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
    const i = Math.min(state.sel, d.chains - 1);
    row1.append(
      compareItem('实测 R' + (d.chains > 1 ? `（链 ${i + 1}）` : ''), fmt(d.R_mag[i])),
      compareItem('理论 h_rms（本模型）', fmt(d.h_rms)),
      compareItem('最可几 h*', fmt(d.h_mp)),
      compareItem('R / h_rms', fmt(d.R_mag[i] / d.h_rms)),
    );
    box.appendChild(row1);
    if (d.chains > 1) {
      // 末端矢量也摆出来：选中某一条时，这三个分量最能说明它朝哪边伸
      const row2 = document.createElement('div');
      row2.className = 'compare-row';
      row2.append(
        compareItem('末端矢量 x', fmt(d.R[i][0])),
        compareItem('y', fmt(d.R[i][1])),
        compareItem('z', fmt(d.R[i][2])),
      );
      box.appendChild(row2);
    }

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
    const extra = [];
    const kuhnEvery = Math.round((Number(d.kuhn_length) || 0) / (Number(d.l) || 1));
    if (kuhnEvery >= 2) {
      extra.push(['dot', `● 每过一个 Kuhn 长度 b = ${fmt(d.kuhn_length)}`]);
    }
    extra.push(['dash', '┄ h_rms 参考球（虚线，参考量）']);
    if (d.chains < 2 && !extra.length) { box.hidden = true; return; }
    box.hidden = false;
    for (let i = 0; i < d.chains; i++) {
      const item = document.createElement('span');
      item.className = 'legend-item' + (d.chains > 1 && i === state.sel ? ' on' : '');
      if (d.chains > 1) {
        // 点图例里的一条链 = 切到看它的数据。键盘也要能选，所以给 role/tabindex，
        // 而不是只挂一个 click。悬停/选中的样式在 style.css 里。
        item.setAttribute('role', 'button');
        item.setAttribute('tabindex', '0');
        item.setAttribute('aria-pressed', String(i === state.sel));
        item.title = `查看链 ${i + 1} 的数据`;
        const pick = () => {
          state.sel = i;
          renderCompare(state.data);
          renderLegend(state.data);
          draw();
        };
        item.addEventListener('click', pick);
        item.addEventListener('keydown', (ev) => {
          if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); pick(); }
        });
      }
      const k = document.createElement('span');
      k.className = 'legend-key';
      k.style.background = `var(--series-${(i % 5) + 1})`;
      const t = document.createElement('span');
      t.textContent = `链 ${i + 1}　R = ${fmt(d.R_mag[i])}`;
      item.append(k, t);
      box.appendChild(item);
    }
    extra.forEach(([kind, text]) => {
      const item = document.createElement('span');
      item.className = 'legend-item';
      const k = document.createElement('span');
      k.className = `legend-key ${kind === 'dash' ? 'dash' : 'dot'}`;
      if (kind === 'dot') k.style.background = 'var(--accent-2)';
      const t = document.createElement('span');
      t.textContent = text;
      item.append(k, t);
      box.appendChild(item);
    });
  }

  /* ---------------- 数据 ---------------- */

  /* ---------------- 跟随上面的 n ---------------- */

  // 与后端 CHAIN_N_MAX 对齐：超过这个数画出来就是一团结，3D 视图不收
  const N_MAX_DRAW = 20000;

  /** 把顶部参数行那套参数导进这张卡片：**链段数 n + 画几条链**。
   *
   *  规则（都是「宁可说清楚，也不要悄悄画一个别的」）：
   *   · n 取顶部**第一个**值（顶部可以填多个做对比）；
   *   · **画几条链取顶部 n 的个数**（夹到 1–5）—— 上面填「100, 300」时这里就画
   *     两条 n = 100 的独立链，正好用来看 ⟨R²⟩ 怎么往理论值收。
   *     注意每条链都用**第一个 n**：后端一次只采一个 n，真要看不同 n 的构象，
   *     得把上面的 n 分别改一次各看一遍。
   *   · 超过 3D 的绘制上限时不跟，保留当前值并在告警条里说明；
   *   · 值没变就不动 —— 免得每敲一个字符都重取一次链。
   *  返回 true 表示「已经派发过 input」，调用方不用再管。
   */
  function syncFromTop() {
    const follow = $('c-follow');
    if (!follow || !follow.checked) return false;
    const vals = String($('in-n').value || '')
      .split(/[,，;；\s]+/).filter((s) => s.length)
      .map((s) => Math.floor(Number(s)))
      .filter((v) => isFinite(v) && v >= 1);
    if (!vals.length) return false;
    const n = vals[0];
    const box = $('c-n');
    const boxChains = $('c-chains');
    const chains = Math.min(5, Math.max(1, vals.length));
    if (n > N_MAX_DRAW) {
      setError(`上面的 n = ${n} 超过 3D 视图的绘制上限 ${N_MAX_DRAW}`
        + `（再多画出来就是一团结），这里仍按 n = ${box.value} 画；`
        + '想单独指定就取消勾选「跟随上面的参数」。');
      return false;
    }
    if (String(n) === box.value.trim() && String(chains) === boxChains.value.trim()) {
      return false;
    }
    box.value = String(n);
    boxChains.value = String(chains);
    box.dispatchEvent(new Event('input', { bubbles: true }));
    boxChains.dispatchEvent(new Event('input', { bubbles: true }));
    return true;
  }

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
          // 3D 构象也按当前链模型采样：自由旋转链固定键角、受累旋转链带内旋转、
          // 蠕虫状链按弯曲刚度 —— 图上要能看出模型之间的差别。
          ...(window.fjcApp ? window.fjcApp.modelParams() : {}),
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
    // 链数变了（或换了种子）以后，选中的下标可能越界 —— 夹一下，不重置，
    // 免得每换一个种子就把用户选的链丢回第一条
    state.sel = Math.min(state.sel, data.chains - 1);
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
    // 链模型和它的三个参数都会改变采样出来的构象：换模型要重取一次，
    // 否则图上还是上一条链，看着像「切了模型但 3D 没动」。
    ['in-model', 'in-theta', 'in-cosphi', 'in-p'].forEach((id) => {
      const el = $(id);
      if (!el) return;
      el.addEventListener('change', schedule);
      el.addEventListener('input', schedule);
    });

    $('c-grid').addEventListener('change', () => {
      state.grid = $('c-grid').checked;
      draw();
    });
    // 跟随上面的参数：勾上时这两个输入框由顶部驱动（置灰，免得改了又被覆盖）
    const follow = $('c-follow');
    if (follow) {
      const apply = () => {
        $('c-n').disabled = follow.checked;
        $('c-chains').disabled = follow.checked;
        if (follow.checked) syncFromTop();
      };
      follow.addEventListener('change', apply);
      apply();
    }
    $('in-n').addEventListener('input', syncFromTop);
    // 四个显示开关只重画一遍：它们不改数据、也不重新取数
    ['c-axis', 'c-sphere', 'c-kuhn', 'c-hud'].forEach((id) => {
      const el = $(id);
      if (el) el.addEventListener('change', draw);
    });
    // 点画布选链：找离点击处最近的**末端点**（40px 以内才算命中）。
    // 拖拽过就不算点击 —— 否则转一下视角就会把选中的链换掉。
    let downAt = null;
    cv.addEventListener('pointerdown', (ev) => { downAt = { x: ev.clientX, y: ev.clientY }; });
    cv.addEventListener('click', (ev) => {
      const d = state.data;
      if (!d || d.chains < 2 || !state.cam) return;
      if (downAt && Math.hypot(ev.clientX - downAt.x, ev.clientY - downAt.y) > 4) return;
      const rect = cv.getBoundingClientRect();
      const mx = ev.clientX - rect.left;
      const my = ev.clientY - rect.top;
      let best = -1;
      let bestD = 40;
      for (let i = 0; i < d.chains; i++) {
        const p = proj(state.cam, d.R[i][0], d.R[i][1], d.R[i][2]);
        const dist = Math.hypot(p.x - mx, p.y - my);
        if (dist < bestD) { bestD = dist; best = i; }
      }
      if (best >= 0 && best !== state.sel) {
        state.sel = best;
        renderCompare(d);
        renderLegend(d);
        draw();
      }
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
