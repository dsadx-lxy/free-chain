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
  // 自回避一侧的 n 上限，与 fjc_core.SAW_N_MAX 一致（后端也会拦，返 400）。
  // 这是**算力**上限：严格拒绝采样的存活率按 0.78^n 衰减。
  const SAW_N_MAX = 30;

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
    // 自回避对照：开了以后 state.data 不参与绘制，画的是 state.saw 那一对格点链
    sawOn: false,
    saw: null,
    sawSeq: 0,        // 单独一个序号：两条取数通路各丢各的过期响应
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

  /**
   * 按**投影后**的实际跨度把 cam.scale 放大到填满视口（只放大，不缩小）。
   *
   * `makeCam` 用的是包围盒的 3D 半对角线：对「一条链」正好，因为链的走向会填满
   * 那个盒子。但自回避对照那边框的是两团各 20 条的**并集**，半对角线远大于投影
   * 出来的跨度，结果两团缩在画布中间一小块。这里按投影值重新定一次比例尺 ——
   * 投影是线性的，所以「算出跨度再按比值缩放」和「先缩放再算跨度」等价，
   * 一次就够，不用迭代。
   *
   * 上下留的边距不对称：半边顶上要放三行说明文字（标题 / ⟨R²⟩ / 溶胀），
   * 链条不能顶到那上面去。缩放绕 (cx, cy) 做，而链团**并不**以它为中心
   * （所有链都从同一点出发，形状是偏的），所以缩完还要平移到留好边距的框里居中。
   */
  function fitScaleToProjection(cam, chains, w, h) {
    let x1 = Infinity, y1 = Infinity, x2 = -Infinity, y2 = -Infinity;
    for (const flat of chains) {
      for (let i = 0; i < flat.length; i += 3) {
        const p = proj(cam, flat[i], flat[i + 1], flat[i + 2]);
        if (p.x < x1) x1 = p.x;
        if (p.x > x2) x2 = p.x;
        if (p.y < y1) y1 = p.y;
        if (p.y > y2) y2 = p.y;
      }
    }
    const spanX = x2 - x1;
    const spanY = y2 - y1;
    if (!(spanX > 0) || !(spanY > 0)) return;
    const PAD_X = 26;
    const PAD_TOP = 84;      // 三行说明文字 + 一点呼吸
    const PAD_BOTTOM = 26;
    const k = Math.min(
      (w - PAD_X * 2) / spanX,
      (h - PAD_TOP - PAD_BOTTOM) / spanY,
    );
    if (!(k > 1.02)) return;   // 已经填满了就别动它
    cam.scale *= k;
    // 缩放后某个原来在 x 的点落到了 cam.cx + k(x − cam.cx)（y 是反的），
    // 平移量就是把实际范围挪进「留好边距、居中」的那个框。
    const slackX = w - PAD_X * 2 - spanX * k;
    const slackY = h - PAD_TOP - PAD_BOTTOM - spanY * k;
    cam.cx += PAD_X + slackX / 2 - cam.cx - k * (x1 - cam.cx);
    cam.cy += PAD_TOP + slackY / 2 - cam.cy + k * (y2 - cam.cy);
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

  /** 落在 [lo, hi] 内的「好看」刻度：步长取 1/2/5×10^k。 */  function axisTicks(lo, hi, count = 4) {
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

  /** 画哪一张由「自回避对照」决定。
   *
   *  保留 draw() 这个名字做分派，是为了不动十几处调用点（视角拖拽、滚轮、
   *  键盘、显示开关、生长动画、resize……）—— 它们要的语义一直都是
   *  「按当前模式重画一遍」，只是以前只有一种模式。
   */
  function draw() {
    if (state.sawOn) { drawSawPair(); return; }
    drawModel();
  }

  /** 当前链模型的那条链（本文件的原有行为，一字未改）。 */
  function drawModel() {
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

  /* ---------------- 自回避对照 ----------------
   * 勾上「自回避对照」以后，这张画布不再画当前链模型的构象，改画**两团链**：
   * 同一个立方格点上的理想链与自回避行走，左右并排、共用一套相机与视角。
   *
   * 为什么值得单独做一块：两者唯一的差别就是「不许重复访问格点」这一条约束，
   * 剩下的采样方式、格点、步长、n、相机全都一样 —— 于是「自回避把链撑开了」
   * 这件事变成一个可以直接看出来的几何事实，而不是一句需要相信的话。
   *
   * 相机取**两团链的并集**为取景框：拿其中一团定框，另一团不是被裁掉，
   * 就是小得看不出差别。
   */

  /** 每边画几条。见下面 drawSawPair 里那段「为什么不是各画一条」。 */
  const SAW_PAIR_CHAINS = 20;

  async function fetchSaw() {
    const nRaw = $('c-n').value;
    const seedRaw = $('c-seed').value.trim();
    const l = Number($('in-l').value);
    if (!isFinite(l) || l <= 0) {
      setError('链段长度 l 必须是大于 0 的数字');
      return;
    }

    // n 夹到 SAW 的上限。**必须说出来**：输入框里可能写着 1000，画布上却是 30，
    // 不解释就成了「软件偷偷改了你的参数」。
    let n = nRaw === '' ? 20 : Math.floor(Number(nRaw));
    let clamped = false;
    if (!isFinite(n) || n < 1) { n = 20; }
    if (n > SAW_N_MAX) { n = SAW_N_MAX; clamped = true; }

    const my = ++state.sawSeq;
    $('chain-body').classList.add('busy');
    let data;
    try {
      const res = await fetch('/api/saw/chain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        // **不发链模型参数**：自回避行走不吃 model / theta / cosphi / p。
        // 每边要 SAW_PAIR_CHAINS 条叠着画（理由见 drawSawPair），但**只画这些** ——
        // 多要一批「只用来算平均、不画」的链会让画面上那两团和读数对不上。
        body: JSON.stringify({
          n, l, seed: seedRaw === '' ? null : Number(seedRaw), chains: SAW_PAIR_CHAINS,
        }),
      });
      data = await res.json().catch(() => ({ error: '服务器返回了无法解析的内容' }));
      if (!res.ok) throw new Error(data.error || `请求失败（HTTP ${res.status}）`);
    } catch (e) {
      if (my === state.sawSeq) {
        setError(e.message);
        $('chain-body').classList.remove('busy');
      }
      return;
    }
    if (my !== state.sawSeq) return;   // 已经有更新的请求了，丢弃这次

    if (seedRaw === '') $('c-seed').value = String(data.seed);
    state.saw = data;
    state.frac = 1;

    const warns = data.warnings.slice();
    if (clamped) {
      warns.unshift(
        `自回避一侧的 n 已按算力上限夹到 ${SAW_N_MAX}（你填的是 ${nRaw}）。`
        + '严格拒绝采样的存活率按 0.78^n 衰减，更大的 n 跑不完 —— 这是算力上限，不是物理上限。'
      );
    }
    setError(null, warns);
    renderSawCompare(data);
    $('chain-body').classList.remove('busy');
    draw();
  }

  /** 两个半宽视口里各画一条链。共用 cam 与 view，所以「谁更舒展」是眼睛能直接判的。 */
  function drawSawPair() {
    const cv = $('chain-canvas');
    if (!cv) return;
    if (!state.colors) readColors();
    const c = state.colors;
    const ctx = cv.getContext('2d');
    const w = cv.clientWidth;
    const h = cv.clientHeight;
    if (!w || !h) return;

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, cv.width, cv.height);
    const dpr = cv.width / w;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = c.surface;
    ctx.fillRect(0, 0, w, h);

    const s = state.saw;
    if (!s) {
      // 还没取到数（或取数失败，错误在告警条里）。画布不能留着上一条模型链 ——
      // 那会让人以为「这就是自回避」。
      ctx.font = FONT;
      ctx.fillStyle = c.muted;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('正在取自回避构象…', w / 2, h / 2);
      ctx.textAlign = 'left';
      return;
    }

    const n = s.n;
    const hw = w / 2;
    // 取景框取**两边画出来的全部链**的并集。只拿一条定框的话，另一条不是被裁掉
    // 就是小得看不出差别 —— 而「谁更舒展」正是这张图要说的事。
    const box = boundsOf(s.ideal_points.concat(s.points));
    // 原点也要在框里：两边都是从原点出发的，而 R 是相对原点量的，
    // 原点跑到框外就没法直观比较了。
    for (let k = 0; k < 3; k++) {
      box.min[k] = Math.min(box.min[k], 0);
      box.max[k] = Math.max(box.max[k], 0);
    }
    // 半宽视口：比例尺按半宽反算，再靠 ctx.translate 把两半摆到位。
    // 所以每个半边的相机中心在 hw/2，右半边整体平移 hw。
    const cam = makeCam(box, hw, h);
    // makeCam 把 3D 包围盒的**半对角线**塞进半边宽度 —— 对「一条链」够用，但这里
    // 的盒子是两团各 20 条的并集，投影出来只占半边的一小半，图会缩在中间。所以按
    // **投影后**的实际跨度再定标一次。两半共用这同一个 cam，所以「谁更舒展」
    // 仍然是一眼可比的 —— 重定标改的是取景，不是两边的相对尺度。
    fitScaleToProjection(cam, s.ideal_points.concat(s.points), hw, h);

    const nseg = state.frac >= 1
      ? n
      : Math.max(1, Math.round(state.frac * n));

    // 左：理想链（允许回头）—— 颜色沿用标度卡里「理想链 = series-1」那一套。
    // 右：自回避 —— series-0，和标度卡里的模拟点同一个色。
    //
    // 每边画 M 条**整体叠加**，不是各画一条。单条链的 R² 涨落比溶胀比本身还大
    // （n = 30 时 std(R²)/⟨R²⟩ ≈ 0.69，而溶胀才 2.0），画一条的话这对图有一小半
    // 时间会把结论反过来 —— 那是涨落，不是模型。叠成两团以后，两团的**尺度差**
    // 是稳的（要的就是这个），而 ⟨R²⟩ 报的仍然是 M 条的系综平均。
    const halves = [
      {
        flats: s.ideal_points,
        color: c.series[1],
        title: '理想链 · 允许回头',
        stat: `⟨R²⟩ = n·l² = ${fmt(s.ideal_R2_mean)}（解析）`,
        sub: `${s.chains} 条叠画`,
      },
      {
        flats: s.points,
        color: c.series[0],
        title: '自回避 · 不许重复访问格点',
        stat: `⟨R²⟩ = ${fmt(s.R2_mean)}（${s.chains} 条实测）`,
        sub: `溶胀 ×${s.swelling.toFixed(2)}`,
      },
    ];

    ctx.font = FONT;
    ctx.textBaseline = 'middle';

    halves.forEach((half, hi) => {
      const ox = hi * hw;
      ctx.save();
      ctx.translate(ox, 0);

      // 纵深域按**这一个半边**自己算：两半各自成立，否则一边的近处粗线
      // 会把另一边的链全压成同一个带。
      const dr = depthRange(cam, half.flats);
      let screen = null;
      half.flats.forEach((flat, k) => {
        const sc = drawChain(ctx, cam, flat, nseg, half.color, null, dr, 1, false, false);
        if (k === 0) screen = sc;
      });

      // 起止标记与 R 矢量只画**第一条**：M 组叠在一起就没法读了，
      // 而且它们说的是「一条链的起止」，不是系综量。
      const first = half.flats[0];
      const p0 = proj(cam, 0, 0, 0);
      const p1 = proj(cam,
        first[n * 3], first[n * 3 + 1], first[n * 3 + 2]);

      // R 矢量（虚线 = 参考量的语义，和别的图一致）
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

      // 起止标记：起点空心圆、终点实心方块（形状不同，不靠颜色区分）
      ctx.beginPath();
      ctx.arc(screen.px[0], screen.py[0], 5, 0, Math.PI * 2);
      ctx.fillStyle = c.surface;
      ctx.fill();
      ctx.strokeStyle = half.color;
      ctx.lineWidth = 2;
      ctx.stroke();

      if (state.frac >= 1) {
        ctx.beginPath();
        ctx.rect(p1.x - 4.5, p1.y - 4.5, 9, 9);
        ctx.fillStyle = half.color;
        ctx.fill();
        ctx.strokeStyle = c.surface;
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      // 两行读数排在这个半边的上沿 —— 坐标框不画，上面本来就空着。
      const cxm = hw / 2;
      ctx.textAlign = 'center';
      ctx.font = '13px system-ui, "Segoe UI", "Microsoft YaHei", sans-serif';
      ctx.fillStyle = half.color;
      ctx.fillText(half.title, cxm, 20);
      ctx.font = FONT;
      ctx.fillStyle = c.secondary;
      ctx.fillText(half.stat, cxm, 40);
      ctx.fillStyle = c.muted;
      ctx.fillText(half.sub, cxm, 58);
      ctx.textAlign = 'left';
      ctx.font = FONT;

      ctx.restore();
    });

    // 中间那道分隔线：两个视口是独立的，得让人看出来边界在哪
    ctx.strokeStyle = c.grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(hw, 8);
    ctx.lineTo(hw, h - 8);
    ctx.stroke();
  }

  /** 自回避模式的对照面板。键位与模型模式一致（compare-row / compare-item）。 */
  function renderSawCompare(s) {
    const box = $('chain-compare');
    box.textContent = '';
    if (!s) return;

    const row = document.createElement('div');
    row.className = 'compare-row';
    const item = (k, v) => {
      const wrap = document.createElement('div');
      wrap.className = 'compare-item';
      const a = document.createElement('span');
      a.className = 'k';
      a.textContent = k;
      const b = document.createElement('span');
      b.className = 'v';
      b.textContent = v;
      wrap.append(a, b);
      row.appendChild(wrap);
    };

    item('理想链 ⟨R²⟩ = n·l²（解析）', fmt(s.ideal_R2_mean));
    item(`自回避 ⟨R²⟩（${s.chains} 条实测）`, fmt(s.R2_mean));
    item('溶胀比', `×${s.swelling.toFixed(3)}`);
    item('n', String(s.n));
    item('l', String(s.l));
    item('种子', String(s.seed));
    item('试投 / 采到', `${s.trials.toLocaleString('en-US')} / ${s.chains.toLocaleString('en-US')}`);
    box.appendChild(row);

    const note = document.createElement('p');
    note.className = 'compare-note';
    note.textContent =
      `每边叠画 ${s.chains} 条独立实现。单条链的 R² 涨落比溶胀比本身还大`
      + '（n = 30 时 std(R²)/⟨R²⟩ ≈ 0.69），所以只画一条的话，这一对有一小半时间'
      + '会把结论反过来 —— 那是涨落不是模型；叠成两团看的才是两边的尺度差。'
      + `上面两个 ⟨R²⟩ 就是这 ${s.chains} 条的平均，条数更多、更准的那组在下面`
      + '「自回避行走」那张卡片里。'
      + '两团链在同一个立方格点上、同一个 n、同一个步长，共用一套相机与视角 ——'
      + '唯一的差别就是「不许重复访问格点」这一条约束，所以右边更舒展这件事'
      + '直接就是那条约束造成的（溶胀）。'
      + '自回避的 ⟨R²⟩ 恒大于 n·l²，这个比值随 n 缓慢上升（本机实测 n=5 约 1.45、n=30 约 2.0），'
      + '渐近地 ∝ n^(2ν)、2ν = 1.175194 —— 完整的标度关系见下面「自回避行走」那张卡片。';
    box.appendChild(note);
  }

  /** 切换自回避对照模式。 */
  function setSawMode(on) {
    state.sawOn = !!on;
    const csv = $('c-csv');
    if (csv) {
      // CSV 导的是**当前链模型**那条链的顶点。这个模式下画布上根本没有它，
      // 让人点下去导出一份「没画出来的东西」比置灰更糟。
      csv.disabled = state.sawOn;
      csv.title = state.sawOn
        ? '自回避对照画的是两团格点链，不是当前链模型的构象 —— 坐标 CSV 不适用（PNG 照常可用）'
        : '';
    }
    if (state.sawOn) {
      // 图例和对照面板本来描述的是当前模型那条链，现在没画它，先清掉，
      // 由 renderSawCompare 重填。
      const lg = $('chain-legend');
      if (lg) lg.hidden = true;
      fetchSaw();
    } else {
      state.saw = null;
      // 告警条里可能还挂着自回避那条「n 已夹到 30」—— 现在这个模式根本没在跑，
      // 留着会让人以为当前的模型链也被夹了。退回当前模型自己的告警。
      setError(null, state.data ? state.data.warnings : []);
      if (state.data) {
        renderLegend(state.data);
        renderCompare(state.data);
      }
      draw();
    }
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
    // 自回避模式下山 state.data 那侧可能还没取到过数，判断得看**当前模式**画的是什么
    if (state.sawOn ? !state.saw : !state.data) return;
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

  // 文件名里带上模型和链段长度：同一个 n、同一个种子，四个模型各导一份的话，
  // 光看 n 和种子分不出谁是谁；l 也写进去是因为 Kuhn 长度预置会把它填成 0.154
  // 这种值，没有 l 就复现不出来。（PNG 和 CSV 共用这个名字，两边一起受益。）
  // 自回避对照模式下画布上是那一对格点链，模型位写 saw。
  const stem = (d) => `fjc-chain-${state.sawOn ? 'saw' : (d.model || 'fjc')}-n${d.n}-l${d.l}-seed${d.seed}`;

  function exportPNG() {
    const cv = $('chain-canvas');
    const d = state.sawOn ? state.saw : state.data;
    if (!cv || !d) return;
    // 导出前确保画的是完整构象（动画中途导出会得到半条链）
    if (state.frac < 1) { state.frac = 1; draw(); }
    cv.toBlob((blob) => {
      if (blob) download(blob, `${stem(d)}.png`);
    }, 'image/png');
  }

  // 与 fjc_core.CHAIN_DECIMALS 对齐：坐标是服务端圆好再发过来的，这里统一再走一遍
  // toFixed，为的是把新算出来的几列（减出来的 dx、开方出来的 r）收拾干净 ——
  // 浮点减法会给出 0.30000000000000004 这种，不收拾就会顺着 CSV 漏出去。
  const CSV_DECIMALS = 4;
  const csvNum = (v) => (v === 0 ? 0 : v).toFixed(CSV_DECIMALS); // -0 也归成 0

  // 当前模型真正在用的参数有哪几个 —— 唯一真源是 /api/models 那份目录
  // （app.js 顶层的 MODEL_BY_KEY；经典脚本共享全局词法环境，直接按裸名取用），
  // 不在这里另抄一份「哪个模型有哪些参数」的映射，否则改一处就会忘一处。
  // d.params 是四个模型都齐的（没用的那几个也带着默认值），所以必须问目录，
  // 不能只看见字段就写。
  function modelParamFields(d) {
    const spec = typeof MODEL_BY_KEY === 'undefined' ? null : MODEL_BY_KEY.get(d.model);
    if (!spec || !d.params) return [];
    return ['theta_deg', 'cos_phi', 'p']
      .filter((id) => spec.params[id] && spec.params[id].used && d.params[id] != null)
      .map((id) => `${id}=${d.params[id]}`);
  }

  function exportCSV() {
    const d = state.data;
    if (!d) return;
    // 自回避对照模式下画布上是那对格点链，这份 CSV 描述的是**当前链模型**那条链。
    // 按钮已经置灰，这里再挡一道，免得将来有别的入口调进来导出一份「没画出来的东西」。
    if (state.sawOn) return;

    // 注释头是**纯 ASCII**：这个文件按既定选择不加 BOM（加了反而会在某些解析器里
    // 多出一个字符），而不带 BOM 的 UTF-8 中文在 Excel 里会按本地代码页解成乱码，
    // 所以这里不写中文模型名 —— 模型 key（fjc / frc / hindered / wlc）本身就是 ASCII，
    // 自描述够用了。
    const head = [
      `model=${d.model}`, `n=${d.n}`, `l=${d.l}`, `seed=${d.seed}`,
      `chains=${d.chains}`, `b=${d.kuhn_length}`,
    ].concat(modelParamFields(d));
    const lines = [
      '# chain conformation vertex coordinates',
      `# ${head.join(' ')}`,
      '# columns: dx/dy/dz = incoming bond vector (blank on segment 0);'
        + ' s = contour length i*l; r = distance from origin',
      // 前五列的名字和顺序与以前一致，后四组是追加的 —— 已有的读取脚本不会坏。
      'chain,segment,x,y,z,dx,dy,dz,s,r',
    ];

    d.points.forEach((flat, ci) => {
      for (let i = 0; i <= d.n; i++) {
        const x = flat[i * 3], y = flat[i * 3 + 1], z = flat[i * 3 + 2];
        // dx/dy/dz 是**到达这个顶点的**那一步。第 0 个顶点是原点、没有入边，留空 ——
        // 空字段读进 pandas 正好是 NaN，语义就是「没有定义」，比填 0 诚实。
        const bond = i === 0
          ? ['', '', '']
          : [csvNum(x - flat[(i - 1) * 3]),
             csvNum(y - flat[(i - 1) * 3 + 1]),
             csvNum(z - flat[(i - 1) * 3 + 2])];
        // s 直接乘 i 而不是逐段累加：四个模型的每段长度都恰好是 l，乘出来没有累加误差。
        // r 用小写，和整链那个末端矢量 R / |R| 区分开。
        lines.push([ci, i, csvNum(x), csvNum(y), csvNum(z)].concat(bond,
          [csvNum(i * d.l), csvNum(Math.hypot(x, y, z))]).join(','));
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

    // 改参数后取哪一路的数，由当前模式决定：自回避对照模式下画布上根本没有
    // 当前链模型那条链，再去 /api/chain 取一次纯属浪费。
    const schedule = () => {
      clearTimeout(fetchTimer);
      const go = state.sawOn ? fetchSaw : fetchChain;
      fetchTimer = setTimeout(go, 180);
    };
    // 链模型和它的三个参数**只影响**模型那一侧，自回避一侧不吃这些，所以不重取。
    const scheduleModel = () => { if (!state.sawOn) schedule(); };

    $('c-n').addEventListener('input', schedule);
    $('c-chains').addEventListener('input', schedule);
    $('c-seed').addEventListener('input', schedule);
    // l 与上方分布图共用，改 l 时这张图也要跟着变
    $('in-l').addEventListener('input', schedule);
    // 链模型和它的三个参数都会改变采样出来的构象：换模型要重取一次，
    // 否则图上还是上一条链，看着像「切了模型但 3D 没动」。
    // **in-model 不在这里** —— 它由下面 onModel 一个人管（模式可能会跟着换），
    // 两个监听器都排一次队就会有两次请求。
    ['in-theta', 'in-cosphi', 'in-p'].forEach((id) => {
      const el = $(id);
      if (!el) return;
      el.addEventListener('change', scheduleModel);
      el.addEventListener('input', scheduleModel);
    });

    /* 模式跟着**模型下拉**走：选「自回避行走（SAW）」时这张画布画那两团格点链，
       选四个解析模型时画当前模型的构象。
       这里原来挂着一个独立的「自回避对照」复选框，删掉了 —— 两个入口各说各话
       （选了蠕虫状链还能勾「自回避对照」，画布上画的和顶部参数说的不是一回事）。
       判据取目录里的 analytic（见 fjc_core.model_catalog），不在前端另写一份
       「哪个模型是 SAW」—— 那正是那份目录要消掉的东西。

       `fjc:models-ready` 每次加载都会来一次，而那之后这里会无条件重取一遍 ——
       也就是加载时多打一次 /api/chain。**这是有意的**：目录是异步到的，它到之前
       画布上那条链可能是按 HTML 里的默认模型采的，而用户存的模型（URL 里那个）
       要等目录回来才知道。多打一次本地请求换取「画的永远是当前模型那条链」，
       比反过来强。 */
    const onModel = () => {
      const spec = window.fjcApp ? window.fjcApp.modelSpec() : null;
      if (!spec) return;                       // 目录还没回来，等它
      setSawMode(!spec.analytic);
      if (spec.analytic) schedule();           // 解析模型：重取这个模型的构象
    };
    $('in-model').addEventListener('change', onModel);
    document.addEventListener('fjc:models-ready', onModel);

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
      // 自回避对照下每半边只有一条链，没有「选哪条」这回事
      if (state.sawOn) return;
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
      (state.sawOn ? fetchSaw : fetchChain)();
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
        // 自回避对照开着时，下面那几个读数说的是**模型那一侧**（可能已经过期，
        // 也可能压根没取过），而画布上是那对格点链 —— 助手必须看得见这件事。
        saw: state.sawOn,
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
      // 自己 await 一次取数 —— 否则这里没法知道它什么时候画完。
      touched.forEach((id) => $(id).dispatchEvent(new Event('input', { bubbles: true })));
      clearTimeout(fetchTimer);
      await (state.sawOn ? fetchSaw : fetchChain)();
      if (state.sawOn ? !state.saw : !state.data) {
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
