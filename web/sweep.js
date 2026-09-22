'use strict';

/* 链长扫描 —— 前端
 *
 * 和 app.js / chain3d.js 一样：这里**不做任何物理计算**。⟨R²⟩、n·l²、相对标准误
 * rel_se、以及 log⟨R²⟩ 对 log n 的拟合斜率，全部由 /api/sweep 算好返回，
 * 前端只做坐标变换和画图。
 *
 * 这张图是双对数坐标：理论值 n·l² 在双对数下恰好是一条斜率 1 的直线，
 * 所以「模拟点是不是落在这条线上、拟合斜率是不是 1」就是 ⟨h²⟩ = n·l² 的数值证据。
 * 点周围那圈 ±2σ 带不是误差棒，是 M 条链的**有限样本涨落**该有多大 ——
 * 点落在带里说明采样器忠实于模型，落在带外才值得追。
 *
 * 复用了 app.js 里的 $ / el / fmt / fmtTick / seriesVar / renderAlerts，
 * 那几个都是顶层声明，加载顺序保证本文件执行时已经就绪。
 */

(function () {

  // 画布尺寸（viewBox 坐标）。比 P(h) 那张宽，因为这张卡片是通栏的；
  // 字号、描边沿用 style.css 里同一套图表类，不另起一套。
  // right 留得宽，是为了把理论线的标注排在绘图区**右侧外面** —— 放在里面
  // 无论贴哪一端都会压住 n 最大的那个点（两条线在那儿几乎重合）。
  const SCH = { w: 1240, h: 500, top: 44, right: 156, bottom: 58, left: 88 };
  const SPW = SCH.w - SCH.left - SCH.right;
  const SPH = SCH.h - SCH.top - SCH.bottom;

  // 「等比序列」填的 15 个 n：10…1000，14 步跨两个数量级，每步 ×10^(1/7)
  const GEO = [10, 14, 19, 27, 37, 52, 72, 100, 139, 193, 268, 372, 517, 718, 1000];

  // 一次最多扫几个 n。与 fjc_core.SWEEP_N_MAX 一致 —— 后端还会再校验一次，
  // 这里只是为了在发请求之前就给一句中文提示。
  const N_MAX = 20;

  // 悬停命中半径（屏幕像素，换算到 viewBox 单位时除以缩放比）
  const HIT_PX = 22;

  const state = {
    data: null,      // /api/sweep 的返回，原样存着（导出 JSON 时直接用）
    ctx: null,       // 坐标映射，供悬停复用
    hoverIdx: null,
    ms: null,        // 一次往返的墙钟耗时（含网络与后端）
  };

  /* ---------------- 坐标与刻度 ---------------- */

  /**
   * 对数轴刻度。三种档位：
   *   major 10^k   画满格网格线 + 刻度值
   *   grid  2/5×10^k 只画网格线（对数轴的通行约定，不标值也不至于误读）
   *   minor 其余   只在轴边点一小横
   * 跨度超过三个数量级时一档十年就够密了，再细分反而糊成一片。
   */
  function logTicks(lo, hi) {
    const mults = hi - lo > 3
      ? [[1, 'major']]
      : [[1, 'major'], [2, 'grid'], [5, 'grid'],
         [3, 'minor'], [4, 'minor'], [6, 'minor'], [7, 'minor'], [8, 'minor'], [9, 'minor']];
    const out = [];
    for (let k = Math.floor(lo); k <= Math.ceil(hi); k++) {
      mults.forEach(([m, kind]) => {
        const v = m * Math.pow(10, k);
        const lg = Math.log10(v);
        if (lg >= lo - 1e-9 && lg <= hi + 1e-9) out.push({ v, kind });
      });
    }
    return out;
  }

  /* ---------------- 画图 ---------------- */

  function drawEmpty(svg) {
    el('text', {
      class: 'axis-title', x: SCH.w / 2, y: SCH.h / 2 - 6, 'text-anchor': 'middle',
    }, svg).textContent = '⟨R²⟩ 对 n 的标度关系';
    el('text', {
      class: 'tick-text', x: SCH.w / 2, y: SCH.h / 2 + 20, 'text-anchor': 'middle',
    }, svg).textContent = '点「运行模拟」开始。默认 6 个 n、每个 10000 条链，要算几秒。';
  }

  function render() {
    const svg = $('sweep-chart');
    svg.textContent = '';
    state.ctx = null;

    const d = state.data;
    if (!d) { drawEmpty(svg); return; }

    const l = d.l;
    const f = 2 * d.rel_se;      // σ 带的半宽（相对值）

    // 按 n 升序 —— 用户可能把 n 填成倒序，连线和悬停都得按顺序走
    const pts = d.n.map((n, i) => ({
      n, y: d.R2_mean[i], t: d.R2_theory[i], dev: d.rel_dev[i], cn: d.Cn_sim[i],
    })).sort((a, b) => a.n - b.n);

    // --- 双对数坐标域 ---
    const x0 = Math.log10(pts[0].n);
    const x1 = Math.log10(pts[pts.length - 1].n);
    // 只给一个 n 时跨度为 0，给一个固定的半格，否则除零
    const xPad = pts.length > 1 ? 0.04 * (x1 - x0) : 0.35;
    const xlo = x0 - xPad;
    const xhi = x1 + xPad;

    // ±2σ 带也要纳入纵轴范围，否则带子会被画到框外
    const bandLo = pts.map((p) => p.t * (1 - f));
    const bandHi = pts.map((p) => p.t * (1 + f));
    const yAll = [...pts.map((p) => p.y), ...pts.map((p) => p.t), ...bandLo, ...bandHi];
    const y0 = Math.log10(Math.min(...yAll));
    const y1 = Math.log10(Math.max(...yAll));
    const yPad = Math.max(0.08 * (y1 - y0), 0.05);
    const ylo = y0 - yPad;
    const yhi = y1 + yPad;

    const sx = (v) => SCH.left + ((Math.log10(v) - xlo) / (xhi - xlo)) * SPW;
    const sy = (v) => SCH.top + SPH - ((Math.log10(v) - ylo) / (yhi - ylo)) * SPH;

    // 直线/带子画到绘图区左右边缘，而不是止于最外侧的点 —— 看起来才像一条"律"
    const nA = Math.pow(10, xlo);
    const nB = Math.pow(10, xhi);
    const tA = nA * l * l;
    const tB = nB * l * l;
    const fitY = d.slope === null ? null
      : (n) => Math.exp(d.intercept) * Math.pow(n, d.slope);

    // --- 网格与刻度 ---
    // 纵轴 major/grid 都拉满格网格线，minor 只在轴边点一小横；横轴同理。
    // 两条护栏：
    //  1) 贴着边框的那一两个刻度（坐标域边缘刚好切进 4、9 这种）离轴线不到一个像素，
    //     画出来是噪点而不是刻度，所以只留白不留痕 —— 但**标了值的**不吃这道护栏，
    //     宁可让一条网格线和轴线重合（画在轴线底下，看不出来），也不能让轴上的数字消失。
    //  2) 逐级放宽「哪一档可以标值」：正常是 10^k；坐标域正好落在两个 10^k 之间时
    //     （比如 n 只填 300 和 400）连 2/5 都不在域内，一直放宽到细分，总要让轴上有数可读。
    const INSET = 7;
    const labelledKinds = (ticks) => {
      const sets = [['major'], ['major', 'grid'], ['major', 'grid', 'minor']];
      return sets.find((s) => ticks.some((t) => s.includes(t.kind))) || sets[0];
    };
    const grid = el('g', {}, svg);

    const yts = logTicks(ylo, yhi);
    const yKinds = labelledKinds(yts);
    yts.forEach((t) => {
      const y = sy(t.v);
      const labelled = yKinds.includes(t.kind);
      if (!labelled && (y < SCH.top + INSET || y > SCH.top + SPH - INSET)) return;
      if (t.kind !== 'minor') {
        el('line', {
          class: 'grid-line', x1: SCH.left, x2: SCH.left + SPW, y1: y, y2: y,
        }, grid);
      } else {
        el('line', {
          class: 'grid-line', x1: SCH.left - 4, x2: SCH.left, y1: y, y2: y,
        }, grid);
      }
      if (labelled) {
        el('text', {
          class: 'tick-text', x: SCH.left - 10, y: y + 4, 'text-anchor': 'end',
        }, grid).textContent = fmtTick(t.v, false);
      }
    });

    const xts = logTicks(xlo, xhi);
    const xKinds = labelledKinds(xts);
    xts.forEach((t) => {
      const x = sx(t.v);
      const labelled = xKinds.includes(t.kind);
      if (!labelled && (x < SCH.left + INSET || x > SCH.left + SPW - INSET)) return;
      if (t.kind !== 'major') {
        // 2/5 那两档画长一点的刻度线，和其余细分区分开
        const len = t.kind === 'grid' ? 6 : 4;
        el('line', {
          class: 'grid-line', x1: x, x2: x, y1: SCH.top + SPH, y2: SCH.top + SPH + len,
        }, grid);
      }
      if (labelled) {
        el('text', {
          class: 'tick-text', x, y: SCH.top + SPH + 20, 'text-anchor': 'middle',
        }, grid).textContent = fmtTick(t.v, false);
      }
    });

    // --- 坐标轴 ---
    el('line', {
      class: 'axis-line', x1: SCH.left, x2: SCH.left + SPW, y1: SCH.top + SPH, y2: SCH.top + SPH,
    }, svg);
    el('line', {
      class: 'axis-line', x1: SCH.left, x2: SCH.left, y1: SCH.top, y2: SCH.top + SPH,
    }, svg);
    el('text', {
      class: 'axis-title', x: SCH.left + SPW / 2, y: SCH.h - 14, 'text-anchor': 'middle',
    }, svg).textContent = '链段数 n（对数）';
    el('text', {
      class: 'axis-title', x: 0, y: 0, 'text-anchor': 'middle',
      transform: `translate(18, ${SCH.top + SPH / 2}) rotate(-90)`,
    }, svg).textContent = '⟨R²⟩（对数）';

    // --- ±2σ 涨落带（区域，不是数据点，所以用低透明度填充）---
    if (f > 0) {
      el('path', {
        class: 'sweep-band',
        d: `M${sx(nA).toFixed(2)},${sy(tA * (1 + f)).toFixed(2)}`
          + `L${sx(nB).toFixed(2)},${sy(tB * (1 + f)).toFixed(2)}`
          + `L${sx(nB).toFixed(2)},${sy(tB * (1 - f)).toFixed(2)}`
          + `L${sx(nA).toFixed(2)},${sy(tA * (1 - f)).toFixed(2)}Z`,
      }, svg);
    }

    // --- 理论线 n·l² ---
    el('line', {
      class: 'series-line', stroke: seriesVar(1),
      x1: sx(nA), y1: sy(tA), x2: sx(nB), y2: sy(tB),
    }, svg);

    // --- 模拟点（连线用细一点才不至于盖住理论线）---
    if (pts.length > 1) {
      el('path', {
        class: 'series-line thin', stroke: seriesVar(0),
        d: pts.map((p, i) => `${i ? 'L' : 'M'}${sx(p.n).toFixed(2)},${sy(p.y).toFixed(2)}`).join(''),
      }, svg);
    }
    const dotG = el('g', {}, svg);
    pts.forEach((p) => {
      // 8px 以上（r=5），加 2px 表面色圆环和理论线分开
      el('circle', {
        class: 'series-dot', cx: sx(p.n), cy: sy(p.y), r: 5, fill: seriesVar(0),
      }, dotG);
    });

    // --- 拟合直线（虚线 = 推导值，和阈值线同一套语义）---
    if (fitY) {
      el('line', {
        class: 'fit-line',
        x1: sx(nA), y1: sy(fitY(nA)), x2: sx(nB), y2: sy(fitY(nB)),
      }, svg);
    }

    // --- 直接标注 ---
    // 两条线在双对数下几乎完全重合（这正是结论），所以没法各贴各的 ——
    // 贴在一起会叠字。标注只给理论线下一个，拟合线的斜率交给图例和汇总行。
    el('text', {
      class: 'sweep-label', x: SCH.left + SPW + 12, y: sy(tB) + 4, 'text-anchor': 'start',
    }, svg).textContent = '理论 n·l²';

    // --- 悬停层 ---
    const hg = el('g', {}, svg);
    el('circle', {
      id: 'sweep-hover-dot', class: 'series-dot', r: 7, cx: -99, cy: -99,
      fill: seriesVar(0), opacity: 0,
    }, hg);

    state.ctx = { pts, sx, sy };
    if (state.hoverIdx !== null) updateHover(Math.min(state.hoverIdx, pts.length - 1));
  }

  /* ---------------- 悬停读数 ---------------- */

  function vbPoint(ev) {
    const svg = $('sweep-chart');
    const r = svg.getBoundingClientRect();
    const k = r.width / SCH.w;
    return { x: (ev.clientX - r.left) / k, y: (ev.clientY - r.top) / k, k, rect: r };
  }

  function updateHover(i) {
    const ctx = state.ctx;
    const dot = $('sweep-hover-dot');
    const tt = $('sweep-tooltip');
    if (!ctx || !dot || !tt) return;

    if (i === null || i === undefined) {
      dot.setAttribute('opacity', '0');
      tt.hidden = true;
      return;
    }

    const p = ctx.pts[i];
    const cx = ctx.sx(p.n);
    const cy = ctx.sy(p.y);
    dot.setAttribute('cx', cx);
    dot.setAttribute('cy', cy);
    dot.setAttribute('opacity', '1');

    const svg = $('sweep-chart');
    const rect = svg.getBoundingClientRect();
    const k = rect.width / SCH.w;

    tt.textContent = '';
    const head = document.createElement('div');
    head.className = 'tt-head';
    head.textContent = `n = ${p.n}`;
    tt.appendChild(head);

    const row = (name, val, color) => {
      const d = document.createElement('div');
      d.className = 'tt-row';
      if (color) {
        const key = document.createElement('span');
        key.className = 'tt-key';
        key.style.background = color;
        d.appendChild(key);
      }
      const a = document.createElement('span');
      a.className = 'tt-name';
      a.textContent = name;
      const b = document.createElement('span');
      b.className = 'tt-val';
      b.textContent = val;
      d.append(a, b);
      tt.appendChild(d);
    };

    row('模拟 ⟨R²⟩', fmt(p.y), seriesVar(0));
    row('理论 n·l²', fmt(p.t), seriesVar(1));
    row('相对偏差', `${p.dev >= 0 ? '+' : ''}${(p.dev * 100).toFixed(2)}%`);
    row('特征比 Cn', fmt(p.cn));
    row('h_rms 模拟', fmt(Math.sqrt(p.y)));

    tt.hidden = false;
    const px = cx * k;
    const py = cy * k;
    const ttW = tt.offsetWidth;
    const left = px + 16 + ttW > rect.width ? px - 16 - ttW : px + 16;
    tt.style.left = `${Math.max(0, left)}px`;
    tt.style.top = `${Math.min(Math.max(0, py - 12), Math.max(0, rect.height - tt.offsetHeight))}px`;
  }

  /* ---------------- 汇总行与图例 ---------------- */

  function h(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  }

  function renderLegend() {
    const box = $('sweep-legend');
    box.textContent = '';
    if (!state.data) { box.hidden = true; return; }
    box.hidden = false;

    const d = state.data;
    // 拟合线的斜率写进图例：它和理论线在图上几乎重合，贴不了直接标注
    const fitText = d.slope === null ? '拟合直线（无）' : `拟合直线 斜率 ${d.slope.toFixed(4)}`;

    [
      { k: 'dot', color: seriesVar(0), text: '模拟 ⟨R²⟩' },
      { k: 'line', color: seriesVar(1), text: '理论 n·l²' },
      { k: 'dash', color: 'var(--text-secondary)', text: fitText },
      { k: 'band', color: seriesVar(1), text: '±2σ 涨落带', fade: true },
    ].forEach((it) => {
      const item = h('span', 'legend-item');
      const key = h('span', `legend-key ${it.k}`);
      key.style.background = it.color;
      if (it.fade) key.style.opacity = '0.3';
      const label = h('span', null, it.text);
      item.append(key, label);
      box.appendChild(item);
    });
  }

  function renderSummary() {
    const box = $('sweep-summary');
    box.textContent = '';
    const d = state.data;
    if (!d) return;

    const devs = d.rel_dev.map((v) => Math.abs(v));
    const worst = Math.max(...devs);
    const worstAt = d.n[devs.indexOf(worst)];

    const row = h('div', 'compare-row');
    const item = (k, v) => {
      const wrap = h('div', 'compare-item');
      wrap.append(h('span', 'k', k), h('span', 'v', v));
      row.appendChild(wrap);
    };

    if (d.slope === null) {
      item('拟合斜率', '–（只有一个 n）');
    } else {
      item('log⟨R²⟩ 对 log n 的拟合斜率', d.slope.toFixed(4));
      item('该拟合的决定系数 R²', d.fit_r2.toFixed(5));
      // 斜率的标准误比 rel_se 小 —— 最小二乘把多个点的误差平均掉了，
      // n 排得越开小得越多。后端判「斜率是不是 1」用的就是它。
      if (typeof d.slope_se === 'number') {
        item('该斜率的标准误', `±${d.slope_se.toFixed(4)}`);
      }
    }
    item('最大 |相对偏差|', `${(worst * 100).toFixed(2)}%（n = ${worstAt}）`);
    item('⟨R²⟩ 的相对标准误', `${(d.rel_se * 100).toFixed(2)}%`);
    if (state.ms !== null) {
      item('本次耗时', state.ms < 1000
        ? `${Math.round(state.ms)} ms`
        : `${(state.ms / 1000).toFixed(1)} s`);
    }
    box.appendChild(row);

    const note = h('p', 'compare-note');
    note.textContent = d.slope === null
      ? `只给了一个 n，拟合不出斜率。至少要两个不同的 n 才能看出 ⟨R²⟩ 随 n 怎么长。`
      : `每个 n 采了 ${d.chains.toLocaleString('en-US')} 条独立链，所以模拟点天然会在理论线周围`
        + `上下跳动 —— 图上的 ±2σ 带就是这个跳动该有的范围（√(2/3)/√M = ${(d.rel_se * 100).toFixed(2)}%），`
        + `点落在带里属正常，落在带外才值得追。斜率偏离 1 超过它自身标准误的三倍时，后端会一并报出来。`;
    box.appendChild(note);
  }

  /** 后端给的警告，外加「参数改了但图还没重扫」这条本地提示。 */
  function refreshAlerts() {
    const d = state.data;
    const warns = d ? [...d.warnings] : [];
    if (d) {
      const now = Number($('in-l').value);
      if (isFinite(now) && now !== d.l) {
        warns.unshift(
          `链段长度 l 现在是 ${now}，这张图是 l = ${d.l} 时扫的 —— 点「运行模拟」重扫一遍。`
        );
      }
    }
    renderAlerts('sweep-alerts', null, warns);
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

  const stem = (d) => `fjc-sweep-${d.n.length}n-M${d.chains}-seed${d.seed}`;

  function exportCSV() {
    const d = state.data;
    if (!d) return;
    // 汇总表：每个 n 一行，全精度。纯 ASCII 表头、不带 BOM（同 chain3d.js）
    const lines = ['n,R2_sim,R2_theory,rel_dev,Cn,h_rms_sim,h_rms'];
    d.n.forEach((n, i) => {
      lines.push([
        n, d.R2_mean[i], d.R2_theory[i], d.rel_dev[i], d.Cn_sim[i], d.h_rms_sim[i], d.h_rms[i],
      ].join(','));
    });
    download(
      new Blob([lines.join('\r\n')], { type: 'text/csv;charset=utf-8' }),
      `${stem(d)}.csv`
    );
  }

  function exportJSON() {
    const d = state.data;
    if (!d) return;
    const out = {
      tool: '自由连接链（FJC）计算器 —— 链长扫描',
      generated: new Date().toISOString(),
      request: { n: d.n, l: d.l, seed: d.seed, chains: d.chains },
      fit: d.slope === null ? null : {
        slope: d.slope,
        intercept: d.intercept,
        r2: d.fit_r2,
        slope_se: d.slope_se,
        note: 'log⟨R²⟩ 对 log n 的最小二乘拟合。理论值：斜率 1。'
          + 'slope_se 是斜率自身的标准误，n 排得越开越小；判偏离用它。',
      },
      result: d,
    };
    download(
      new Blob([JSON.stringify(out, null, 2)], { type: 'application/json;charset=utf-8' }),
      `${stem(d)}.json`
    );
  }

  /* ---------------- 运行 ---------------- */

  function parseNs(raw) {
    const parts = String(raw).split(/[,，、;；\s]+/).filter((s) => s.length > 0);
    if (parts.length === 0) throw new Error('请输入链段数 n');
    if (parts.length > N_MAX) throw new Error(`一次最多扫 ${N_MAX} 个 n，收到 ${parts.length} 个`);
    const out = parts.map((p) => {
      const v = Number(p);
      if (!isFinite(v)) throw new Error(`「${p}」不是数字`);
      return v;
    });
    if (new Set(out).size !== out.length) throw new Error('链段数 n 有重复，请去掉重复值');
    return out;
  }

  async function run() {
    let ns;
    let l;
    let chains;
    let seed;
    try {
      ns = parseNs($('s-n').value);
      l = Number($('in-l').value);          // l 与顶部参数行共用，和 3D 视图一致
      if (!isFinite(l)) throw new Error('链段长度 l 必须是数字');
      const rawM = $('s-chains').value.trim();
      chains = rawM === '' ? undefined : Number(rawM);
      if (chains !== undefined && !isFinite(chains)) throw new Error('每个 n 的链数 M 必须是整数');
      const rawSeed = $('s-seed').value.trim();
      seed = rawSeed === '' ? null : Number(rawSeed);
      if (seed !== null && (!isFinite(seed) || seed < 0)) throw new Error('随机种子必须是非负整数');
    } catch (e) {
      renderAlerts('sweep-alerts', e.message, []);
      return;
    }

    $('sweep-body').classList.add('busy');
    $('s-run').disabled = true;
    $('s-csv').disabled = true;
    $('s-json').disabled = true;
    $('s-png').disabled = true;
    renderAlerts('sweep-alerts', null, []);

    const t0 = performance.now();
    try {
      const res = await fetch('/api/sweep', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ n: ns, l, seed, chains }),
      });
      const data = await res.json().catch(() => ({ error: '服务器返回了无法解析的内容' }));
      if (!res.ok) throw new Error(data.error || `请求失败（HTTP ${res.status}）`);

      state.data = data;
      state.ms = performance.now() - t0;
      state.hoverIdx = null;
      $('s-seed').value = String(data.seed);   // 后端回传的种子，写回去即可复现

      render();
      renderLegend();
      renderSummary();
      refreshAlerts();
      $('s-csv').disabled = false;
      $('s-json').disabled = false;
      $('s-png').disabled = false;
    } catch (e) {
      renderAlerts('sweep-alerts', e.message, []);
    } finally {
      $('sweep-body').classList.remove('busy');
      $('s-run').disabled = false;
    }
  }

  /* ---------------- 事件 ---------------- */

  function initEvents() {
    $('s-run').addEventListener('click', run);
    $('s-geo').addEventListener('click', () => {
      $('s-n').value = GEO.join(', ');
      $('s-chains').value = '10000';
      run();
    });
    ['s-n', 's-chains', 's-seed'].forEach((id) => {
      $(id).addEventListener('keydown', (ev) => {
        if (ev.key === 'Enter') { ev.preventDefault(); run(); }
      });
    });
    $('s-csv').addEventListener('click', exportCSV);
    $('s-json').addEventListener('click', exportJSON);
    // PNG 走 app.js 那份共用的 SVG→PNG（序列化 + 样式内联 + 垫底色），
    // 三张图各写一份的话，导出的样子迟早不一样。
    $('s-png').addEventListener('click', () => {
      if (!state.data) return;
      window.fjcExport.svgToPng($('sweep-chart'), `${stem(state.data)}.png`)
        .catch((e) => window.fjcExport.pngFail('sweep-alerts', e));
    });

    // l 改了：图没变，但要立刻把「该重扫了」说出来
    $('in-l').addEventListener('input', refreshAlerts);

    const svg = $('sweep-chart');
    svg.addEventListener('pointermove', (ev) => {
      if (!state.ctx) return;
      const { x, y, k } = vbPoint(ev);
      const lim = HIT_PX / k;
      let best = null;
      let bestD = lim * lim;
      state.ctx.pts.forEach((p, i) => {
        const dx = state.ctx.sx(p.n) - x;
        const dy = state.ctx.sy(p.y) - y;
        const dd = dx * dx + dy * dy;
        if (dd < bestD) { bestD = dd; best = i; }
      });
      if (best !== state.hoverIdx) {
        state.hoverIdx = best;
        updateHover(best);
      }
    });
    svg.addEventListener('pointerleave', () => {
      state.hoverIdx = null;
      updateHover(null);
    });

    // 键盘：←/→ 在扫描点之间走，Esc 收起。和 P(h) 那张图的键盘支持一个路子。
    svg.addEventListener('keydown', (ev) => {
      const pts = state.ctx ? state.ctx.pts : null;
      let used = true;
      switch (ev.key) {
        case 'ArrowLeft':
          if (!pts) { used = false; break; }
          state.hoverIdx = state.hoverIdx === null
            ? pts.length - 1 : Math.max(0, state.hoverIdx - 1);
          break;
        case 'ArrowRight':
          if (!pts) { used = false; break; }
          state.hoverIdx = state.hoverIdx === null
            ? 0 : Math.min(pts.length - 1, state.hoverIdx + 1);
          break;
        case 'Escape':
          state.hoverIdx = null;
          updateHover(null);
          return;
        default:
          used = false;
      }
      if (used) { ev.preventDefault(); updateHover(state.hoverIdx); }
    });
  }

  /* ---------------- 启动 ---------------- */

  // 深浅色不用管：这张图是 SVG，fill/stroke 都是 CSS 变量，主题一变自动跟上
  // （P(h) 那张要重绘是因为 canvas 解析不了 CSS 变量，这里没这个问题）。
  initEvents();
  render();
})();
