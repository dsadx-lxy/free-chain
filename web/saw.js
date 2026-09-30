'use strict';

/* 自回避行走（SAW）—— 前端
 *
 * 和 app.js / sweep.js 一样：这里**不做任何物理计算**。⟨R²⟩、理想链的 n·l²、
 * 每个 n 的实测相对标准误 rel_se、拟合斜率与它的标准误、两条参考线的截距，
 * 全部由 /api/saw/sweep 算好返回，前端只做坐标变换和画图。
 *
 * 这张图回答的问题和「链长扫描」那张不一样：
 *   · 链长扫描问「⟨R²⟩ 是不是正比于 n」—— 那四个解析模型的答案都是「是」，
 *     所以图上有一条斜率 1 的理论线，模拟点应当落在它上面。
 *   · 这一张问「自回避行走的 ⟨R²⟩ 是不是**仍然**正比于 n」—— 答案是否定的。
 *     ⟨R²⟩ ∝ n^(2ν)、2ν = 1.175194，所以点会系统性地跑到斜率 1 的上方，
 *     而且**越大的 n 跑得越远**（溶胀随链长增强）。
 *
 * 两条参考线都**锚在数据质心**上，不引任何无法自证的振幅 —— 后端只回截距，
 * 前端不自己拟合、也不自己算振幅。
 *
 * 一个如实交代：拟合斜率（约 1.19–1.22）和 2ν 参考线（1.1752）只差百分之一点几，
 * 在双对数图上**几乎重合**，肉眼看不出分开。这不是画得不好，恰恰是「n ≤ 30 还远没到
 * 渐近区」这件事的可视化。判据是上面那个 σ 数，不是两条线分不分得开。
 *
 * 复用了 app.js 里的 $ / el / fmt / fmtTick / seriesVar / renderAlerts，
 * 以及 sweep.js 末尾挂出来的 window.fjcSweepHelpers（两个纯刻度函数）——
 * 这是张**同规格**的双对数图，刻度逻辑不该有第二个版本。
 */

(function () {

  // 画布尺寸与「链长扫描」那张**完全一致**（同一个 viewBox、同一套留白）。
  // 两张卡片上下相邻，规格一致才不会有「同一张纸上的两张图长得不一样」的感觉。
  const SCH = { w: 1240, h: 500, top: 44, right: 156, bottom: 58, left: 88 };
  const SPW = SCH.w - SCH.left - SCH.right;
  const SPH = SCH.h - SCH.top - SCH.bottom;

  // 与 fjc_core.SAW_N_MAX 一致。后端还会再校验一次（超了返 400），
  // 这里先拦一道只是为了在发请求之前就给一句中文提示。
  const N_MAX = 30;

  // 悬停命中半径（屏幕像素，换算到 viewBox 单位时除以缩放比）
  const HIT_PX = 22;

  // 「小 n 大样本」预置：短链便宜得多，同样的预算下能把统计误差压小一个数量级。
  // n = 4/6/8/10 各 20000 条链的估算试投量约 3.1×10⁵，在预算（1.5×10⁶）之内。
  const SMALL = { n: '4, 6, 8, 10', m: '20000' };

  const { logTicks, labelledKinds } = window.fjcSweepHelpers;

  const state = {
    data: null,      // /api/saw/sweep 的返回，原样存着（导出 JSON 时直接用）
    ctx: null,       // 坐标映射，供悬停复用
    hoverIdx: null,
    ms: null,        // 一次往返的墙钟耗时（含网络与后端）
  };

  /* ---------------- 画图 ---------------- */

  function drawEmpty(svg) {
    el('text', {
      class: 'axis-title', x: SCH.w / 2, y: SCH.h / 2 - 6, 'text-anchor': 'middle',
    }, svg).textContent = '自回避行走 ⟨R²⟩ 对 n 的标度关系';
    el('text', {
      class: 'tick-text', x: SCH.w / 2, y: SCH.h / 2 + 20, 'text-anchor': 'middle',
    }, svg).textContent = '点「运行模拟」开始。默认 7 个 n、每个 1000 条链，约 3 秒。';
  }

  /** 参考线/拟合线在 n 处的值。后端给的是**对数截距**，所以这里是 exp(截距)·n^斜率。 */
  const at = (intercept, slope, n) => Math.exp(intercept) * Math.pow(n, slope);

  function render() {
    const svg = $('saw-chart');
    svg.textContent = '';
    state.ctx = null;

    const d = state.data;
    if (!d) { drawEmpty(svg); return; }

    // 按 n 升序 —— 用户可能把 n 填成倒序，连线和悬停都得按顺序走
    const pts = d.n.map((n, i) => ({
      n,
      y: d.R2_mean[i],
      ideal: d.R2_ideal[i],
      swell: d.swelling[i],
      se: d.rel_se[i],
      trials: d.trials[i],
      used: d.chains_used[i],
    })).sort((a, b) => a.n - b.n);

    // --- 双对数坐标域 ---
    const x0 = Math.log10(pts[0].n);
    const x1 = Math.log10(pts[pts.length - 1].n);
    const xPad = pts.length > 1 ? 0.04 * (x1 - x0) : 0.35;
    const xlo = x0 - xPad;
    const xhi = x1 + xPad;
    const nA = Math.pow(10, xlo);
    const nB = Math.pow(10, xhi);

    // 三条直线（理想、拟合、2ν）都要画到绘图区左右边缘才像一条「律」，
    // 所以纵轴范围必须把它们在**两端**的值也算进去，否则线会被画出框外。
    const lines = [];
    if (d.intercept_ideal !== null) lines.push((n) => at(d.intercept_ideal, 1, n));
    if (d.intercept_ref !== null) lines.push((n) => at(d.intercept_ref, d.slope_ref, n));
    if (d.slope !== null) lines.push((n) => at(d.intercept, d.slope, n));

    const yAll = [];
    pts.forEach((p) => {
      yAll.push(p.y, p.ideal, p.y * (1 + 2 * p.se), p.y * (1 - 2 * p.se));
    });
    lines.forEach((f) => yAll.push(f(nA), f(nB)));
    const y0 = Math.log10(Math.min(...yAll));
    const y1 = Math.log10(Math.max(...yAll));
    const yPad = Math.max(0.08 * (y1 - y0), 0.05);
    const ylo = y0 - yPad;
    const yhi = y1 + yPad;

    const sx = (v) => SCH.left + ((Math.log10(v) - xlo) / (xhi - xlo)) * SPW;
    const sy = (v) => SCH.top + SPH - ((Math.log10(v) - ylo) / (yhi - ylo)) * SPH;

    // --- 网格与刻度 ---
    // 两条护栏（与 sweep.js 逐字同源，规则在那边注释里，这里只留结论）：
    //   1) 贴着边框的刻度不留痕，但标了值的例外；
    //   2) 哪一档可以标值由 labelledKinds 逐级放宽。
    const INSET = 7;
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

    // --- 每个 n 的 ±2σ 涨落（竖线 + 上下小横）---
    // 用**实测**的 rel_se 逐点画，而不是一条公式算的带子：SAW 的 std(R²)/⟨R²⟩
    // 随 n 变（n=4 时约 0.50，n=16 时约 0.63），拿一个高斯公式套会低估大 n 的涨落。
    const err = pts.filter((p) => p.se > 0);
    if (err.length) {
      const bar = el('g', {}, svg);
      err.forEach((p) => {
        const x = sx(p.n);
        const hi = sy(p.y * (1 + 2 * p.se));
        const lo = sy(p.y * (1 - 2 * p.se));
        el('line', { class: 'saw-err', stroke: seriesVar(0), x1: x, x2: x, y1: hi, y2: lo }, bar);
        el('line', { class: 'saw-err', stroke: seriesVar(0), x1: x - 4, x2: x + 4, y1: hi, y2: hi }, bar);
        el('line', { class: 'saw-err', stroke: seriesVar(0), x1: x - 4, x2: x + 4, y1: lo, y2: lo }, bar);
      });
    }

    // --- 理想链：斜率 1（锚在质心）---
    // 颜色沿用扫描卡里「理论线 = series-1」的那一套，两张卡片学一次就够。
    if (d.intercept_ideal !== null) {
      el('line', {
        class: 'series-line', stroke: seriesVar(1),
        x1: sx(nA), y1: sy(at(d.intercept_ideal, 1, nA)),
        x2: sx(nB), y2: sy(at(d.intercept_ideal, 1, nB)),
      }, svg);
    }

    // --- 模拟点（连线用细一点，别盖住参考线）---
    if (pts.length > 1) {
      el('path', {
        class: 'series-line thin', stroke: seriesVar(0),
        d: pts.map((p, i) => `${i ? 'L' : 'M'}${sx(p.n).toFixed(2)},${sy(p.y).toFixed(2)}`).join(''),
      }, svg);
    }
    const dotG = el('g', {}, svg);
    pts.forEach((p) => {
      el('circle', {
        class: 'series-dot', cx: sx(p.n), cy: sy(p.y), r: 5, fill: seriesVar(0),
      }, dotG);
    });

    // --- 拟合直线（本次实测）---
    if (d.slope !== null) {
      el('line', {
        class: 'series-line', stroke: seriesVar(2),
        x1: sx(nA), y1: sy(at(d.intercept, d.slope, nA)),
        x2: sx(nB), y2: sy(at(d.intercept, d.slope, nB)),
      }, svg);
    }

    // --- 2ν 参考线（文献值）---
    // **画在拟合线之后**：两者只差百分之一点几，几乎重合，虚线压在实线上才看得出
    // 「这里叠着两条线」。参考语义用灰色虚线（同 .fit-line 那条约定）。
    if (d.intercept_ref !== null) {
      el('line', {
        class: 'fit-line',
        x1: sx(nA), y1: sy(at(d.intercept_ref, d.slope_ref, nA)),
        x2: sx(nB), y2: sy(at(d.intercept_ref, d.slope_ref, nB)),
      }, svg);
    }

    // --- 直接标注（排在绘图区右侧外面，分两行；两条线的斜率都写出来）---
    const label2 = (x, y, a, b) => {
      el('text', {
        class: 'sweep-label', x, y: y - 5, 'text-anchor': 'start',
      }, svg).textContent = a;
      el('text', {
        class: 'sweep-label', x, y: y + 11, 'text-anchor': 'start',
      }, svg).textContent = b;
    };
    if (d.intercept_ideal !== null) {
      label2(SCH.left + SPW + 12, sy(at(d.intercept_ideal, 1, nB)),
        '理想链', '斜率 1');
    }
    if (d.slope !== null) {
      label2(SCH.left + SPW + 12, sy(at(d.intercept, d.slope, nB)),
        '拟合', `斜率 ${d.slope.toFixed(3)}`);
    }

    // --- 悬停层 ---
    const hg = el('g', {}, svg);
    el('circle', {
      id: 'saw-hover-dot', class: 'series-dot', r: 7, cx: -99, cy: -99,
      fill: seriesVar(0), opacity: 0,
    }, hg);

    state.ctx = { pts, sx, sy, d };
    if (state.hoverIdx !== null) updateHover(Math.min(state.hoverIdx, pts.length - 1));
  }

  /* ---------------- 悬停读数 ---------------- */

  function vbPoint(ev) {
    const svg = $('saw-chart');
    const r = svg.getBoundingClientRect();
    const k = r.width / SCH.w;
    return { x: (ev.clientX - r.left) / k, y: (ev.clientY - r.top) / k, k, rect: r };
  }

  function updateHover(i) {
    const ctx = state.ctx;
    const dot = $('saw-hover-dot');
    const tt = $('saw-tooltip');
    if (!ctx || !dot || !tt) return;

    if (i === null || i === undefined) {
      dot.setAttribute('opacity', '0');
      tt.hidden = true;
      return;
    }

    const p = ctx.pts[i];
    const d = ctx.d;
    const cx = ctx.sx(p.n);
    const cy = ctx.sy(p.y);
    dot.setAttribute('cx', cx);
    dot.setAttribute('cy', cy);
    dot.setAttribute('opacity', '1');

    const svg = $('saw-chart');
    const rect = svg.getBoundingClientRect();
    const k = rect.width / SCH.w;

    tt.textContent = '';
    const head = document.createElement('div');
    head.className = 'tt-head';
    head.textContent = `n = ${p.n}`;
    tt.appendChild(head);

    const row = (name, val, color) => {
      const el2 = document.createElement('div');
      el2.className = 'tt-row';
      if (color) {
        const key = document.createElement('span');
        key.className = 'tt-key';
        key.style.background = color;
        el2.appendChild(key);
      }
      const a = document.createElement('span');
      a.className = 'tt-name';
      a.textContent = name;
      const b = document.createElement('span');
      b.className = 'tt-val';
      b.textContent = val;
      el2.append(a, b);
      tt.appendChild(el2);
    };

    row('模拟 ⟨R²⟩', fmt(p.y), seriesVar(0));
    row('理想链 n·l²', fmt(p.ideal), seriesVar(1));
    // 把 2ν 参考线在那个 n 上的值一并给出：两条线在图上分不开，
    // 差多少只能靠数字读 —— 这正是这张图要讲的事。
    if (d.intercept_ref !== null) {
      row(`2ν = ${d.slope_ref.toFixed(4)} 参考`, fmt(at(d.intercept_ref, d.slope_ref, p.n)));
    }
    row('溶胀比 ⟨R²⟩/(n·l²)', `×${p.swell.toFixed(3)}`);
    row('相对标准误', `${(p.se * 100).toFixed(2)}%`);
    row('试投 / 采到', `${p.trials.toLocaleString('en-US')} / ${p.used.toLocaleString('en-US')}`);

    tt.hidden = false;
    const px = cx * k;
    const py = cy * k;
    const ttW = tt.offsetWidth;
    const left = px + 16 + ttW > rect.width ? px - 16 - ttW : px + 16;
    tt.style.left = `${Math.max(0, left)}px`;
    tt.style.top = `${Math.min(Math.max(0, py - 12), Math.max(0, rect.height - tt.offsetHeight))}px`;
  }

  /* ---------------- 汇总行与图例 ---------------- */

  /** 五行 DOM 糖。不复用 sweep.js 里那个同名函数：它锁在对方的 IIFE 里，
   *  而这点东西没有「两份实现会各说各话」的风险（不含任何约定或数字）。 */
  function mk(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  }

  function renderLegend() {
    const box = $('saw-legend');
    box.textContent = '';
    if (!state.data) { box.hidden = true; return; }
    box.hidden = false;

    const d = state.data;
    const fitText = d.slope === null ? '拟合直线（无）' : `拟合直线 斜率 ${d.slope.toFixed(4)}`;

    [
      { k: 'dot', color: seriesVar(0), text: '模拟 ⟨R²⟩（自回避行走）' },
      { k: 'bar', color: seriesVar(0), text: '±2σ（每个 n 实测）' },
      { k: 'line', color: seriesVar(1), text: '理想链 n·l²（斜率 1）' },
      { k: 'line', color: seriesVar(2), text: fitText },
      { k: 'dash', color: 'var(--text-secondary)', text: `2ν = ${d.slope_ref.toFixed(4)}（文献）` },
    ].forEach((it) => {
      const item = mk('span', 'legend-item');
      const key = mk('span', `legend-key ${it.k}`);
      key.style.background = it.color;
      item.append(key, mk('span', null, it.text));
      box.appendChild(item);
    });
  }

  function renderSummary() {
    const box = $('saw-summary');
    box.textContent = '';
    const d = state.data;
    if (!d) return;

    const row = mk('div', 'compare-row');
    const item = (k, v) => {
      const wrap = mk('div', 'compare-item');
      wrap.append(mk('span', 'k', k), mk('span', 'v', v));
      row.appendChild(wrap);
    };

    if (d.slope === null) {
      item('拟合斜率', '–（只有一个 n）');
    } else {
      item('log⟨R²⟩ 对 log n 的拟合斜率', d.slope.toFixed(4));
      if (typeof d.slope_se === 'number') {
        item('该斜率的标准误', `±${d.slope_se.toFixed(4)}`);
      }
      if (typeof d.nu_eff === 'number') {
        item('这片区间的有效 ν = 斜率/2', d.nu_eff.toFixed(4));
      }
      item('该拟合的决定系数 R²', d.fit_r2.toFixed(5));
      if (typeof d.slope_se === 'number' && d.slope_se > 0) {
        const sig = (d.slope - d.slope_ref) / d.slope_se;
        item(`与 2ν 相差 ${(sig >= 0 ? '+' : '')}${sig.toFixed(1)}σ`, sig > 0 ? '偏高' : '偏低');
      }
    }
    const sw = d.swelling;
    item('溶胀比 ⟨R²⟩/(n·l²)', `${Math.min(...sw).toFixed(3)} → ${Math.max(...sw).toFixed(3)}`);
    // rel_se 是分数（0.0172 = 1.72%），后端给的就是这个口径 —— 和上面工具条里
    // 那一行同一个换算，别在这里漏掉 ×100。
    item('⟨R²⟩ 的相对标准误', `${(Math.min(...d.rel_se) * 100).toFixed(2)}% → ${(Math.max(...d.rel_se) * 100).toFixed(2)}%`);
    item('每个 n 采到的链数 M', d.chains.toLocaleString('en-US'));
    item('总试投链数', d.trials.reduce((a, b) => a + b, 0).toLocaleString('en-US'));
    if (state.ms !== null) {
      item('本次耗时', state.ms < 1000
        ? `${Math.round(state.ms)} ms`
        : `${(state.ms / 1000).toFixed(1)} s`);
    }
    box.appendChild(row);

    const note = mk('p', 'compare-note');
    if (d.slope === null) {
      note.textContent = '只给了一个 n，拟合不出斜率。至少要两个不同的 n 才能看出 ⟨R²⟩ 随 n 怎么长。';
    } else {
      note.textContent =
        `每个 n 采了 ${d.chains.toLocaleString('en-US')} 条独立链。`
        + '「试投」比「采到」大得多，因为这里是严格拒绝采样：每步在全部 6 个方向里'
        + '均匀选，一旦踩到走过的格点就整条丢弃 —— 只有这样才能保证每条存活链的权重相同。'
        + '存活率随 n 急剧下降（每加一个链段大约乘 0.78），所以 n = 30 比 n = 10 贵约 250 倍，'
        + '这也是 n ≤ 30 这个上限的来源：算力上限，不是物理上限。'
        + `拟合斜率与 2ν 相差 ${((d.slope - d.slope_ref) / d.slope_se).toFixed(1)} 个标准误，`
        + '两者在图上几乎重合（差百分之一点几）—— 这片 n 区间还没进渐近区，'
        + '判据是那个 σ 数，不是肉眼看两条线分不分得开。';
    }
    box.appendChild(note);
  }

  /** 后端给的警告，外加「l 改了但图还没重扫」这条本地提示。
   *
   *  **只盯着 l**：自回避行走和上面选的链模型、以及它的 θ / ⟨cosφ⟩ / p 全都无关
   *  （格点上只有 6 个单位方向，没有键角也没有持久长度）。sweep.js 里那几条
   *  「换模型了，重扫一遍」的提示在这里一条都不适用，写了反而是误导。
   */
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
    renderAlerts('saw-alerts', null, warns);
  }

  /* ---------------- 导出 ---------------- */

  const stem = (d) => `fjc-saw-${d.n.length}n-M${d.chains}-seed${d.seed}`;

  function exportCSV() {
    const d = state.data;
    if (!d) return;
    // 汇总表：每个 n 一行，全精度。纯 ASCII 表头、不带 BOM（同 chain3d.js / sweep.js）
    const lines = ['n,R2_sim,R2_ideal,swelling,rel_se,R2_2nu,h_rms_sim,h_rms_ideal,trials,chains_used'];
    d.n.forEach((n, i) => {
      lines.push([
        n, d.R2_mean[i], d.R2_ideal[i], d.swelling[i], d.rel_se[i],
        at(d.intercept_ref, d.slope_ref, n),
        d.h_rms_sim[i], d.h_rms_ideal[i], d.trials[i], d.chains_used[i],
      ].join(','));
    });
    window.fjcExport.download(
      new Blob([lines.join('\r\n')], { type: 'text/csv;charset=utf-8' }),
      `${stem(d)}.csv`
    );
  }

  function exportJSON() {
    const d = state.data;
    if (!d) return;
    const out = {
      tool: '高分子链构象计算器 —— 自回避行走标度扫描',
      generated: new Date().toISOString(),
      request: { n: d.n, l: d.l, seed: d.seed, chains: d.chains },
      model: {
        lattice: 'cubic',
        directions: 6,
        step: 'l（= 格点常数）',
        constraint: '同一个格点不重复访问',
        sampler: '严格拒绝采样：每步在全部 6 个方向里均匀选，踩到已访问格点即整条丢弃',
        note: '存活率 = c_n/6^n（c_n 为 n 步自回避行走的条数），随 n 急剧下降。'
          + '这与在**空闲**邻居里均匀选（Rosenbluth）不同：后者几乎不会死，但有偏。',
      },
      fit: d.slope === null ? null : {
        slope: d.slope,
        intercept: d.intercept,
        r2: d.fit_r2,
        slope_se: d.slope_se,
        nu_eff: d.nu_eff,
        slope_ref: d.slope_ref,
        intercept_ref: d.intercept_ref,
        intercept_ideal: d.intercept_ideal,
        note: 'log⟨R²⟩ 对 log n 的最小二乘拟合。两条参考线（理想链斜率 1、文献 2ν）'
          + '都锚在数据质心上。**有限尺寸**：n ≤ 30 这片区间测出的斜率偏高，'
          + '渐近值 2ν = 1.175194 要 n 很大才到；报的是有效 ν = 斜率/2。',
      },
      result: d,
    };
    window.fjcExport.download(
      new Blob([JSON.stringify(out, null, 2)], { type: 'application/json;charset=utf-8' }),
      `${stem(d)}.json`
    );
  }

  /* ---------------- 运行 ---------------- */

  function parseNs(raw) {
    const parts = String(raw).split(/[,，、;；\s]+/).filter((s) => s.length > 0);
    if (parts.length === 0) throw new Error('请输入链段数 n');
    const out = parts.map((p) => {
      const v = Number(p);
      if (!isFinite(v)) throw new Error(`「${p}」不是数字`);
      if (v > N_MAX) {
        throw new Error(`自回避行走的 n 不能超过 ${N_MAX}（收到 ${v}）：`
          + '严格拒绝采样的存活率按 0.78^n 衰减，更大的 n 跑不完。这是算力上限，不是物理上限。');
      }
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
      ns = parseNs($('w-n').value);
      l = Number($('in-l').value);          // l 与顶部参数行共用，和 3D 视图一致
      if (!isFinite(l) || l <= 0) throw new Error('链段长度 l 必须是大于 0 的数字');
      const rawM = $('w-chains').value.trim();
      chains = rawM === '' ? undefined : Number(rawM);
      if (chains !== undefined && !isFinite(chains)) throw new Error('每个 n 的链数 M 必须是整数');
      const rawSeed = $('w-seed').value.trim();
      seed = rawSeed === '' ? null : Number(rawSeed);
      if (seed !== null && (!isFinite(seed) || seed < 0)) throw new Error('随机种子必须是非负整数');
    } catch (e) {
      renderAlerts('saw-alerts', e.message, []);
      return;
    }

    $('saw-body').classList.add('busy');
    $('w-run').disabled = true;
    $('w-csv').disabled = true;
    $('w-json').disabled = true;
    $('w-png').disabled = true;
    renderAlerts('saw-alerts', null, []);

    const t0 = performance.now();
    try {
      const res = await fetch('/api/saw/sweep', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        // **不发链模型参数**：自回避行走不吃 model / theta / cosphi / p，
        // 后端也一律忽略（它只认 n、l、seed、chains）。
        body: JSON.stringify({ n: ns, l, seed, chains }),
      });
      const data = await res.json().catch(() => ({ error: '服务器返回了无法解析的内容' }));
      if (!res.ok) throw new Error(data.error || `请求失败（HTTP ${res.status}）`);

      state.data = data;
      state.ms = performance.now() - t0;
      state.hoverIdx = null;
      $('w-seed').value = String(data.seed);

      render();
      renderLegend();
      renderSummary();
      refreshAlerts();
      $('w-csv').disabled = false;
      $('w-json').disabled = false;
      $('w-png').disabled = false;
    } catch (e) {
      renderAlerts('saw-alerts', e.message, []);
    } finally {
      $('saw-body').classList.remove('busy');
      $('w-run').disabled = false;
    }
  }

  /* ---------------- 事件 ---------------- */

  function initEvents() {
    $('w-run').addEventListener('click', run);
    $('w-small').addEventListener('click', () => {
      $('w-n').value = SMALL.n;
      $('w-chains').value = SMALL.m;
      run();
    });
    ['w-n', 'w-chains', 'w-seed'].forEach((id) => {
      $(id).addEventListener('keydown', (ev) => {
        if (ev.key === 'Enter') { ev.preventDefault(); run(); }
      });
    });
    $('w-csv').addEventListener('click', exportCSV);
    $('w-json').addEventListener('click', exportJSON);
    $('w-png').addEventListener('click', () => {
      if (!state.data) return;
      window.fjcExport.svgToPng($('saw-chart'), `${stem(state.data)}.png`)
        .catch((e) => window.fjcExport.pngFail('saw-alerts', e));
    });

    // l 改了：图没变，但要立刻把「该重扫了」说出来（唯一的失效条件，见 refreshAlerts）
    $('in-l').addEventListener('input', refreshAlerts);

    const svg = $('saw-chart');
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

    // 键盘：←/→ 在扫描点之间走，Esc 收起。和另外两张图一个路子。
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

  // 和扫描卡一样不点不算：这张最贵（默认一次约 3 秒，M 拉大就是几十秒），
  // 页面一打开就自动跑没有道理。
  initEvents();
  render();

  /* 但**在模型下拉里选中自回避行走时**要自动跑一次：别的模型一切换就自动重算，
     这张不自动的话，选中 SAW 只会看到一张空卡片加一个按钮 —— 与整页手感不一致。
     已经有数据就不重跑：来回切模型不该每次都烧掉这一两秒。

     推到下一个宏任务再跑：让出卡片可见的那一步。reveal 这张卡的是 app.js 的
     syncModelFields()（它把 body[data-model-kind] 设成 simulated），而它的监听器
     是在 DOMContentLoaded 里注册的，比这里晚 —— 同一个事件上这里先跑，
     此刻这张卡还是 display:none。东西不落地也能画对（图用的是固定 viewBox），
     但没必要踩这个边界。 */
  const autoRun = () => {
    const spec = window.fjcApp ? window.fjcApp.modelSpec() : null;
    if (!spec || spec.analytic || state.data) return;
    setTimeout(() => { if (!state.data) run(); }, 0);
  };
  $('in-model').addEventListener('change', autoRun);
  document.addEventListener('fjc:models-ready', autoRun);
})();
