'use strict';

/* 模型指纹 —— 两张**纯理论**曲线（不采样、不算随机数）
 *
 *   左：方向关联 ⟨u₀·u_k⟩ —— 隔 k 个链段以后，两个键还指得多一致。
 *   右：Cn 随链长的收敛 —— 教科书那个 C∞ 是怎么在长链极限下才出现的。
 *
 * 和这个项目里别的图一样：**这里不做任何物理计算**，两条曲线都由
 * POST /api/model-curves 算好返回（唯一那份在 fjc_core.model_fingerprint()），
 * 本文件只做坐标变换和画图。四条曲线是四个模型，当前选中的那条画粗一点。
 *
 * 颜色直接写 `var(--series-N)`：SVG 的 fill/stroke 认 CSS 变量，所以深浅色切换时
 * 不用重画（canvas 不认，3D 那张图才需要 getComputedStyle 现取）。
 */
(function () {
  const $ = (id) => document.getElementById(id);
  const SERIES = ['var(--series-1)', 'var(--series-2)', 'var(--series-3)', 'var(--series-4)'];
  const BOX = { left: 64, right: 96, top: 20, bottom: 42, w: 600, h: 320 };
  const PW = BOX.w - BOX.left - BOX.right;
  const PH = BOX.h - BOX.top - BOX.bottom;
  const FONT = '12px system-ui, "Segoe UI", "Microsoft YaHei", sans-serif';

  let fp = null;        // /api/model-curves 的返回

  function mk(tag, attrs, parent) {
    const n = document.createElementNS('http://www.w3.org/2000/svg', tag);
    Object.entries(attrs || {}).forEach(([k, v]) => n.setAttribute(k, v));
    if (parent) parent.appendChild(n);
    return n;
  }

  /** 给一张图配好坐标系：清空、画坐标轴、返回 (sx, sy) 两个映射函数。 */
  function frame(svg, xlo, xhi, ylo, yhi) {
    svg.textContent = '';
    const sx = (v) => BOX.left + ((v - xlo) / (xhi - xlo || 1)) * PW;
    const sy = (v) => BOX.top + PH - ((v - ylo) / (yhi - ylo || 1)) * PH;
    mk('rect', {
      x: BOX.left, y: BOX.top, width: PW, height: PH,
      fill: 'none', class: 'axis-line',
    }, svg);
    return { sx, sy };
  }

  function text(svg, x, y, str, cls, anchor) {
    const t = mk('text', { x, y, class: cls || 'tick-text' }, svg);
    if (anchor) t.setAttribute('text-anchor', anchor);
    t.textContent = str;
    return t;
  }

  function grid(svg, sx, sy, xs, ys) {
    ys.forEach((v) => {
      mk('line', {
        x1: BOX.left, x2: BOX.left + PW, y1: sy(v), y2: sy(v), class: 'grid-line',
      }, svg);
      text(svg, BOX.left - 10, sy(v) + 4, fmt(v), 'tick-text', 'end');
    });
    xs.forEach((v) => {
      mk('line', {
        x1: sx(v), x2: sx(v), y1: BOX.top, y2: BOX.top + PH, class: 'grid-line',
      }, svg);
      text(svg, sx(v), BOX.top + PH + 18, fmt(v), 'tick-text', 'middle');
    });
  }

  /* ---------------- 左：方向关联 ---------------- */

  function drawCorr(d) {
    const svg = $('corr-chart');
    const kmax = d.k[d.k.length - 1];
    const all = d.compare.flatMap((r) => r.corr);
    const ylo = Math.min(0, Math.min(...all));
    const { sx, sy } = frame(svg, 0, kmax, ylo, 1.02);
    const kstep = kmax <= 10 ? 2 : 10;
    const xs = [];
    for (let v = 0; v <= kmax; v += kstep) xs.push(v);
    const ys = ylo < 0 ? [ylo, 0, 0.5, 1] : [0, 0.25, 0.5, 0.75, 1];
    grid(svg, sx, sy, xs, ys);

    // 0 那条参考线（FJC 的关联贴着它）
    mk('line', { x1: BOX.left, x2: BOX.left + PW, y1: sy(0), y2: sy(0), class: 'ref-line' }, svg);

    d.compare.forEach((row, i) => {
      const cur = row.key === d.model;
      const pts = row.corr.map((v, k) => `${sx(k)},${sy(v)}`).join(' ');
      mk('polyline', {
        points: pts, fill: 'none', stroke: SERIES[i % 4],
        'stroke-width': cur ? 2.6 : 1.4,
        'stroke-opacity': cur ? 1 : 0.55,
        'stroke-linejoin': 'round',
      }, svg);
    });
    text(svg, BOX.left + PW / 2, BOX.h - 8, '间隔的链段数 k', 'axis-title', 'middle');
    // 纵轴标题摆在**绘图区上方**，不是内部左上角：放在内部时它和顶端那个刻度「1」
    // 会在 x 上撞到一起（包围盒重叠检测实测抓到过）。上方那一行本来就是空的。
    text(svg, 12, BOX.top - 8, '⟨u₀·u_k⟩', 'axis-title');
    legend(d);
  }

  /* ---------------- 右：Cn 的收敛 ---------------- */

  function drawCn(d) {
    const svg = $('cn-chart');
    const xlo = Math.log10(1);
    const xhi = Math.log10(d.ns[d.ns.length - 1]);
    const top = Math.max(...d.compare.flatMap((r) => r.cn)) * 1.08;
    const { sx, sy } = frame(svg, xlo, xhi, 0, top);
    const lx = (n) => Math.log10(n);
    const xs = [1, 10, 100, 1000, 10000, 100000, 1000000].filter((v) => lx(v) <= xhi + 1e-9);
    const ys = [0, top / 3, (2 * top) / 3, top].map((v) => Math.round(v * 1000) / 1000);
    // 对数横轴：格线按 10 的幂走，刻度文字用 1k / 100k 这种短写法
    ys.forEach((v) => {
      mk('line', { x1: BOX.left, x2: BOX.left + PW, y1: sy(v), y2: sy(v), class: 'grid-line' }, svg);
      text(svg, BOX.left - 10, sy(v) + 4, fmt(v), 'tick-text', 'end');
    });
    xs.forEach((v) => {
      mk('line', { x1: sx(lx(v)), x2: sx(lx(v)), y1: BOX.top, y2: BOX.top + PH, class: 'grid-line' }, svg);
      const label = v >= 1000 ? `${v / 1000}k` : String(v);
      text(svg, sx(lx(v)), BOX.top + PH + 18, label, 'tick-text', 'middle');
    });

    d.compare.forEach((row, i) => {
      const cur = row.key === d.model;
      const pts = d.ns.map((n, j) => `${sx(lx(n))},${sy(row.cn[j])}`).join(' ');
      mk('polyline', {
        points: pts, fill: 'none', stroke: SERIES[i % 4],
        'stroke-width': cur ? 2.6 : 1.4,
        'stroke-opacity': cur ? 1 : 0.55,
        'stroke-linejoin': 'round',
      }, svg);
    });

    // 当前模型的长链极限：虚线（参考量的语义，和别的图一致）+ 直接标在线上
    const inf = d.cn_infinite;
    if (inf < top) {
      mk('line', {
        x1: BOX.left, x2: BOX.left + PW, y1: sy(inf), y2: sy(inf), class: 'ref-line',
      }, svg);
      text(svg, BOX.left + PW + 6, sy(inf) + 4,
           `C∞ = ${fmt(inf)}`, 'ref-text');
    }
    text(svg, BOX.left + PW / 2, BOX.h - 8, '链段数 n（对数）', 'axis-title', 'middle');
    text(svg, 12, BOX.top - 8, 'Cn', 'axis-title');
  }

  function legend(d) {
    const box = $('fp-legend');
    box.textContent = '';
    box.hidden = false;
    d.compare.forEach((row, i) => {
      const item = document.createElement('span');
      item.className = 'legend-item';
      const key = document.createElement('span');
      key.className = 'legend-key';
      key.style.background = SERIES[i % 4];
      key.style.opacity = row.key === d.model ? '1' : '0.55';
      const t = document.createElement('span');
      t.textContent = row.label + (row.key === d.model ? '（当前）' : '');
      item.append(key, t);
      box.appendChild(item);
    });
  }

  /* ---------------- 取数与联动 ---------------- */

  async function load() {
    // 模型指纹画的是**纯理论**曲线（不采样、不算随机数），自回避行走没有闭式解、
    // 根本没有这样的曲线可画。卡片已由 CSS 收起，这里再省掉这次请求，
    // 并把上一份数据丢掉 —— 否则下次切回解析模型时会先闪一下旧曲线。
    const spec = window.fjcApp ? window.fjcApp.modelSpec() : null;
    if (spec && !spec.analytic) { fp = null; return; }

    const params = window.fjcApp ? window.fjcApp.modelParams() : {};
    try {
      const res = await fetch('/api/model-curves', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      });
      const data = await res.json().catch(() => ({ error: '服务器返回了无法解析的内容' }));
      if (!res.ok) throw new Error(data.error || `请求失败（HTTP ${res.status}）`);
      fp = data;
      drawCorr(fp);
      drawCn(fp);
    } catch (e) {
      const box = $('fp-legend');
      box.textContent = `模型指纹暂时取不到数据：${e.message}`;
      box.hidden = false;
    }
  }

  ['in-model', 'in-theta', 'in-cosphi', 'in-p'].forEach((id) => {
    const el = $(id);
    if (!el) return;
    el.addEventListener('change', load);
    el.addEventListener('input', load);
  });
  // 模型目录到齐以后才知道用户存的是哪个模型，所以等 app.js 发这个信号再画一次
  document.addEventListener('fjc:models-ready', load);
  load();

  window.fjcModels = { reload: load, data: () => fp };
})();
