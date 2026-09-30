'use strict';

/* 能量图景 —— 只有**真有能量自由度**的两个模型才有内容
 *
 *   受阻旋转链（hindered）：内旋转势垒。三态里 trans 在 φ=0°、gauche± 在 φ=±120°，
 *     gauche 相对 trans 的能量差 ΔE/kT = ln((1+2⟨cosφ⟩)/(1−⟨cosφ⟩))。
 *     这是「受阻」两个字的量化说法，也是 ⟨cosφ⟩ 这个滑块唯一的物理解释。
 *     左图画的是**势** U(φ)：一条三井曲线，井底正好落在那三个态上
 *     （后端 hindered_potential() 造出来的）。**井底位置与井深 ΔE 是模型定的，
 *     势垒高度是画法约定** —— 模型只给了 ΔE 一个数。右图画的才是**布居**，
 *     而采样器只抽那三个**离散**值，所以是三根柱；对左边那条曲线做玻尔兹曼积分
 *     得到的权重与这三根柱**不是**同一个数，别把两张图当成一回事。
 *   蠕虫状链（wlc）：弯曲刚度 κ。采样器抽的 p(x) ∝ e^(κx) 就是势
 *     U(cosθ)/kT = κ(1−cosθ) 的玻尔兹曼分布，值域 0…2κ。
 *
 * 自由连接链与自由旋转链**没有**能量自由度（前者除固定键长外无约束，后者的键角是
 * 硬约束不是势），接口对它们返回 kind="none" 连同一句中文理由，这里就什么都不画 ——
 * 「为什么没有」写在后端一处（fjc_core.energy_figure()），本文件不抄第二份判断。
 *
 * 和这个项目里别的图一样：**这里不做任何物理计算**，四个数组全都由 POST /api/energy
 * 算好返回（唯一那份在 fjc_core.energy_figure()），本文件只做坐标变换和画图。
 * 颜色直接写 `var(--series-N)`：SVG 认 CSS 变量，深浅色切换时不用重画。
 */
(function () {
  const $ = (id) => document.getElementById(id);
  const NS = 'http://www.w3.org/2000/svg';
  const S1 = 'var(--series-1)';
  const S2 = 'var(--series-2)';

  let d = null;        // /api/energy 的返回

  /* ---------------- 画图小工具 ---------------- */

  function mk(tag, attrs, parent) {
    const n = document.createElementNS(NS, tag);
    Object.entries(attrs || {}).forEach(([k, v]) => n.setAttribute(k, v));
    if (parent) parent.appendChild(n);
    return n;
  }

  function txt(svg, x, y, s, cls, anchor) {
    const t = mk('text', { x, y, class: cls || 'tick-text' }, svg);
    if (anchor) t.setAttribute('text-anchor', anchor);
    t.textContent = s;
    return t;
  }

  function ln(svg, x1, y1, x2, y2, cls) {
    return mk('line', { x1, y1, x2, y2, class: cls || 'grid-line' }, svg);
  }

  function poly(svg, pts, stroke, width) {
    return mk('polyline', {
      points: pts.map((p) => `${p[0]},${p[1]}`).join(' '),
      fill: 'none', stroke,
      'stroke-width': width || 2.2, 'stroke-linejoin': 'round',
    }, svg);
  }

  function dot(svg, x, y, color) {
    return mk('circle', { cx: x, cy: y, r: 4.5, fill: color || S1 }, svg);
  }

  /** 一块绘图区：清空、画边框、返回 (sx, sy) 两个映射。logx / logy 走 10 为底。 */
  function pane(svg, r, xlo, xhi, ylo, yhi, opts) {
    const o = opts || {};
    const fx = o.logx
      ? (v) => (Math.log10(v) - Math.log10(xlo)) / (Math.log10(xhi) - Math.log10(xlo))
      : (v) => (v - xlo) / (xhi - xlo || 1);
    const fy = o.logy
      ? (v) => (Math.log10(v) - Math.log10(ylo)) / (Math.log10(yhi) - Math.log10(ylo))
      : (v) => (v - ylo) / (yhi - ylo || 1);
    mk('rect', { x: r.x, y: r.y, width: r.w, height: r.h, fill: 'none', class: 'axis-line' }, svg);
    return {
      r,
      sx: (v) => r.x + fx(v) * r.w,
      sy: (v) => r.y + r.h - fy(v) * r.h,
    };
  }

  /** 绘图区里的水平格线 + 左侧刻度文字。 */
  function yGrid(svg, p, values) {
    values.forEach((v) => {
      ln(svg, p.r.x, p.sy(v), p.r.x + p.r.w, p.sy(v));
      txt(svg, p.r.x - 10, p.sy(v) + 4, fmtTick(v, false), 'tick-text', 'end');
    });
  }

  /** 绘图区里的竖直格线 + 下方刻度文字。label 可覆盖默认写法。 */
  function xGrid(svg, p, values, label) {
    values.forEach((v) => {
      ln(svg, p.sx(v), p.r.y, p.sx(v), p.r.y + p.r.h);
      txt(svg, p.sx(v), p.r.y + p.r.h + 18, (label || fmtTick)(v, false), 'tick-text', 'middle');
    });
  }

  /** [lo, hi] 之内的 10 的整数次幂。对数轴用。 */
  function logTicks(lo, hi) {
    const out = [];
    for (let e = Math.ceil(Math.log10(lo) - 1e-9); e <= Math.floor(Math.log10(hi) + 1e-9); e++) {
      out.push(Math.pow(10, e));
    }
    return out;
  }

  /** 10 的幂的短写法：0.01 / 0.1 / 1 / 10，不用 1.0e-2。 */
  const powLabel = (v) => (v >= 1 ? String(v) : v.toFixed(-Math.floor(Math.log10(v) + 1e-9)));

  /* ---------------- 受阻旋转链：势能台阶 + 三态布居 ---------------- */

  function drawTorsion() {
    const svg = $('torsion-chart');
    svg.textContent = '';
    const lv = d.levels;
    const b = d.barrier;
    // gauche 相对 trans 的高度。cos_phi = COS_PHI_MIN 时它是后端截断过的 −3
    // （真值是 −∞，画不出来），此时那条虚线标的是「ΔE = −∞」而不是这个数。
    const uG = lv[1].u_over_kt;
    const phis = lv.map((x) => x.phi_deg);
    // 纵轴范围由后端给 —— 它才知道曲线长什么样（ΔE < 0 时 0° 那个井被抬到两个
    // gauche 井上面，上界该留多少不是前端能推的）。这里只加一点上下余量。
    const pad = (d.u_max - d.u_min) * 0.08 || 0.1;
    const uHi = d.u_max + pad;
    const uLo = d.u_min - pad;

    // --- 左栏：U(φ)/kT 的连续曲线（三井曲线，井底落在三个态上） ---
    // 这是**势**（模型的输入），右栏那三根柱才是**布居**（模型的输出）。
    // 井底位置与井深 ΔE 由模型定死；**势垒高度是约定**（后端 TORSION_BARRIER_RATIO）。
    // 注意它**不是 120° 周期的**：井底间距 120°，但两条势垒不一样高，整条是 360° 周期。
    // 对这条曲线做玻尔兹曼积分**不等于**右栏那三根柱，见 fjc_core.hindered_potential()。
    const p1 = pane(svg, { x: 78, y: 46, w: 344, h: 258 }, -180, 180, uLo, uHi);
    yGrid(svg, p1, (d.u_min < 0 ? [d.u_min, d.u_min / 2] : []).concat([0, d.u_max / 2]));
    xGrid(svg, p1, [-180, -120, 0, 120, 180]);
    ln(svg, p1.r.x, p1.sy(0), p1.r.x + p1.r.w, p1.sy(0), 'ref-line');
    txt(svg, 12, p1.r.y - 12,
      `U / kT（1 kT = ${fmt(d.kt_in_kj_per_mol)} kJ/mol）`, 'axis-title');

    poly(svg, d.phi_deg.map((p, i) => [p1.sx(p), p1.sy(d.potential[i])]), S1);

    // 三个离散态标在**曲线上**（后端保证井底与 u_over_kt 逐点相等，点正好落在井底）。
    // 名字写在井底**下方** —— 井是向上张开的，底下一定是空的。
    lv.forEach((x) => {
      dot(svg, p1.sx(x.phi_deg), p1.sy(x.u_over_kt), S2);
      txt(svg, p1.sx(x.phi_deg), p1.sy(x.u_over_kt) + 18, x.name, 'tick-text', 'middle');
    });

    // 两个井底之间的高度差就是 ΔE。φ = 60° 那根竖虚线从 trans 那级量到 gauche 那级；
    // 曲线在 60° 附近是隆起的，所以这段虚线整段落在曲线**下方**的空处。
    // ΔE 恰好为 0（⟨cosφ⟩ = 0）时**不画** —— 那正是三态等高、退回自由旋转链的那一点，
    // 画一条零长度的虚线加个「ΔE」反而像是还有位垒。
    if (b.all_gauche || b.delta_e_over_kt) {
      ln(svg, p1.sx(60), p1.sy(0), p1.sx(60), p1.sy(uG), 'ref-line');
      txt(svg, p1.sx(60) + 8, (p1.sy(0) + p1.sy(uG)) / 2 + 4,
        b.all_gauche ? 'ΔE = −∞' : `ΔE = ${fmt(b.delta_e_over_kt)} kT`,
        'ref-text', 'start');
    }
    txt(svg, p1.r.x + p1.r.w / 2, 366, '内旋转角 φ（度）', 'axis-title', 'middle');

    // --- 右栏：三态布居 ---
    const wMax = Math.max(...lv.map((x) => x.weight)) * 1.28 || 1;
    const p2 = pane(svg, { x: 592, y: 46, w: 258, h: 258 }, -180, 180, 0, wMax);
    yGrid(svg, p2, [0, wMax / 2]);
    xGrid(svg, p2, phis);
    txt(svg, 534, p2.r.y - 12, '布居 P', 'axis-title');
    lv.forEach((x) => {
      mk('rect', {
        x: p2.sx(x.phi_deg - 22), y: p2.sy(x.weight),
        width: p2.sx(x.phi_deg + 22) - p2.sx(x.phi_deg - 22),
        height: p2.r.y + p2.r.h - p2.sy(x.weight),
        fill: S2, 'fill-opacity': 0.75,
      }, svg);
      txt(svg, p2.sx(x.phi_deg), p2.sy(x.weight) - 8, fmt(x.weight), 'tick-text', 'middle');
    });
    txt(svg, p2.r.x + p2.r.w / 2, 366, '内旋转角 φ（度）', 'axis-title', 'middle');
  }

  /* ---------------- 受阻旋转链：Cn 随位垒 ---------------- */

  function drawCnBarrier() {
    const svg = $('cn-barrier-chart');
    svg.textContent = '';
    const xs = d.cn.delta_e;
    const ys = d.cn.values;
    const xlo = xs[0];
    const xhi = xs[xs.length - 1];
    const ylo = ys[0] / 1.7;
    const yhi = ys[ys.length - 1] * 1.7;
    const p = pane(svg, { x: 92, y: 34, w: 700, h: 292 }, xlo, xhi, ylo, yhi, { logy: true });
    yGrid(svg, p, logTicks(ylo, yhi));
    xGrid(svg, p, [-3, -2, -1, 0, 1, 2, 3, 4, 5].filter((v) => v >= xlo && v <= xhi));
    txt(svg, 12, p.r.y - 12, 'Cn（对数）', 'axis-title');
    txt(svg, p.r.x + p.r.w / 2, 376, '位垒 ΔE/kT（gauche 相对 trans）', 'axis-title', 'middle');

    // ΔE = 0 就是三态等权重、绕键自由旋转 —— 此时 Cn 精确等于自由旋转链那个值。
    // 这条参考线是整张图最硬的一个锚点。
    ln(svg, p.sx(0), p.r.y, p.sx(0), p.r.y + p.r.h, 'ref-line');
    txt(svg, p.sx(0) + 8, p.r.y + 16, `ΔE = 0：自由旋转链 Cn = ${fmt(d.cn.frc)}`, 'ref-text');
    ln(svg, p.r.x, p.sy(d.cn.frc), p.r.x + p.r.w, p.sy(d.cn.frc), 'ref-line');

    poly(svg, xs.map((x, i) => [p.sx(x), p.sy(ys[i])]), S1);

    // 聚乙烯那个点（θ=109.47°、⟨cosφ⟩=0.5）与当前点
    dot(svg, p.sx(d.cn.pe_delta_e), p.sy(d.cn.pe_cn), S2);
    txt(svg, p.sx(d.cn.pe_delta_e) - 6, p.sy(d.cn.pe_cn) - 10, `聚乙烯 ${fmt(d.cn.pe_cn)}`,
      'ref-text', 'end');

    const allGauche = d.barrier.all_gauche;
    const xNow = allGauche ? xlo : d.barrier.delta_e_over_kt;
    dot(svg, p.sx(xNow), p.sy(d.cn.now), S1);
    txt(svg, p.sx(xNow) + (allGauche ? 10 : -8), p.sy(d.cn.now) + 18,
      allGauche ? `当前（ΔE = −∞）Cn = ${fmt(d.cn.now)}` : `当前 Cn = ${fmt(d.cn.now)}`,
      'ref-text', allGauche ? 'start' : 'end');
  }

  /* ---------------- 蠕虫状链：弯曲势 + 键角余弦分布 ---------------- */

  function drawBending() {
    const svg = $('bending-chart');
    svg.textContent = '';
    const cs = d.cos;
    const uMax = Math.max(d.u_max, 1e-9);

    // --- 左栏：U(cosθ)/kT = κ(1 − cosθ)，从 2κ（完全反向）降到 0（对齐） ---
    const p1 = pane(svg, { x: 86, y: 46, w: 330, h: 258 }, -1, 1, 0, uMax * 1.16);
    yGrid(svg, p1, [0, uMax / 2, uMax]);
    xGrid(svg, p1, [-1, 0, 1]);
    txt(svg, 12, p1.r.y - 12, 'U / kT', 'axis-title');
    poly(svg, cs.map((c, i) => [p1.sx(c), p1.sy(d.potential[i])]), S1);
    txt(svg, p1.sx(1), p1.sy(0) - 10, '相邻键对齐 0', 'ref-text', 'end');
    txt(svg, p1.sx(-1), p1.sy(uMax) - 10, `完全反向 ${fmt(uMax)}`, 'ref-text', 'start');
    txt(svg, p1.r.x + p1.r.w / 2, 366, '相邻键夹角的余弦 cos θ', 'axis-title', 'middle');

    // --- 右栏：该势下的玻尔兹曼分布 ---
    const dMax = Math.max(...d.density) * 1.14 || 1;
    const p2 = pane(svg, { x: 600, y: 46, w: 258, h: 258 }, -1, 1, 0, dMax);
    yGrid(svg, p2, [0, dMax / 2]);
    xGrid(svg, p2, [-1, 0, 1]);
    txt(svg, 546, p2.r.y - 12, 'P(cos θ)', 'axis-title');
    poly(svg, cs.map((c, i) => [p2.sx(c), p2.sy(d.density[i])]), S2, 2.2);
    // ⟨cosθ⟩ = L(κ) = e^(−l/p) 是构造出来的恒等式，页面上模型指纹卡的 corr[1] 也是它
    ln(svg, p2.sx(d.corr), p2.r.y, p2.sx(d.corr), p2.r.y + p2.r.h, 'ref-line');
    txt(svg, p2.sx(d.corr) - 8, p2.r.y + 16, `⟨cosθ⟩ = ${fmt(d.corr)}`, 'ref-text', 'end');
    txt(svg, p2.r.x + p2.r.w / 2, 366, '相邻键夹角的余弦 cos θ', 'axis-title', 'middle');
  }

  /* ---------------- 蠕虫状链：持久长度随弯曲刚度 ---------------- */

  function drawPlKappa() {
    const svg = $('pl-kappa-chart');
    svg.textContent = '';
    const ks = d.kappa_scan;
    const pl = d.pl_scan;
    const ylo = Math.min(...pl) / 1.6;
    const yhi = Math.max(...pl) * 1.6;
    const p = pane(svg, { x: 96, y: 34, w: 690, h: 292 }, ks[0], ks[ks.length - 1], ylo, yhi,
      { logx: true, logy: true });
    yGrid(svg, p, logTicks(ylo, yhi));
    xGrid(svg, p, logTicks(ks[0], ks[ks.length - 1]), powLabel);
    txt(svg, 12, p.r.y - 12, '持久长度 p/l（对数）', 'axis-title');
    txt(svg, p.r.x + p.r.w / 2, 376, '弯曲刚度 κ（kT，对数）', 'axis-title', 'middle');

    // 渐近线 κ ≈ p/l + 1/2：大 κ 端两条线几乎重合，小 κ 端才分家 —— 分家的地方
    // 正是「κ 就是 p/l」这个直觉失效的地方
    poly(svg, d.pl_asymptote.map((q) => [p.sx(q[0]), p.sy(q[1])]), 'var(--series-3)', 1.6);
    const aMid = d.pl_asymptote[Math.floor(d.pl_asymptote.length * 0.75)];
    txt(svg, p.sx(aMid[0]) + 8, p.sy(aMid[1]) + 4, 'κ ≈ p/l + 1/2', 'ref-text');
    poly(svg, ks.map((k, i) => [p.sx(k), p.sy(pl[i])]), S1);
    dot(svg, p.sx(d.kappa), p.sy(d.p_over_l), S1);
    txt(svg, p.sx(d.kappa) - 8, p.sy(d.p_over_l) + 18,
      `当前 κ = ${fmt(d.kappa)} kT`, 'ref-text', 'end');
  }

  /* ---------------- 读数条 ---------------- */

  function readout() {
    const box = $('en-readout');
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
    if (d.kind === 'torsion') {
      const b = d.barrier;
      // 「井深差」而不是「位垒」：ΔE 是两个**井底**的高度差（模型定的），
      // 势垒是井与井之间那道梁的高度（画法约定的），两者不是一回事。
      item('井深差 ΔE', b.all_gauche ? '−∞（全 gauche）'
        : `${fmt(b.delta_e_over_kt)} kT`);
      // 势垒高度按后端的 TORSION_BARRIER_RATIO 约定画；all_gauche 时它建在
      // 截断值 −3 上（真值是 −∞），不注明的话 12 kT 看着像个真数。
      item('势垒（gauche↔gauche）',
        `${fmt(b.gg_over_kt)} kT${b.all_gauche ? '（按截断的 −3 算）' : ''}`);
      item('gauche : trans', b.sigma === null ? '∞' : `${fmt(b.sigma)} : 1`);
      item('trans 占比', `${fmt(100 * b.p_trans)}%`);
      item('Cn', fmt(d.cn.now));
    } else if (d.kind === 'bending') {
      item('弯曲刚度 κ', `${fmt(d.kappa)} kT`);
      item('相邻键对齐的收益', `${fmt(d.u_max)} kT`);
      item('⟨cos θ⟩', fmt(d.corr));
      item('持久长度', `${fmt(d.p_over_l)} l`);
    }
    box.hidden = box.childElementCount === 0;
  }

  /* ---------------- 取数与联动 ---------------- */

  // 请求序号。参数框上 input 与 change 各挂了一次 load（和 models.js 一样），
  // 手快连改两下就会有两个请求在飞 —— 它们的**返回顺序没有保证**，后到的旧响应
  // 会把新图画回去（实测抓到过一次：cos_phi 已经改成 0，画面上还是 0.5 的位垒）。
  // 所以只认最后一次发出去的那个请求。
  let seq = 0;

  async function load() {
    // 自由连接链 / 自由旋转链没有能量自由度，接口也只能给个 none。卡片本身由
    // CSS 按 body[data-model] 收着，这里连这一次请求都省掉 ——
    // 与 models.js 对自回避行走那条守卫同一条规矩：不取不该取的数。
    const spec = window.fjcApp ? window.fjcApp.modelSpec() : null;
    const key = spec ? spec.key : null;
    const card = $('energy-card');
    if (key !== 'hindered' && key !== 'wlc') { d = null; delete card.dataset.kind; return; }

    const params = window.fjcApp ? window.fjcApp.modelParams() : {};
    const mine = ++seq;
    try {
      const res = await fetch('/api/energy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      });
      const data = await res.json().catch(() => ({ error: '服务器返回了无法解析的内容' }));
      if (!res.ok) throw new Error(data.error || `请求失败（HTTP ${res.status}）`);
      if (mine !== seq) return;
      d = data;
      card.dataset.kind = d.kind;
      if (d.kind === 'torsion') { drawTorsion(); drawCnBarrier(); }
      if (d.kind === 'bending') { drawBending(); drawPlKappa(); }
      readout();
    } catch (e) {
      if (mine !== seq) return;
      d = null;
      delete card.dataset.kind;
      const box = $('en-readout');
      box.textContent = `能量图景暂时取不到数据：${e.message}`;
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

  window.fjcEnergy = { reload: load, data: () => d };
})();
