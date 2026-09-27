'use strict';

/* 背景节点网络 —— 纯装饰的一层底纹
 *
 * 为什么偏偏是「节点 + 连线」：这个软件算的就是自由连接链，一层慢慢漂移的
 * 节点网络正好是它自己的视觉隐喻。换成随便一串粒子也好看，但那是另一回事。
 * 它底下还压着一层**会闪的星**：细小的点各自按自己的相位一明一暗，少数几颗
 * 亮到峰值时迸出十字耀斑。星和节点是两种尺度的东西（星是「远处」、节点是
 * 「近处」），合起来这层底纹才不像一张贴图，而像一片有纵深的空间。
 * 浅色主题下白点几乎看不见 —— 所以亮过一档的星会多一圈**冷色柔光**，
 * 那圈光才是它在浅底上「看得见」的原因（也顺便让深浅两套共用同一份代码）。
 *
 * 四条规矩，改这个文件之前先读一遍：
 *
 *  1. **零物理、零数据。** 这里画的东西与 ⟨h²⟩ = n·l² 没有任何关系：不读接口、
 *     不碰 DOM、不写全局状态，只在一张 position:fixed / z-index:-1 的 canvas 上
 *     画点线。「物理公式只在 fjc_core.py 一处」这条没有被它破坏。
 *
 *  2. **颜色全部从 CSS 的 --net-* token 现取**，深浅色切换时重取一遍 ——
 *     和 chain3d.js 取 --series-* 是同一个做法：canvas 不参与 CSS 变量解析，
 *     不重取的话切完主题画布颜色不会跟着变。
 *
 *  3. **用户不想动，它就不动。** prefers-reduced-motion 下只画一帧静止的图；
 *     标签页切到后台时停掉 rAF —— 一层背景装饰没有任何理由在后台烧 CPU。
 *
 *  4. **它够不着任何东西。** pointer-events:none 写在 CSS 里，图上没有可点、
 *     可聚焦、可读的内容，所以标记成 aria-hidden="true"。
 *
 * 必须在 .viz-root 内部加载：那套 token 定义在 .viz-root 上，不在 :root 上。
 */
(function () {
  const cv = document.getElementById('bg-canvas');
  if (!cv || !cv.getContext) return;
  const ctx = cv.getContext('2d');
  if (!ctx) return;

  /* --- 参数：都是「够用就停」的量，不是可调项 --- */
  const AREA_PER_NODE = 26000;   // 每 26000 CSS px² 一个节点
  const MAX_NODES = 90;          // 1440×900 约 50 个；4K 也不会翻上去
  const MIN_NODES = 14;          // 很小的窗口也得有东西可看
  const LINK_PX = 170;           // 连线的距离阈值
  const SPEED = 0.55;            // 节点速度上限 px/秒 —— 是氛围，不是动画
  const CURSOR_PX = 200;         // 指针的影响半径
  const HOT_RATIO = 0.14;        // 有多少节点吃第二个颜色
  const BUCKETS = 3;             // 连线的透明度分档数（见 paint 里的说明）

  /* --- 星光 --- */
  const STAR_AREA = 6500;        // 每 6500 CSS px² 一颗（1440×900 约 200 颗）
  const STAR_MAX = 320;
  const STAR_MIN = 50;
  const SPARKLE_RATIO = 0.09;    // 其中 9% 带十字耀斑
  const STAR_BUCKETS = 6;        // 星光亮度分档数（见 paintStars）
  const STAR_DRIFT = 3.2;        // 星星的漂移速度 px/秒
  const HALO_MIN = 0.62;         // 亮过这个值才画那圈柔光

  const motionOK = !window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  let nodes = [];
  let stars = [];
  let w = 0;
  let h = 0;
  let dpr = 1;
  let raf = 0;
  let last = 0;
  let elapsed = 0;               // 累计秒数：星光的相位靠它推
  let colors = null;
  const pointer = { x: -1e5, y: -1e5 };
  // 每帧算一遍亮度，核心/柔光/耀斑三个循环共用，不重复算三次
  const starBrightness = new Float32Array(STAR_MAX);

  /** 读当前主题下这几个颜色。canvas 认不得 var()，只能自己去问计算样式。 */
  function readColors() {
    const root = document.querySelector('.viz-root');
    if (!root) return;
    const cs = getComputedStyle(root);
    const g = (n, fallback) => cs.getPropertyValue(n).trim() || fallback;
    colors = {
      node: g('--net-node', 'rgba(120, 180, 255, 0.45)'),
      hot: g('--net-hot', 'rgba(160, 140, 255, 0.6)'),
      link: g('--net-link', 'rgba(120, 180, 255, 0.13)'),
      star: g('--star', 'rgba(230, 245, 255, 0.9)'),
      starHalo: g('--star-halo', 'rgba(34, 211, 238, 0.45)'),
      starHot: g('--star-hot', 'rgba(167, 139, 250, 0.85)'),
    };
  }

  function makeNode() {
    const a = Math.random() * Math.PI * 2;
    const s = SPEED * (0.35 + Math.random() * 0.65);
    return {
      x: Math.random() * w,
      y: Math.random() * h,
      vx: Math.cos(a) * s,
      vy: Math.sin(a) * s,
      r: 0.9 + Math.random() * 1.1,
      hot: Math.random() < HOT_RATIO,
    };
  }

  /** 一颗星。相位和角速度各自随机，所以整片天不会一起亮、一起暗。 */
  function makeStar() {
    return {
      x: Math.random() * w,
      y: Math.random() * h,
      // 半径取 pow(1.7)：分布明显偏小的一头，于是「一堆小星 + 少数几颗大的」，
      // 整片天才有了层次（均匀分布会看着像撒了一把同样的沙子）
      r: 0.6 + 1.6 * Math.pow(Math.random(), 1.7),
      ph: Math.random() * Math.PI * 2,
      sp: 0.4 + Math.random() * 1.6,           // rad/s
      base: 0.15 + Math.random() * 0.3,        // 最暗时还剩多少
      sparkle: Math.random() < SPARKLE_RATIO,
    };
  }

  /** 画布尺寸跟着视口走。节点数按面积重算，多出来的裁掉、缺的补上。 */
  function resize() {
    w = Math.max(1, window.innerWidth);
    h = Math.max(1, window.innerHeight);
    // 2 倍以上肉眼分辨不出，白烧一倍像素
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(h * dpr);

    const want = Math.round(Math.min(MAX_NODES,
      Math.max(MIN_NODES, (w * h) / AREA_PER_NODE)));
    if (nodes.length > want) nodes.length = want;
    while (nodes.length < want) nodes.push(makeNode());

    const wantStars = Math.round(Math.min(STAR_MAX,
      Math.max(STAR_MIN, (w * h) / STAR_AREA)));
    if (stars.length > wantStars) stars.length = wantStars;
    while (stars.length < wantStars) stars.push(makeStar());

    // 静止模式没有 rAF 来重画，尺寸变了得自己补一帧
    if (!motionOK) paint(elapsed);
  }

  function step(dt) {
    for (const p of nodes) {
      // 指针附近轻轻推开一点：鼠标扫过时底纹会「活」，但不抢注意力
      const dx = p.x - pointer.x;
      const dy = p.y - pointer.y;
      const d2 = dx * dx + dy * dy;
      if (d2 > 1 && d2 < CURSOR_PX * CURSOR_PX) {
        const d = Math.sqrt(d2);
        const f = (1 - d / CURSOR_PX) * 8 * dt;
        p.x += (dx / d) * f;
        p.y += (dy / d) * f;
      }
      p.x += p.vx * dt;
      p.y += p.vy * dt;
      // 出界就从对面回来：比「反弹」安静，看不见折返的那一刻
      if (p.x < -24) p.x = w + 24;
      else if (p.x > w + 24) p.x = -24;
      if (p.y < -24) p.y = h + 24;
      else if (p.y > h + 24) p.y = -24;
    }

    /* 星星只做一件事：极慢地往下沉一点、往右挪一点点，出底了就绕回上边。
       这点漂移自己看不出来，但它让整片天和「镜头在飘」的节点网络同步起来。 */
    for (const s of stars) {
      s.y += STAR_DRIFT * dt;
      s.x += STAR_DRIFT * 0.35 * dt;
      if (s.y > h + 8) { s.y = -8; s.x = Math.random() * w; }
      if (s.x > w + 8) s.x = -8;
    }
  }

  /** 一颗星此刻的亮度（0..1）：底座 + |sin| 的 2.4 次幂。
   *  取幂是关键 —— 单纯的 sin 是匀速明暗，看着像呼吸；取幂之后它大部分时间
   *  都暗着，只在峰值附近「跳」一下，那才是闪。 */
  function starBright(s, t) {
    const wv = 0.5 + 0.5 * Math.sin(t * s.sp + s.ph);
    return s.base + (1 - s.base) * Math.pow(wv, 2.4);
  }

  function paintStars(t) {
    /* 每颗星的亮度都不一样，但一条 path 只能有一个 globalAlpha ——
       所以把亮度量化成 6 档，每档一条 path：260 颗星是 6 次 fill，
       而不是 260 次（和连线分档是同一个手法）。 */
    const buckets = [];
    for (let i = 0; i < STAR_BUCKETS; i++) buckets.push([]);
    for (let i = 0; i < stars.length; i++) {
      const s = stars[i];
      const b = starBright(s, t);
      starBrightness[i] = b;
      const k = Math.min(STAR_BUCKETS - 1, Math.floor(b * STAR_BUCKETS));
      buckets[k].push(s.x, s.y, s.r);
    }
    ctx.fillStyle = colors.star;
    for (let k = 0; k < STAR_BUCKETS; k++) {
      const list = buckets[k];
      if (!list.length) continue;
      ctx.globalAlpha = (k + 0.7) / STAR_BUCKETS;
      ctx.beginPath();
      for (let i = 0; i < list.length; i += 3) {
        // 先 moveTo 再 arc：arc 会自动从当前点拉一条线过来，不先挪开就连成一片
        ctx.moveTo(list[i] + list[i + 2], list[i + 1]);
        ctx.arc(list[i], list[i + 1], list[i + 2], 0, Math.PI * 2);
      }
      ctx.fill();
    }

    /* 柔光：亮起来的星外面罩一小片冷色。
       深色主题下它是一圈青，浅色主题下它是一小片冷雾 —— 浅底上白点本身几乎
       看不见，那圈雾才是它「看得见」的原因。 */
    ctx.fillStyle = colors.starHalo;
    for (let i = 0; i < stars.length; i++) {
      const b = starBrightness[i];
      if (b < HALO_MIN) continue;
      const s = stars[i];
      ctx.globalAlpha = ((b - HALO_MIN) / (1 - HALO_MIN)) * 0.55;
      ctx.beginPath();
      ctx.arc(s.x, s.y, s.r * 2.6, 0, Math.PI * 2);
      ctx.fill();
    }

    // 十字耀斑：只有 SPARKLE_RATIO 的星有，而且只在它亮过一半的那一小段才画
    ctx.strokeStyle = colors.starHot;
    ctx.lineWidth = 1;
    for (let i = 0; i < stars.length; i++) {
      const s = stars[i];
      if (!s.sparkle) continue;
      const b = starBrightness[i];
      if (b < 0.5) continue;
      const f = (b - 0.5) / 0.5;
      const len = s.r * (2 + 4.5 * f);
      /* 十字对齐到半像素格：1px 的线落在像素缝上才是锐的，落在像素中间会被
         抗锯齿摊成两条 40% 的灰线 —— 那样就不「闪」了，只是一团糊。
         只对齐十字，星核照旧走真实坐标，所以整片天的漂移不受影响。 */
      const cx = Math.round(s.x) + 0.5;
      const cy = Math.round(s.y) + 0.5;
      ctx.globalAlpha = 0.25 + 0.75 * f;
      ctx.beginPath();
      ctx.moveTo(cx - len, cy);
      ctx.lineTo(cx + len, cy);
      ctx.moveTo(cx, cy - len);
      ctx.lineTo(cx, cy + len);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  function paint(t) {
    if (!colors) readColors();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    paintStars(t);
    ctx.lineWidth = 1;

    /* 连线：两两比对是 O(n²)，n ≤ 90 时每帧不到 4000 次距离计算，不值一提。
       透明度按距离衰减才好看，但一条 path 只能有一个 globalAlpha ——
       所以按距离分 3 档、每档一条 path：3 次 stroke 换一条连续的淡出，
       比「每条线各自 stroke」便宜一个数量级（那会是几千次 draw call）。 */
    const segs = [];
    for (let i = 0; i < BUCKETS; i++) segs.push([]);
    for (let i = 0; i < nodes.length; i++) {
      const a = nodes[i];
      for (let j = i + 1; j < nodes.length; j++) {
        const b = nodes[j];
        const dx = a.x - b.x;
        const dy = a.y - b.y;
        const d2 = dx * dx + dy * dy;
        if (d2 > LINK_PX * LINK_PX) continue;
        const q = Math.sqrt(d2) / LINK_PX;              // 0（最近）→ 1（阈值）
        const k = Math.min(BUCKETS - 1, Math.floor(q * BUCKETS));
        segs[k].push(a.x, a.y, b.x, b.y);
      }
    }
    for (let k = 0; k < BUCKETS; k++) {
      const list = segs[k];
      if (!list.length) continue;
      ctx.globalAlpha = 1 - k / BUCKETS;
      ctx.strokeStyle = colors.link;
      ctx.beginPath();
      for (let i = 0; i < list.length; i += 4) {
        ctx.moveTo(list[i], list[i + 1]);
        ctx.lineTo(list[i + 2], list[i + 3]);
      }
      ctx.stroke();
    }

    // 指针到近处节点的那几根：整层底纹里唯一的交互反馈
    ctx.globalAlpha = 1;
    ctx.strokeStyle = colors.hot;
    ctx.beginPath();
    for (const p of nodes) {
      const dx = p.x - pointer.x;
      const dy = p.y - pointer.y;
      const d2 = dx * dx + dy * dy;
      if (d2 > (CURSOR_PX * 0.55) ** 2) continue;
      ctx.moveTo(pointer.x, pointer.y);
      ctx.lineTo(p.x, p.y);
    }
    ctx.stroke();

    for (const p of nodes) {
      ctx.globalAlpha = 1;
      ctx.fillStyle = p.hot ? colors.hot : colors.node;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  }

  function loop(now) {
    const dt = Math.min(0.05, (now - last) / 1000 || 0);   // 卡顿时别一次跳很远
    last = now;
    elapsed += dt;
    step(dt);
    paint(elapsed);
    raf = requestAnimationFrame(loop);
  }

  function start() {
    if (raf) return;
    last = performance.now();
    raf = requestAnimationFrame(loop);
  }

  function stop() {
    if (!raf) return;
    cancelAnimationFrame(raf);
    raf = 0;
  }

  readColors();
  resize();

  window.addEventListener('resize', resize, { passive: true });

  if (motionOK) {
    window.addEventListener('pointermove', (ev) => {
      pointer.x = ev.clientX;
      pointer.y = ev.clientY;
    }, { passive: true });
    // 指针离开窗口（relatedTarget 为 null）就把影响收掉，
    // 否则那簇亮线会永远钉在最后停住的位置上
    window.addEventListener('mouseout', (ev) => {
      if (!ev.relatedTarget) { pointer.x = -1e5; pointer.y = -1e5; }
    }, { passive: true });
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) stop(); else start();
    });
    start();
  }

  /* 主题：data-theme 是页面按钮写的，系统偏好是另一条路，两条都得跟 */
  const onTheme = () => { readColors(); if (!motionOK) paint(elapsed); };
  new MutationObserver(onTheme).observe(document.documentElement,
    { attributes: true, attributeFilter: ['data-theme'] });
  const mq = window.matchMedia('(prefers-color-scheme: dark)');
  if (mq.addEventListener) mq.addEventListener('change', onTheme);
})();
