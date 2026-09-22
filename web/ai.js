'use strict';

/* AI 助手 —— 抽屉 UI + 动作总线
 *
 * 分工很清楚：
 *   服务端（ai.py / ai_tools.py）管配置、调网关、跑 agent 循环、本地兜底。
 *   这里只管两件事：把答复画出来；把助手点名要做的**前端动作**在真界面上执行掉。
 *
 * 三条硬规矩，改这个文件之前先读一遍：
 *
 *  1. 助手正文一律 textContent，**绝不 innerHTML**。模型输出和用户输入都是不可信
 *     内容，一次提示注入就能变成 HTML 注入。所以正文只按纯文本排版（CSS 的
 *     white-space: pre-wrap 就能保住换行），不解析 Markdown。
 *
 *  2. 前端工具的实参**先按白名单清洗再落地**。参数是从模型来的外部输入，
 *     不认识的键一律丢掉，数值一律过类型和范围，永远不 eval。
 *
 *  3. 导出 PNG / CSV、复制表格**不是动作**。那几个会落盘或进剪贴板，留给人点按钮；
 *     助手只能说「点 3D 卡片上的 PNG」。这条是产品决定，不是没来得及做。
 *
 * 界面这一侧的实现要点：所有 setter 都调 app.js / chain3d.js 末尾暴露的那两个
 * 接口（window.fjcApp / window.fjc3d），**不直接改 DOM** —— 直接改 value 图不会重画。
 * 扫描卡片由另一个会话写（sweep.js），它对外什么都不暴露，所以只能通过 DOM 驱动：
 * 填好输入框、点 #s-run、等它自己把按钮解禁。这条也符合「走界面自己的路」。
 */

(function () {

  const byId = (id) => document.getElementById(id);
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  // 浏览器这一侧最多替助手连着执行几批界面动作。服务端有 6 轮的硬上限，
  // 这里再守一道：万一某个动作一直返回「没成功」，也不至于两边来回刷个没完。
  const MAX_ACTION_ROUNDS = 4;
  // 等扫描那种慢动作的上限。sweep 走的是界面预算，最坏约 4 秒，60 秒足够宽。
  const WAIT_TOOL_MS = 60000;

  const NO_APP = { ok: false, error: '这个页面版本没有暴露主参数接口，改不了 n / l' };
  const NO_3D = { ok: false, error: '这个页面版本没有暴露 3D 接口，改不了 3D 卡片' };
  const NO_SWEEP = { ok: false, error: '扫描卡片暂不可用（页面上找不到它的控件）' };

  const els = {};
  let sessionId = null;
  let busy = false;
  let stopped = false;
  let ctl = null;      // 当前请求的 AbortController
  let cfg = null;      // /api/ai/status 的返回

  /* ---------------- 实参清洗 ----------------
   *
   * spec 的每一项：{ kind, min, max, gt, clamp, maxLen }
   *   kind   'int' | 'num' | 'bool' | 'intArray'
   *   min/max 边界，默认**越界就拒**（那是真不合法的参数）
   *   gt     下界（开区间），比如 l > 0
   *   clamp  true 改成夹到边界而不是拒 —— 留给「夹一下更顺手」的量，比如视角
   *
   * 故意**不给 n 设上限**：上限是后端的事（/api/chain 有绘制预算，/api/compute 有
   * 自己的校验）。在这里再抄一份等于把同一个数字维护两处，抄歪了反而更难查；
   * 后端拒绝时那句中文错误会被下面的 afterAlerts 原样报回给模型。
   */
  function clean(args, spec) {
    const src = (args && typeof args === 'object' && !Array.isArray(args)) ? args : {};
    const out = {};
    for (const key of Object.keys(spec)) {
      const v = src[key];
      if (v === undefined || v === null) continue;
      const s = spec[key];

      if (s.kind === 'bool') {
        out[key] = !!v;
        continue;
      }

      if (s.kind === 'intArray') {
        if (!Array.isArray(v) || v.length === 0) return { error: `${key} 必须是非空数组` };
        const arr = [];
        for (const item of v) {
          const x = Number(item);
          if (!isFinite(x) || Math.floor(x) !== x || x < 1) {
            return { error: `${key} 里每一项都得是 ≥ 1 的整数` };
          }
          arr.push(x);
        }
        if (s.maxLen && arr.length > s.maxLen) return { error: `${key} 最多 ${s.maxLen} 个` };
        out[key] = arr;
        continue;
      }

      let x = Number(v);
      if (!isFinite(x)) return { error: `${key} 必须是数字` };
      if (s.kind === 'int') {
        x = Math.floor(x);
        if (s.min !== undefined && x < s.min) {
          if (!s.clamp) return { error: `${key} 不能小于 ${s.min}` };
          x = s.min;
        }
      } else {
        if (s.gt !== undefined && x <= s.gt) {
          if (!s.clamp) return { error: `${key} 必须大于 ${s.gt}` };
          x = s.gt;
        }
        if (s.min !== undefined && x < s.min) {
          if (!s.clamp) return { error: `${key} 不能小于 ${s.min}` };
          x = s.min;
        }
      }
      if (s.max !== undefined && x > s.max) {
        if (!s.clamp) return { error: `${key} 不能大于 ${s.max}` };
        x = s.max;
      }
      out[key] = x;
    }
    return { value: out };
  }

  /** 读某个提示条里当前显示的错误。界面自己报的错比我们猜的准。 */
  function uiError(boxId) {
    const box = byId(boxId);
    if (!box) return '';
    const err = box.querySelector('.alert-error');
    if (!err) return '';
    const spans = err.querySelectorAll('span');
    const last = spans.length ? spans[spans.length - 1].textContent : err.textContent;
    return (last || '').trim();
  }

  /**
   * 动作跑完之后看一眼界面有没有报错。
   * 设成 `ok: false` 很关键：模型拿到「界面把它拒了」才会换个说法重试，
   * 而拿到一个假的 ok:true 会顺着错的前提继续往下讲。
   */
  function afterAlerts(r, boxId) {
    if (!r) return { ok: false, error: '界面动作没有返回结果' };
    if (r.ok === false) return r;
    const err = uiError(boxId);
    if (err) return { ok: false, error: err };
    return r;
  }

  function pageState() {
    const st = {};
    if (window.fjcApp) { try { st.main = window.fjcApp.getState(); } catch (e) { st.main = null; } }
    if (window.fjc3d) { try { st.three_d = window.fjc3d.getState(); } catch (e) { st.three_d = null; } }
    const sn = byId('s-n');
    if (sn) {
      const sum = byId('sweep-summary');
      st.sweep = {
        n_input: sn.value,
        chains_input: byId('s-chains') ? byId('s-chains').value : null,
        seed_input: byId('s-seed') ? byId('s-seed').value : null,
        // 「汇总 CSV」按钮只有跑出结果后才会解禁，拿它当「有没有结果」的判据
        has_result: !!(byId('s-csv') && !byId('s-csv').disabled),
        summary: sum ? sum.textContent.replace(/\s+/g, ' ').trim().slice(0, 600) : '',
      };
    }
    return st;
  }

  /* ---------------- 动作表 ----------------
   * 键名必须和 ai_tools.py 里的前端工具一一对应。少一个不会崩：服务端会收到
   * 「这个界面不认识 xxx」然后改用别的办法 —— 但别指望它猜得出来。
   */
  const ACTIONS = {

    get_page_state() {
      return { ok: true, data: pageState() };
    },

    async set_params(args) {
      if (!window.fjcApp) return NO_APP;
      const c = clean(args, { n: { kind: 'int', min: 1 }, l: { kind: 'num', gt: 0 } });
      if (c.error) return { ok: false, error: c.error };
      return afterAlerts(await window.fjcApp.setParams(c.value), 'alerts');
    },

    async set_curves(args) {
      if (!window.fjcApp) return NO_APP;
      // 5 这条是界面的显示约定（服务端 MAX_CURVES 也是 5），拦在前面省一次往返
      const c = clean(args, { ns: { kind: 'intArray', maxLen: 5 } });
      if (c.error) return { ok: false, error: c.error };
      return afterAlerts(await window.fjcApp.setCurves(c.value), 'alerts');
    },

    async set_normalize(args) {
      if (!window.fjcApp) return NO_APP;
      const c = clean(args, { on: { kind: 'bool' } });
      if (c.error) return { ok: false, error: c.error };
      if (!('on' in c.value)) return { ok: false, error: '没给 on' };
      return afterAlerts(await window.fjcApp.setNormalize(c.value), 'alerts');
    },

    async set_chain_params(args) {
      if (!window.fjc3d) return NO_3D;
      // 1~5 条是 3D 视图的显示约定（第 i 条用第 i 个系列色槽），拦在前面
      const c = clean(args, {
        n: { kind: 'int', min: 1 },
        chains: { kind: 'int', min: 1, max: 5 },
        seed: { kind: 'int', min: 0 },
      });
      if (c.error) return { ok: false, error: c.error };
      return afterAlerts(await window.fjc3d.setChainParams(c.value), 'chain-alerts');
    },

    set_chain_view(args) {
      if (!window.fjc3d) return NO_3D;
      // 视角这几个夹一下比拒掉好：模型说「仰角 120 度」时它想要的大概就是「从上往下看」
      const c = clean(args, {
        azim: { kind: 'num' },
        elev: { kind: 'num', min: -89, max: 89, clamp: true },
        zoom: { kind: 'num', gt: 0, min: 0.4, max: 4, clamp: true },
      });
      if (c.error) return { ok: false, error: c.error };
      return window.fjc3d.setView(c.value);
    },

    play_chain() {
      if (!window.fjc3d) return NO_3D;
      return window.fjc3d.play();
    },

    set_sweep_params(args) {
      const sn = byId('s-n');
      if (!sn) return NO_SWEEP;
      const c = clean(args, {
        ns: { kind: 'intArray', maxLen: 20 },
        chains: { kind: 'int', min: 1 },
        seed: { kind: 'int', min: 0 },
      });
      if (c.error) return { ok: false, error: c.error };

      const touched = [];
      if (c.value.ns) { sn.value = c.value.ns.join(', '); touched.push('ns'); }
      if (c.value.chains !== undefined) {
        byId('s-chains').value = String(c.value.chains);
        touched.push('chains');
      }
      if (c.value.seed !== undefined) {
        byId('s-seed').value = String(c.value.seed);
        touched.push('seed');
      }
      if (!touched.length) return { ok: false, error: '没给任何扫描参数' };

      // 扫描卡片只在点「运行模拟」时读这些输入框，所以写进去就已经生效了；
      // 但图上画的还是上一次扫的 —— 这一点必须说清楚，不然会被当成已经重扫过。
      return {
        ok: true,
        data: {
          ...pageState().sweep,
          note: '参数已填好，但图还是上一次扫的结果 —— 要重扫请调 run_sweep_view。',
        },
      };
    },

    async run_sweep_view() {
      const btn = byId('s-run');
      if (!btn) return NO_SWEEP;
      if (btn.disabled) return { ok: false, error: '扫描正在跑，等它跑完再点' };

      btn.click();
      // run() 是 async，但它到第一个 await 之前是同步的：点下去之后按钮要么已经被
      // 禁用（真的开始跑了），要么它当场就返回了（参数不合法，提示条上已经有错误）。
      if (!btn.disabled) {
        const err = uiError('sweep-alerts');
        return { ok: false, error: err || '扫描没能启动' };
      }

      const t0 = Date.now();
      while (btn.disabled && Date.now() - t0 < WAIT_TOOL_MS) await sleep(120);
      if (btn.disabled) {
        return { ok: false, error: '扫描超过 60 秒还没结束，先不等了（它还在后台跑）' };
      }
      const err = uiError('sweep-alerts');
      if (err) return { ok: false, error: err };

      const sum = byId('sweep-summary');
      return {
        ok: true,
        data: {
          note: '扫描已完成，结果画在扫描卡片上',
          summary: sum ? sum.textContent.replace(/\s+/g, ' ').trim().slice(0, 800) : '',
        },
      };
    },
  };

  /** 一批动作顺序执行。**顺序很重要** —— 模型可能先 set_params 再 get_page_state。 */
  async function runActions(actions) {
    const out = [];
    for (const a of actions) {
      const fn = ACTIONS[a.tool];
      let r;
      if (!fn) {
        r = { ok: false, error: `这个界面不认识 ${a.tool} 这个动作` };
      } else {
        try {
          r = await fn(a.args || {});
        } catch (e) {
          r = { ok: false, error: `执行 ${a.tool} 时出错：${(e && e.message) || e}` };
        }
      }
      out.push({
        tool: a.tool,
        ok: !!(r && r.ok),
        data: r && r.data,
        error: r && r.error,
      });
    }
    return out;
  }

  /* ---------------- 渲染 ----------------
   * 全程 createElement + textContent。这个文件里不该出现 innerHTML，
   * 也不该出现 insertAdjacentHTML / document.write / new Function / eval。
   */

  function el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = String(text);
    return n;
  }

  function fmtArgs(a) {
    if (!a || typeof a !== 'object') return '';
    return Object.keys(a).map((k) => {
      const v = a[k];
      return `${k}=${Array.isArray(v) ? `[${v.join(', ')}]` : String(v)}`;
    }).join(', ');
  }

  function fmtStep(s) {
    const a = fmtArgs(s.args);
    return `${s.tool}(${a})`;
  }

  function stepsRow(steps) {
    const box = el('details', 'ai-steps');
    box.appendChild(el('summary', null, `执行：${steps.map(fmtStep).join('、')}`));
    const ul = el('ul');
    steps.forEach((s) => {
      let tail = '';
      if (s.ok === false) tail = `　✕ ${s.error || '没成功'}`;
      else if (s.deferred) tail = '　（由界面执行）';
      ul.appendChild(el('li', s.ok === false ? 'bad' : null, fmtStep(s) + tail));
    });
    box.appendChild(ul);
    return box;
  }

  function addMsg(who, build) {
    const wrap = el('div', `ai-msg ${who}`);
    build(wrap);
    els.msgs.appendChild(wrap);
    els.msgs.scrollTop = els.msgs.scrollHeight;
  }

  function addUser(text) {
    addMsg('user', (w) => w.appendChild(el('div', 'ai-bubble', text)));
  }

  /** 一条助手消息：红字说明（可选）→ 正文 → 「执行」折叠行 → 状态标签。 */
  function addBot(b) {
    addMsg('bot', (w) => {
      if (b.error) w.appendChild(el('div', 'ai-note err', b.error));
      if (b.answer) w.appendChild(el('div', 'ai-bubble', b.answer));
      if (b.steps && b.steps.length) w.appendChild(stepsRow(b.steps));
      if (b.tag) w.appendChild(el('div', b.tagErr ? 'ai-tag err' : 'ai-tag', b.tag));
    });
  }

  function tagFor(b) {
    if (b.source === 'local') {
      return b.degraded ? `本地模式（${b.degraded}）` : '本地模式';
    }
    if (b.source === 'llm') {
      return `${b.model || (cfg && cfg.model) || '模型网关'} · 网关`;
    }
    return '来源不明';
  }

  /** 把一次 /api/ai/ask 或 /api/ai/resume 的返回画出来。 */
  function render(b) {
    const out = { steps: b.steps || [], source: b.source, model: b.model };
    if (b.answer) out.answer = b.answer;
    if (b.detail) out.answer = out.answer ? `${out.answer}\n\n${b.detail}` : b.detail;
    if (!b.ok && b.error) {
      out.error = b.error;
      out.tagErr = true;
    }
    out.tag = tagFor(b);
    addBot(out);
  }

  /* ---------------- 传输 ---------------- */

  async function postJson(url, body, signal) {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal,
    });
    let data = null;
    try { data = await res.json(); } catch (e) { data = null; }
    // 解析不出 JSON 才算真失败。4xx/5xx 只要带了 JSON body 就交给调用方 ——
    // 网关挂掉时那段本地兜底的答复就在 body 里，它是要显示给用户看的答案；
    // 接入设置那两个路由的字段校验错误（400）同理，正文里就是那句中文原因。
    if (data === null || typeof data !== 'object') {
      throw new Error(`服务器返回了无法解析的内容（HTTP ${res.status}）`);
    }
    return data;
  }

  /** 对话用的那条：挂上 ctl，好让「停止」能 abort。 */
  async function post(url, body) {
    ctl = new AbortController();
    return postJson(url, body, ctl.signal);
  }

  /**
   * 把实际执行结果并回服务端给的那份 steps。
   * 服务端**不知道**前端动作成没成 —— 所以在浏览器执行完之后，那份 steps 里
   * 关于前端工具的 ok 全是猜的。不校正的话「执行」那一行会报喜不报忧。
   * 同名工具按出现顺序配对（同一轮里出现两次 set_params 是可能的）。
   */
  function mergeSteps(steps, actions, results) {
    const pool = new Map();
    actions.forEach((a, i) => {
      if (!pool.has(a.tool)) pool.set(a.tool, []);
      pool.get(a.tool).push(results[i]);
    });
    const cursor = new Map();
    return (steps || []).map((s) => {
      const q = pool.get(s.tool);
      const at = cursor.get(s.tool) || 0;
      if (!q || at >= q.length) return s;
      cursor.set(s.tool, at + 1);
      const r = q[at];
      return { ...s, ok: !!(r && r.ok), error: r && r.error };
    });
  }

  /** 处理一次返回。碰上"要我动界面"就执行完再回调服务端，直到它把话说完整。 */
  async function handle(first) {
    let b = first;
    if (b.session_id) sessionId = b.session_id;

    // 网关那条路：模型先要动界面，执行完再把结果交回去让它接着说。
    for (let i = 0; i < MAX_ACTION_ROUNDS && b.status === 'actions'; i++) {
      if (stopped) return;
      const acts = Array.isArray(b.actions) ? b.actions : [];
      setPhase(`正在操作界面…（${acts.length} 个动作：${acts.map((a) => a.tool).join('、')}）`);
      const results = await runActions(acts);
      if (stopped) return;

      setPhase('正在把执行结果交回模型…');
      b = await post('/api/ai/resume', { session_id: sessionId, results });
      if (b.session_id) sessionId = b.session_id;
    }

    if (b.status === 'actions') {
      // 循环被自己的上限掐断，手上这份 body 里没有答复 —— 如实说，别硬编一句话出来
      addBot({
        error: `连着执行了 ${MAX_ACTION_ROUNDS} 批界面动作，助手还想继续调。先停在这里 ——`
          + '界面上已经改过的参数都是真的，接着问一句就能继续。',
        steps: b.steps || [],
        tag: '已到动作上限',
        tagErr: true,
      });
      return;
    }

    // 本地模式那条路：它一次就把话说完了（status 是 final），但同样会把「要动界面」
    // 的动作一起交出来。这些动作照样得执行 —— 不执行的话，「把 n 改成 400」只会回一句
    // 「已改」而界面纹丝不动，那是最糟的一种错。这里没有第二轮可回，所以执行完只把
    // 真实结果并进 steps，不调 resume。
    if (Array.isArray(b.actions) && b.actions.length) {
      setPhase('正在操作界面…');
      const results = await runActions(b.actions);
      b = { ...b, steps: mergeSteps(b.steps, b.actions, results) };
    }

    render(b);
  }

  /* ---------------- 发送 ---------------- */

  async function send() {
    if (busy) return;
    const text = els.input.value.trim();
    if (!text) return;

    addUser(text);
    els.input.value = '';
    autosize();

    stopped = false;
    setBusy(true);
    setPhase('正在思考…');
    try {
      const b = await post('/api/ai/ask', { message: text, session_id: sessionId });
      if (b.session_id) sessionId = b.session_id;
      await handle(b);
    } catch (e) {
      if (e && e.name === 'AbortError') {
        addBot({ error: '已经停下了。（服务端那一轮可能还在跑，但它不会再往这里写东西。）',
                 tag: '已中止' });
      } else {
        addBot({ error: (e && e.message) || String(e), tag: '请求失败', tagErr: true });
      }
    } finally {
      setPhase('');
      setBusy(false);
    }
  }

  function stop() {
    stopped = true;
    if (ctl) ctl.abort();
  }

  function setPhase(text) {
    if (text) {
      els.phase.textContent = text;
      els.phase.hidden = false;
    } else {
      els.phase.hidden = true;
      els.phase.textContent = '';
    }
  }

  function setBusy(on) {
    busy = on;
    els.send.disabled = on;
    els.stop.disabled = !on;
    els.input.disabled = on;
    if (!on) ctl = null;
  }

  function autosize() {
    // 最多长到 5 行左右，再多就内部滚动
    els.input.style.height = 'auto';
    els.input.style.height = `${Math.min(120, els.input.scrollHeight)}px`;
  }

  /* ---------------- 开合 ---------------- */

  function setOpen(open) {
    els.drawer.hidden = !open;
    els.toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    // 给页面让出右边缘（样式里这条只在 ≥960px 时生效）。
    // 不让位的话抽屉正好盖住 3D 卡片右侧，而「改个参数、眼看图重画」正是最常用的用法。
    const root = document.querySelector('.viz-root');
    if (root) root.classList.toggle('ai-open', open);
    if (open) els.input.focus();
  }

  /* ---------------- 配置显示 ---------------- */

  const GREET_LLM = [
    '我可以调用这个软件的全部功能：算统计量、对比多个 n、模拟单链、跑标度扫描，',
    '也能直接改界面参数（n、l、归一化、3D 参数与视角）并把图重画。',
    '',
    '试试：「n=400 的 h_rms 是多少」「把 n 改成 400 并打开归一化」「画一条 n=500 的链」',
    '「100 和 1000 比一下」「扫一下标度关系」。',
    '',
    '导出 PNG / CSV、复制表格不在我这儿 —— 那几个要落盘，留给你自己点按钮。',
  ].join('\n');

  const GREET_LOCAL = [
    '现在运行在本地模式：没有配置模型网关，我按关键词认几种固定问法，',
    '答案里的数字全部来自这个软件自己的计算内核，不是我又算了一遍。',
    '',
    '能认的：「n=400 的 h_rms 是多少」「把 n 改成 400」「100 和 1000 比一下」',
    '「打开归一化」「画一条 n=500 的链」「扫一下标度关系」「播放生长动画」。',
    '',
    '认不出的问题我会直说认不出，不猜数字。想让它能自由对话，',
    '点抽屉里的「接入设置」，填一个 OpenAI 兼容网关的地址和 key，保存即可，不用重启。',
  ].join('\n');

  function renderCfg() {
    if (!cfg) {
      els.mode.textContent = '';
      els.cfgEl.textContent = '';
      return;
    }
    if (cfg.configured) {
      els.mode.textContent = `${cfg.model} · 网关`;
      els.cfgEl.textContent = `密钥 ${cfg.masked || '已配置'}（${cfg.source === 'env' ? '环境变量' : '配置文件'}）`;
    } else {
      els.mode.textContent = '本地模式';
      els.cfgEl.textContent = '未配置模型网关';
    }
  }

  /** 开场白：先说清楚它现在能干什么、不能干什么。 */
  function greet() {
    const configured = !!(cfg && cfg.configured);
    addMsg('bot', (w) => {
      w.appendChild(el('div', 'ai-bubble', configured ? GREET_LLM : GREET_LOCAL));
      w.appendChild(el('div', 'ai-tag', configured ? '已连上网关' : '本地模式 · 无需密钥'));
    });
  }

  async function loadStatus() {
    try {
      const res = await fetch('/api/ai/status');
      cfg = await res.json();
    } catch (e) {
      cfg = null;
    }
    renderCfg();
    fillProviders();
    greet();
  }

  /* ---------------- 接入设置 ----------------
   *
   * 这一块**不是**助手的工具：模型改不了网关配置。能改的话，一次提示注入
   * （页面状态里、工具结果里塞一句话）就能把 endpoint 指到攻击者那边，
   * 下一问连同密钥一起发过去。所以它只走用户点按钮这条路，也不在 ACTIONS 表里 ——
   * test_ai.py 里有一条断言专门盯着「配置表单进不了动作表」。
   *
   * 密钥那栏三条，改之前想清楚：
   *   · `type="password"`，而且**从不回填**。页面上只有掩码，把掩码当值填进去的话，
   *     用户一进表单点保存就会把真 key 覆盖成 ••••3333 —— 那是个无声的自伤。
   *   · 留空的语义是「**不改**」（服务端也是这么实现的），要清得点「清除密钥」。
   *   · 保存成功就清空输入框，明文别留在 DOM 里。
   */

  function setSetupOpen(open) {
    if (!els.setup) return;
    els.setup.hidden = !open;
    [els.setupBtn, els.setupOpen].forEach((b) => {
      if (b) b.setAttribute('aria-expanded', open ? 'true' : 'false');
    });
    if (open) {
      fillSetupForm();
      (els.fPreset || els.fEndpoint || els.setup).focus();
    }
  }

  function setSetupMsg(text, kind) {
    if (!els.fMsg) return;
    els.fMsg.textContent = text || '';
    els.fMsg.className = 'ai-setup-msg' + (kind ? ` ${kind}` : '');
  }

  /** 把服务端当前的配置回填进表单。密钥**只回填掩码到 placeholder**，value 永远清空。 */
  function fillSetupForm() {
    if (!els.fEndpoint) return;
    els.fEndpoint.value = (cfg && cfg.endpoint) || '';
    els.fModel.value = (cfg && cfg.model) || '';
    if (els.fKey) {
      els.fKey.value = '';
      els.fKey.placeholder = (cfg && cfg.masked)
        ? `已保存 ${cfg.masked}　留空则不改`
        : 'sk-…';
    }
    if (els.preset) els.preset.value = '';

    if (cfg && Array.isArray(cfg.warnings) && cfg.warnings.length) {
      setSetupMsg(cfg.warnings.join('\n'), 'warn');
    } else if (cfg && cfg.configured) {
      setSetupMsg(`当前：${cfg.model}（${cfg.source === 'env' ? '环境变量' : '配置文件'}）`, '');
    } else {
      setSetupMsg((cfg && cfg.note) || '还没配网关，助手跑在本地模式。', '');
    }
  }

  /** 「选择服务商」下拉的数据来自 /api/ai/status 的 providers —— 名单只有一份（ai.py），
   *  不在这里写死，免得和 README 那张表对不上。 */
  function fillProviders() {
    const sel = els.preset;
    if (!sel || sel.options.length > 1) return;   // 只填一次（loadStatus 可能重跑）
    (cfg && cfg.providers || []).forEach((p) => {
      if (!p || !p.endpoint) return;
      const o = document.createElement('option');
      o.value = p.endpoint;
      o.textContent = p.name || p.endpoint;
      o.dataset.model = p.model || '';
      sel.appendChild(o);
    });
  }

  function onPreset() {
    const opt = els.preset && els.preset.selectedOptions[0];
    if (!opt || !opt.value) return;
    els.fEndpoint.value = opt.value;
    if (opt.dataset.model) els.fModel.value = opt.dataset.model;
    setSetupMsg('已填入该服务商的 base 地址和模型，按需再改。密钥要自己粘。', '');
  }

  /** 保存后的统一收尾：换掉 cfg、刷新状态条、清空明文、把服务端的告警一起显示。 */
  function afterConfigSaved(data, headline) {
    cfg = data;
    if (els.fKey) els.fKey.value = '';
    renderCfg();
    fillSetupForm();
    const w = (data.warnings || []).join('\n');
    setSetupMsg(headline + (w ? `\n${w}` : ''), w ? 'warn' : 'ok');
  }

  async function saveSetup() {
    if (!els.fEndpoint) return;
    setSetupMsg('正在保存…', '');
    try {
      const data = await postJson('/api/ai/config', {
        endpoint: els.fEndpoint.value.trim(),
        model: els.fModel.value.trim(),
        key: els.fKey.value,          // 空串 = 服务端不改
      });
      if (data.error) { setSetupMsg(data.error, 'err'); return; }
      afterConfigSaved(
        data,
        data.configured
          ? '已保存到 ai_config.json，下一句提问就走网关了（不用重启）。'
          : '已保存，但三样还没配齐 —— 现在仍是本地模式。',
      );
    } catch (e) {
      setSetupMsg((e && e.message) || String(e), 'err');
    }
  }

  async function testSetup() {
    if (!els.fEndpoint) return;
    const body = {
      endpoint: els.fEndpoint.value.trim(),
      model: els.fModel.value.trim(),
    };
    // 密钥那栏**空着就不发** —— 服务端会退回去用已保存的那个。
    // 发个空串的话，「已经存好了、进表单直接点测试」会误报「三样要齐了」。
    if (els.fKey.value) body.key = els.fKey.value;

    setSetupMsg('正在连网关…（最多 20 秒）', '');
    try {
      const data = await postJson('/api/ai/test', body);
      if (data.ok) {
        setSetupMsg(`通了 · ${data.model} · ${data.ms} ms\n${data.endpoint}`, 'ok');
      } else {
        setSetupMsg(data.error || '连不上', 'err');
      }
    } catch (e) {
      setSetupMsg((e && e.message) || String(e), 'err');
    }
  }

  async function clearKey() {
    setSetupMsg('正在清除…', '');
    try {
      const data = await postJson('/api/ai/config', { clear_key: true });
      if (data.error) { setSetupMsg(data.error, 'err'); return; }
      afterConfigSaved(
        data,
        data.configured
          ? '密钥已从 ai_config.json 移除（但环境变量里还有一个，当前仍算已配置）。'
          : '密钥已移除，助手回到本地模式。',
      );
    } catch (e) {
      setSetupMsg((e && e.message) || String(e), 'err');
    }
  }

  /* ---------------- 启动 ---------------- */

  function init() {
    els.drawer = byId('ai-drawer');
    els.msgs = byId('ai-msgs');
    els.toggle = byId('ai-toggle');
    els.close = byId('ai-close');
    els.input = byId('ai-input');
    els.send = byId('ai-send');
    els.stop = byId('ai-stop');
    els.phase = byId('ai-phase');
    els.mode = byId('ai-mode');
    els.cfgEl = byId('ai-cfg');
    els.setup = byId('ai-setup');
    els.setupBtn = byId('ai-setup-btn');
    els.setupOpen = byId('ai-setup-open');
    els.preset = byId('ai-f-preset');
    els.fEndpoint = byId('ai-f-endpoint');
    els.fModel = byId('ai-f-model');
    els.fKey = byId('ai-f-key');
    els.fSave = byId('ai-f-save');
    els.fTest = byId('ai-f-test');
    els.fClear = byId('ai-f-clear');
    els.fMsg = byId('ai-f-msg');
    if (!els.drawer || !els.msgs || !els.toggle) return;   // 页面里没有抽屉，安静退出

    els.toggle.addEventListener('click', () => setOpen(els.drawer.hidden));
    els.close.addEventListener('click', () => setOpen(false));
    els.send.addEventListener('click', send);
    els.stop.addEventListener('click', stop);

    // 接入设置：头部那颗 ⚙ 和底部状态条旁边那颗，开的是同一个面板。
    // 底下逐个判空 —— 页面标记要是少了几件，表单整块不工作，但对话照常能用。
    if (els.setup && els.setupBtn) {
      els.setupBtn.addEventListener('click', () => setSetupOpen(els.setup.hidden));
      if (els.setupOpen) {
        els.setupOpen.addEventListener('click', () => setSetupOpen(els.setup.hidden));
      }
      if (els.fSave) els.fSave.addEventListener('click', saveSetup);
      if (els.fTest) els.fTest.addEventListener('click', testSetup);
      if (els.fClear) els.fClear.addEventListener('click', clearKey);
      if (els.preset) els.preset.addEventListener('change', onPreset);
    }

    els.input.addEventListener('input', autosize);
    els.input.addEventListener('keydown', (ev) => {
      if (ev.key === 'Enter' && !ev.shiftKey) {
        ev.preventDefault();
        send();
      }
    });

    // Esc 收抽屉。绑在 document 上而不是输入框上：焦点在哪儿都该管用。
    // 先收设置面板、再收抽屉 —— 一次只收一层，否则面板开着按 Esc 会把整个抽屉
    // 带走，用户以为自己只是关了个表单。
    document.addEventListener('keydown', (ev) => {
      if (ev.key !== 'Escape' || els.drawer.hidden) return;
      if (els.setup && !els.setup.hidden) { setSetupOpen(false); return; }
      setOpen(false);
    });

    loadStatus();
  }

  init();
})();
