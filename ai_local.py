"""本地兜底：没配网关（或网关挂了）时，按关键词和数字认意图。

**它不是一个小号模型，也不做任何自由推理。** 它是一张关键词表加几个已注册的
工具 —— 认得出的问题就调工具、拿真实数字回答，认不出就明说认不出。

「认不出就明说」是刻意的：宁可说「我不会」，也不能给一个编造的数字。
所以这里的每一条回答，数字都来自 ai_tools → fjc_core，没有一条是这里算的。

和 LLM 那条路的两个区别：
  · 可以同时命中多个意图（「把 n 改成 400，并打开归一化」会两个都做），
    因为关键词匹配没有「谁更重要」这回事，能认出来的都做掉反而更符合预期。
  · 不返回 status:"actions" 那种待执行状态：没有模型需要看结果，
    所以动作随回答一起交出去，前端先执行、再显示文字。
"""

from __future__ import annotations

import math
import re
import zlib
from dataclasses import dataclass, field

import ai_tools


@dataclass
class LocalReply:
    """本地模式的一次回答。"""

    text: str
    actions: list[dict] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)
    matched: bool = False   # 认出来了没有。认不出时 ai.py 会改用报错文案


# --- 数字与意图的抽取 ----------------------------------------------------

# 「3d」「3D」里的 3 不是参数，扫数字前先抠掉，否则「3D 视图」会被读成 n=3
_NOISE = re.compile(r"3\s*[dD](?![A-Za-z])")

_NUM_RE = re.compile(r"\d+(?:\.\d+)?")

# n 的提示：n=400 / 链段数 400 / n 改成 400。
# 前后加字母边界，否则 "in 400" 里的 n 会被当成参数名。
_N_HINT = re.compile(
    r"(?<![A-Za-z])(?:n|链段数|链节数)(?![A-Za-z])\s*"
    r"(?:[=:：＝]|改成|改为|设为|设成|换成|变成|调整到|调到|取|是|为)?\s*(\d+)",
    re.I,
)
# l 的提示：l=2 / 链段长度 2 / Kuhn 长度 1.5
_L_HINT = re.compile(
    r"(?<![A-Za-z])(?:l|链段长度|kuhn\s*长度|库恩长度)(?![A-Za-z])\s*"
    r"(?:[=:：＝]|改成|改为|设为|设成|换成|变成|调整到|调到|取|是|为)?\s*(\d+(?:\.\d+)?)",
    re.I,
)
# 链数的提示：链数 5 / 5 条链 / chains=5
_CHAINS_HINT = re.compile(
    r"(?:链数|条数|chains?)\s*(?:[=:：＝]|改成|改为|设为|设成|换成|变成|调成)?\s*(\d+)"
    r"|(\d+)\s*条链",
    re.I,
)
# 键角：θ=109.47 / 键角 109.47 / theta 90
_THETA_HINT = re.compile(
    r"(?:θ|theta|键角)\s*(?:[=:：＝]|为|是|取|改成|设为)?\s*(\d+(?:\.\d+)?)",
    re.I,
)
# 区间：10 到 1000 / 10~1000 / 10-1000
_RANGE = re.compile(r"(\d+)\s*(?:到|至|~|～|—|–|-)\s*(\d+)")

# 「改」类的动词。注意**不含**单独的 `=`：
# 「n=400 的 h_rms 是多少」是提问，不是让改界面 —— 这个区分很重要，
# 混起来会导致用户只是想问个数，界面却被改掉了。
_CHANGE_WORDS = (
    "改成", "改为", "设为", "设成", "换成", "变成", "调成", "调整到", "调到",
    "设置成", "设置为", "换成", "改一下", "帮我改", "修改",
)
_OFF_WORDS = ("关掉", "关闭", "取消", "去掉", "不要归一", "别归一", "取消归一")
_ON_WORDS = ("打开", "开启", "开一下", "启用")

# 各意图的关键词表都集中在这里：有几个是被两处用的（比如 _try_set_params
# 要判断「这个链数归谁管」），散在各处容易走偏。
_SWEEP_WORDS = ("扫描", "扫一下", "扫一遍", "扫个", "扫一次", "标度", "斜率",
                "sweep", "拟合", "幂律", "标度律")
_CHAIN_WORDS = ("构象", "画一条", "画个", "画一根", "画一张", "生成一条", "随机走",
                "长出来", "生长", "动画", "换一条", "重画", "新的链")
_VIEW_WORDS = ("视角", "角度", "旋转", "转一下", "换个方向", "缩放", "放大", "缩小")
# 统计量意图的词分强弱。强的（下面这些）足以单独认出一个问题；
# 弱的（多少、算、求……）太泛，得同时找得到一个 n 才算数 ——
# 否则「自由旋转链 θ=109.47 的 Cn 是多少」会因为一个「多少」被拽进来，
# 白白在回答里塞一句「算统计量需要一个 n」。
_STATS_STRONG = ("h_rms", "hrms", "末端距", "回转半径", "最可几", "均方", "统计量", "端距")
_STATS_WEAK = ("多少", "多大", "多长", "算", "求", "是几", "等于")

# 这两组词一出现，问的就**不是**统计量：上面那张表会把它们抢走。
# 「末端距 37 要多少段」里有「末端距」（强词），不挡的话会照 n=37 算一遍统计量，
# 然后回答一个没人问的数 —— 比认不出来还糟。
_SOLVE_RE = re.compile(
    r"反解|反推|倒推|反算|"
    r"(?:多少|几)\s*(?:个\s*)?(?:链段|链节|段)|"
    r"(?:要|需要|求|得)\s*(?:多少|几)\s*n",
    re.I,
)
_FORCE_RE = re.compile(
    r"多大力|多大的力|多大拉力|多大的拉力|拉力|受力|受多大|多少力|"
    r"需要\s*(?:多大|多少)\s*力|要\s*(?:多大|多少)\s*力|"
    r"力\s*伸长|伸长\s*曲线|朗之万|langevin|无量纲力|"
    r"\d\s*p\s*n|\d\s*皮牛",
    re.I,
)

# 反解要的那个目标末端距：h=37 / 末端距 37 / h_rms 到 100 / 全伸展长度 50。
# **别把「链段长度」这种也放进来** —— 那是 l，不是 h；`长度` 单独一个词太泛，
# 一进来就会把「l=2」读成目标 h=2。
_H_HINT = re.compile(
    r"(?<![A-Za-z0-9_])(?:h_rms|h_max|h\*|h|全伸展长度|伸展长度|全伸展|nl"
    r"|末端距|端距)(?![A-Za-z_])\s*"
    r"(?:[=:：＝]|到|为|是|取|要|需要|改成)?\s*(\d+(?:\.\d+)?)",
    re.I,
)
# 四种 h 的说法。**顺序有意义**：最可几要排在根均方前面，
# 否则「最可几 h_rms」这种混着写的会被前面那条先吃掉。
_KIND_HINTS = (
    (("最可几", "h*", "峰值"), "h_mp"),
    (("平均末端距", "平均", "⟨h⟩", "<h>"), "h_mean"),
    (("全伸展", "伸展长度", "全伸长"), "h_max"),
    (("h_rms", "根均方", "均方根", "rms"), "h_rms"),
)

# 力–伸长的目标。三选一：百分比、口说的分数、或者一个 pN 的力。
_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[%％]")
# 「百分之八十」—— 中文数字，不是 ASCII 数字，所以上面那条抓不到
_PCT_WORD_RE = re.compile(
    r"百分之\s*([零〇一二两三四五六七八九十百]+|\d+(?:\.\d+)?)"
)
_PN_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:p\s*n|皮牛)", re.I)
_WORD_FRACTIONS = {
    "一半": 0.5, "半": 0.5, "四分之一": 0.25, "四分之三": 0.75,
    "三分之一": 1.0 / 3.0, "三成": 0.3, "五成": 0.5, "七成": 0.7, "九成": 0.9,
}

_CN_DIGITS = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3,
    "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}


def _cn_int(s: str) -> int | None:
    """中文数字 → 整数，只覆盖百分比里会用到的 0~99。认不出返回 None。

    「八十一」= 81 走的是「逢十进位」：读到「十」时把累计值乘十，
    个位数补在后面。所以 二十=20、十五=15、九十九=99、八=8。
    """
    section = num = 0
    seen = False
    for ch in s:
        if ch in _CN_DIGITS:
            num = _CN_DIGITS[ch]
            seen = True
        elif ch == "十":
            section += (num if num else 1) * 10
            num = 0
            seen = True
        else:
            return None
    return (section + num) if seen else None


def _clean(q: str) -> str:
    return _NOISE.sub(" ", q)


def _is_3d(q: str, low: str) -> bool:
    """这句话说的「链数」是 3D 卡片那一处的吗？

    「条链」这个说法两个卡片都用（3D 卡片有「链数」，扫描卡片有「每个 n 的链数」），
    所以它只有在**没有**扫描上下文的时候才算数。判断错了的代价是把另一张卡片改掉，
    所以宁可判「说不清」，也不要猜。
    """
    if "3d" in low or any(k in q for k in _CHAIN_WORDS):
        return True
    return "条链" in q and not any(k in q for k in _SWEEP_WORDS)


def _numbers(q: str) -> list[float]:
    out = []
    for m in _NUM_RE.finditer(_clean(q)):
        try:
            v = float(m.group(0))
        except ValueError:      # 理论上到不了，正则已经限定了形状
            continue
        if math.isfinite(v):
            out.append(v)
    return out


def _int_list(q: str) -> list[int]:
    """文本里的正整数，按出现顺序、去重。"""
    seen, out = set(), []
    for v in _numbers(q):
        i = int(v)
        if v == i and i >= 1 and i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _has_change(q: str) -> bool:
    return any(w in q for w in _CHANGE_WORDS)


def _num(x) -> str:
    """给用户看的数字：整数就不带小数点，否则 6 位有效数字。"""
    v = float(x)
    if math.isfinite(v) and v == int(v) and abs(v) < 1e15:
        return str(int(v))
    return f"{v:.6g}"


def _const(x) -> float | None:
    """把一个值转成常浮点数，失败回 None（工具只接受常数）。"""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _act(tool: str, **args) -> dict:
    return {"tool": tool, "args": args}


def _geo_seq(a: int, b: int, count: int = 5) -> list[int]:
    """a 到 b 的等比序列（两端都取到、取整去重）。

    标度关系要在**对数**刻度上看，所以用等比而不是等差。
    """
    a, b = int(a), int(b)
    if a <= 0 or b <= 0:
        return []
    if a == b:
        return [a]
    if a > b:
        a, b = b, a
    if count < 2:
        count = 2
    ratio = (b / a) ** (1.0 / (count - 1))
    out = []
    for i in range(count):
        v = int(round(a * ratio ** i))
        if v >= 1 and v not in out:
            out.append(v)
    if out and out[-1] != b:      # 取整可能差一点，把右端点补回去
        out[-1] = b
    return out


# --- 每条回答里都会用到的固定文案 ---------------------------------------

_L_CAVEAT = "（按 l=1 计算；结果与 l 成正比，界面上 l 不是 1 的话乘上它就行）"

# 反解不能借上面那句：n ∝ (h/l)²，**与 l 成反比、还是平方**。
# 照搬「乘上它就行」会把 l 改成 2 时的链段数说成翻倍，而正确的答案是四分之一。
_SOLVE_L_CAVEAT = (
    "（按 l=1 计算。这里 n 与 l 的关系是 n ∝ (h/l)²：l 改成 k 倍，"
    "同一个 h 就只需要 1/k² 的链段数 —— 和统计量那边「与 l 成正比」方向相反，别照搬。）"
)

_EXAMPLES = (
    "· n=400 的 h_rms 是多少\n"
    "· 100 和 1000 比一下\n"
    "· 把 n 改成 400，并打开归一化显示\n"
    "· 画一条 n=500 的链\n"
    "· 扫一下 10 到 1000 的标度关系\n"
    "· 拉到 60% 需要多大力\n"
    "· h=37 要多少个链段"
)


def _seed_of(text: str) -> int:
    """由问题文本推出一个稳定的随机种子。

    用 crc32 而不是 hash()：内置 hash() 对字符串是**逐进程随机化**的
    （PYTHONHASHSEED），重启后同一个问题会给出不同的链，
    那「同一个问题给同一个结果」这件事就不成立了。
    """
    return zlib.crc32(text.encode("utf-8")) & 0x7FFFFFFF


def _refuse(extra: str = "", matched: bool = False) -> LocalReply:
    """兜底回答。

    matched 用来区分两种「答不了」：
      False —— 没认出这个问题（真的不会）
      True  —— 认出来了，但给的参数不合法（会算，只是这次的输入不行）
    这两种在 ai.py 里待遇不同：网关也挂了的时候，前者要如实报网关的错，
    后者仍然值得把这句有用的提示显示出来。
    """
    head = (extra + "\n") if extra else ""
    if matched:
        return LocalReply(text=extra or "这次的输入我没法用。", matched=True)
    return LocalReply(
        text=(
            f"{head}我是「本地模式」（没有配置模型网关），只认几种固定的问法，"
            f"不会自由推理 —— 所以我不猜。\n"
            f"能认出来的有：{ai_tools.capability_list()}。\n"
            f"可以这样问：\n{_EXAMPLES}"
        ),
        matched=False,
    )


# --- 意图：帮助 ----------------------------------------------------------

def _try_help(q: str, low: str) -> LocalReply | None:
    if not any(k in q for k in ("你能做什么", "你能干什么", "你会什么", "能干啥",
                                 "帮助", "怎么用", "用法", "help", "介绍一下你")):
        return None
    return LocalReply(
        text=(
            "我是这个软件内置的助手，现在是「本地模式」：没有配置模型网关，"
            "所以我只能按关键词认几种固定的问法，不会自由推理（也因此我不会编数字）。\n\n"
            f"我认得的：{ai_tools.capability_list()}。\n\n"
            f"可以这样问：\n{_EXAMPLES}\n\n"
            "想让我能自由对话，就配上 FJC_LLM_KEY 和网关地址（见 README 的「AI 助手」一节）。"
        ),
        matched=True,
    )


# --- 意图：归一化 --------------------------------------------------------

def _try_normalize(q: str, low: str) -> LocalReply | None:
    if "归一" not in q and "normalize" not in low:
        return None
    if any(w in q for w in _OFF_WORDS):
        on = False
    elif any(w in q for w in _ON_WORDS):
        on = True
    else:
        # 只说「归一化」而不说开关：打开是更有用的那一边（能看出曲线重合）
        on = True
    tail = ("各 n 的曲线现在会精确重合 —— 这是 √n 标度律在整个分布上都成立的证据。"
            if on else "曲线恢复成各自的 P(h)。")
    return LocalReply(
        text=f"已把归一化显示{'打开' if on else '关闭'}。{tail}",
        actions=[_act("set_normalize", on=on)],
        steps=[{"tool": "set_normalize", "args": {"on": on}, "ok": True}],
        matched=True,
    )


# --- 意图：改参数 --------------------------------------------------------

def _try_set_params(q: str, low: str) -> LocalReply | None:
    if not _has_change(q):
        return None

    n_hit = _N_HINT.search(q)
    l_hit = _L_HINT.search(q)
    args: dict = {}
    said: list[str] = []

    if n_hit:
        n_val = int(n_hit.group(1))
        if n_val < 1:
            return _refuse(f"链段数 n 至少为 1，收到 {n_val}。", matched=True)
        args["n"] = n_val
    if l_hit:
        l_val = _const(l_hit.group(1))
        if l_val is None or l_val <= 0:
            return _refuse("链段长度 l 必须大于 0。", matched=True)
        args["l"] = l_val

    if not args:
        # 有「改」这个动词、但没写 n= 也没写 l=。
        if _CHAINS_HINT.search(q):
            # 数字是链数，不是主参数 n —— 绝不能顺手把它当 n 用。
            # 「把扫描的链数改成 5000」里的 5000 一旦被当成 n，
            # 用户只是想调扫描精度，主图的 n 却被改成 5000 了。
            if any(k in q for k in _SWEEP_WORDS) or _is_3d(q, low):
                return None          # 有上下文，让扫描 / 3D 那两个意图去接
            return _refuse(
                "改链数得说清是哪一处：3D 卡片（1~5 条，同时画几条链），"
                "还是扫描卡片（每个 n 采几条链）。比如「画 3 条链」"
                "或「把扫描的链数改成 5000」。",
                matched=True,
            )
        nums = _int_list(q)
        if len(nums) == 1:
            args["n"] = nums[0]
        else:
            return None      # 说不清改什么，交给别的意图或拒绝

    parts = []
    if "n" in args:
        said.append(f"n 改为 {args['n']}")
    if "l" in args:
        said.append(f"l 改为 {_num(args['l'])}")
    parts.append("已把 " + "、".join(said) + "。")

    # 顺手把改完之后的 h_rms 报出来 —— 用户改 n 通常就是想看这个数
    if "n" in args:
        probe = {"n": args["n"]}
        if "l" in args:
            probe["l"] = args["l"]
        r = ai_tools.run("compute_statistics", probe)
        if r["ok"]:
            d = r["data"]
            parts.append(
                f"此时 h_rms = {_num(d['h_rms'])}（h_rms = l√n），"
                f"最可几末端距 h* = {_num(d['h_mp'])}。"
            )
            if "l" not in args:
                parts.append(_L_CAVEAT)
            for w in d.get("warnings", []):
                parts.append(f"注意：{w}")

    return LocalReply(
        text="\n".join(parts),
        actions=[_act("set_params", **args)],
        steps=[{"tool": "set_params", "args": args, "ok": True}],
        matched=True,
    )


# --- 意图：多 n 对比 -----------------------------------------------------

def _try_compare(q: str, low: str) -> LocalReply | None:
    if not any(k in q for k in ("对比", "比较", "比一下", "哪个大", "谁大",
                                 "放一起", "放在一起", "比一比")):
        return None
    ns = _int_list(q)
    if len(ns) < 2:
        return None
    ns = ns[:ai_tools.ASSIST_COMPARE_MAX]

    r = ai_tools.run("compare_n", {"ns": ns})
    if not r["ok"]:
        return _refuse(f"这个我算不了：{r['error']}", matched=True)
    rows = r["data"]["rows"]
    base = rows[0]

    lines = [f"对比 {len(rows)} 个 n（l=1）："]
    for i, row in enumerate(rows):
        rel = ""
        if i:
            rel = f"，是 n={base['n']} 的 {_num(row['h_rms_ratio_to_first'])} 倍"
        lines.append(f"· n = {row['n']}：h_rms = {_num(row['h_rms'])}{rel}")
    lines.append("注意 h_rms ∝ √n：n 变成 k 倍，h_rms 只变成 √k 倍。")
    for w in r["data"].get("warnings", []):
        lines.append(f"注意：{w}")

    return LocalReply(
        text="\n".join(lines) + f"\n{_L_CAVEAT}",
        actions=[_act("set_curves", ns=ns)],
        steps=[{"tool": "compare_n", "args": {"ns": ns}, "ok": True},
               {"tool": "set_curves", "args": {"ns": ns}, "ok": True}],
        matched=True,
    )


# --- 意图：标度扫描 ------------------------------------------------------

def _try_sweep(q: str, low: str) -> LocalReply | None:
    if not any(k in q for k in _SWEEP_WORDS):
        return None

    ns = _int_list(q)
    rng = _RANGE.search(_clean(q))
    given = False      # 用户**自己指定**了扫描点吗？决定了要不要动界面上的卡片
    # 先看区间再看数字列表：「扫一下 10 到 1000」会被 _int_list 读成 [10, 1000]
    # 两个点，那就画不出标度线了 —— 区间要的是等比序列，不是那两个端点。
    if rng:
        ns = _geo_seq(int(rng.group(1)), int(rng.group(2)))
        how = f"按 {rng.group(1)}→{rng.group(2)} 生成了 {len(ns)} 个等比点"
        given = True
    elif len(ns) >= 2:
        ns = ns[:ai_tools.ASSIST_SWEEP_N_MAX]
        how = f"用了你给的 {len(ns)} 个 n"
        given = True
    else:
        ns = [10, 30, 100, 300, 1000]
        how = "没给范围，默认扫 10, 30, 100, 300, 1000"

    chains = None
    ch = _CHAINS_HINT.search(q)
    if ch:
        # 「每个 500 条链」这种写法本身就说清了要多少条；
        # 而「链数 500」可能只是在复述现状，得配上「改成」才算数。
        if ch.group(2) is not None or _has_change(q):
            raw = ch.group(1) or ch.group(2)
            if raw:
                chains = int(raw)

    args: dict = {"ns": ns}
    if chains is not None:
        args["chains"] = chains

    r = ai_tools.run("run_sweep", args)
    if not r["ok"]:
        return _refuse(f"这个我算不了：{r['error']}", matched=True)
    d = r["data"]

    lines = [f"标度扫描：{how}，每个 n 采 {d['chains']:,} 条独立链。"]
    for row in d["rows"]:
        lines.append(
            f"· n = {row['n']}：模拟 ⟨R²⟩ = {_num(row['R2_mean'])}，"
            f"理论 nl² = {_num(row['R2_theory'])}，"
            f"偏差 {row['rel_dev_percent']:+.2f}%"
        )
    if d.get("slope") is not None:
        lines.append(
            f"log⟨R²⟩ 对 log n 的拟合斜率 = {d['slope']:.5f}（理论值 1），"
            f"R² = {d['fit_r2']:.6f}。"
        )
    lines.append(
        f"每个点的相对标准误约 {d['rel_se_percent']:.2f}%，"
        f"所以偏差落在这个量级的几倍以内都属正常。"
    )
    if d.get("clamped"):
        lines.append(d["clamped"])
    for w in d.get("warnings", []):
        lines.append(f"注意：{w}")

    actions: list[dict] = []
    steps = [{"tool": "run_sweep", "args": args, "ok": True}]
    if given or chains is not None:
        # 只在用户**自己指定**了扫描点时才同步到界面上的卡片。
        # 用的是默认值时不动它 —— 人家可能正存着自己的一组参数，
        # 一次随口的提问不该把它冲掉。
        sweep_args: dict = {"ns": ns}
        if chains is not None:
            sweep_args["chains"] = chains
        actions.append(_act("set_sweep_params", **sweep_args))
        steps.append({"tool": "set_sweep_params", "args": sweep_args, "ok": True})
        lines.append("已把这组参数填进下面的扫描卡片，点「运行模拟」就能看到曲线。")

    return LocalReply(
        text="\n".join(lines) + f"\n{_L_CAVEAT}",
        actions=actions,
        steps=steps,
        matched=True,
    )


# --- 意图：3D 构象 / 视角 ------------------------------------------------
# 关键词表在文件顶部（_CHAIN_WORDS / _VIEW_WORDS），那里集中放，便于看出谁和谁重了。


def _try_chain_view(q: str, low: str) -> LocalReply | None:
    has_chain = _is_3d(q, low)
    # 「自由旋转链」里的「旋转」说的是链模型，不是让人转画布。抠掉它，
    # 否则「自由旋转链 θ=109.47 的 Cn 是多少」会顺手把 3D 视角转掉 ——
    # 用户问的是物理，界面却被改了。
    vq = q.replace("自由旋转", " ").replace("自由链接", " ")
    has_view = any(k in vq for k in _VIEW_WORDS)
    if not has_chain and not has_view:
        return None

    lines: list[str] = []
    actions: list[dict] = []
    steps: list[dict] = []

    if has_chain:
        n_hit = _N_HINT.search(q)
        n_val = int(n_hit.group(1)) if n_hit else None
        ch = _CHAINS_HINT.search(q)
        chains_val = None
        if ch:
            raw = ch.group(1) or ch.group(2)
            if raw:
                chains_val = int(raw)

        # 现取一个种子，**同时**交给界面和模拟：这样下面报出来的 |R|
        # 和画布上那条链是同一条。否则数字和图对不上，比不报还糟。
        seed = _seed_of(q)
        args: dict = {"seed": seed}
        if n_val is not None:
            if n_val > 20000:
                return _refuse(
                    f"3D 视图最多画 20000 个链段（再多画出来就是一团结），收到 {n_val}。",
                    matched=True,
                )
            args["n"] = n_val
        if chains_val is not None:
            if not 1 <= chains_val <= 5:
                return _refuse(f"3D 视图一次最多画 5 条链，收到 {chains_val}。",
                               matched=True)
            args["chains"] = chains_val

        actions.append(_act("set_chain_params", **args))
        steps.append({"tool": "set_chain_params", "args": args, "ok": True})

        what = []
        if n_val is not None:
            what.append(f"n = {n_val}")
        if chains_val is not None:
            what.append(f"{chains_val} 条链")
        lines.append(
            "已在右边的 3D 卡片上画了 " + ("、".join(what) if what else "一条新链")
            + f"，随机种子 {seed}（记下这个种子就能复现同一条链）。"
        )

        # 报末端距 |R|。只有知道 n 才报 —— 不知道界面上 n 是多少还硬报一个数，
        # 就成了编造。都是用同一个种子算的，和画布上那条链一致。
        if n_val is not None:
            sim_args: dict = {"n": n_val, "seed": seed}
            if chains_val is not None:
                sim_args["chains"] = chains_val
            sr = ai_tools.run("simulate_chain", sim_args)
            if sr["ok"]:
                sd = sr["data"]
                if sd["chains"] == 1:
                    lines.append(
                        f"这条链 |R| = {_num(sd['R_mag'][0])}，"
                        f"理论 h_rms = {_num(sd['h_rms'])}，"
                        f"比值 {_num(sd['R_over_h_rms'])}。"
                        f"单条链偏离 h_rms 是正常涨落，固有幅度约 "
                        f"{sd['rel_se_percent']:.0f}%。"
                    )
                else:
                    lines.append(
                        f"{sd['chains']} 条链的 ⟨R²⟩ = {_num(sd['R2_mean'])}，"
                        f"理论 nl² = {_num(sd['R2_theory'])}"
                        f"（相对标准误约 {sd['rel_se_percent']:.0f}%）。"
                    )
                lines.append(_L_CAVEAT)
        else:
            lines.append(
                "（你没给 n，卡片会沿用它现在的值，所以我不好替它报 |R| —— "
                "卡片下方的对照条里就有实测 R 和理论 h_rms。）"
            )

        if any(k in q for k in ("生长", "动画", "长出来", "播放", "演示")):
            actions.append(_act("play_chain"))
            steps.append({"tool": "play_chain", "args": {}, "ok": True})
            lines.append("已经在播放生长动画了。")

    if has_view:
        if any(k in q for k in ("放大", "缩小", "缩放")):
            zoom = 1.6 if "放大" in q else (0.6 if "缩小" in q else 1.0)
            args = {"zoom": zoom}
            actions.append(_act("set_chain_view", **args))
            steps.append({"tool": "set_chain_view", "args": args, "ok": True})
            lines.append(f"缩放设为 {_num(zoom)} 倍（1 是初始大小）。")
        else:
            # 本地模式读不到当前视角，所以是「设成这个值」而不是「叠加转一点」。
            # 说清楚这一点，用户才知道要不要接着拖。
            args = {"azim": 45, "elev": 25}
            actions.append(_act("set_chain_view", **args))
            steps.append({"tool": "set_chain_view", "args": args, "ok": True})
            lines.append(
                "已把视角设为水平角 45°、仰角 25°（本地模式读不到当前角度，"
                "所以是设成这个值而不是在上面的基础上转；想微调直接拖画布就行）。"
            )

    if not actions:
        return None
    return LocalReply(text="\n".join(lines), actions=actions, steps=steps, matched=True)


# --- 意图：自由旋转链交叉验证 -------------------------------------------

def _try_freerotating(q: str, low: str) -> LocalReply | None:
    if not any(k in q for k in ("自由旋转", "键角", "theta", "θ")):
        return None

    th = _THETA_HINT.search(q)
    theta = _const(th.group(1)) if th else 90.0
    if theta is None:
        return _refuse("键角我没读懂。", matched=True)

    n_hit = _N_HINT.search(q)
    n_val = int(n_hit.group(1)) if n_hit else 100
    placeholder = "" if n_hit else "（你没给 n，这里取 100 —— Cn 与 n 无关，不影响结论）"

    r = ai_tools.run("freerotating_check", {"n": n_val, "theta_deg": theta})
    if not r["ok"]:
        return _refuse(f"这个我算不了：{r['error']}", matched=True)
    d = r["data"]
    lines = [
        f"自由旋转链，键角 θ = {_num(theta)}°{placeholder}：",
        f"· ⟨h²⟩ = {_num(d['h2_freerotating'])}（自由旋转链）",
        f"· ⟨h²⟩ = {_num(d['h2_fjc'])}（FJC，即 θ=90° 的退化情形）",
        f"· 比值 = {_num(d['ratio'])}，相当于 Cn = {_num(d['Cn_equivalent'])}",
    ]
    if abs(theta - 90.0) < 1e-9:
        lines.append("θ=90° 时应当精确退化回 FJC 的 nl²，比值 1 —— 这条通过了。")
    elif abs(d["ratio"] - 2.0) < 1e-6:
        lines.append("这就是聚乙烯键角 arccos(−1/3)≈109.47° 的教科书结果 Cn = 2。")
    return LocalReply(text="\n".join(lines), steps=[
        {"tool": "freerotating_check", "args": {"n": n_val, "theta_deg": theta}, "ok": True}
    ], matched=True)


# --- 意图：由末端距反解链段数 -------------------------------------------

def _try_solve_n(q: str, low: str) -> LocalReply | None:
    if not _SOLVE_RE.search(q):
        return None
    h_hit = _H_HINT.search(q)
    if h_hit is None:
        # 认出了「反解」这个意图，但没读到目标末端距 —— 返回 None 而不是拒绝，
        # 免得和同一句里的别的意图叠成两段答非所问的废话。
        return None

    h_val = _const(h_hit.group(1))
    if h_val is None or h_val <= 0:
        return _refuse(f"目标末端距要是一个正数，收到 {h_hit.group(1)}。", matched=True)

    kind = "h_rms"
    for words, k in _KIND_HINTS:
        if any(w in q for w in words) or any(w in low for w in words):
            kind = k
            break

    l_hit = _L_HINT.search(q)
    l_val = _const(l_hit.group(1)) if l_hit else None
    if l_val is not None and l_val <= 0:
        return _refuse("链段长度 l 必须大于 0。", matched=True)

    args: dict = {"h": h_val, "kind": kind}
    if l_val:
        args["l"] = l_val
    r = ai_tools.run("solve_n", args)
    if not r["ok"]:
        return _refuse(f"这个我算不了：{r['error']}", matched=True)
    d = r["data"]

    unit = f"l = {_num(l_val)}" if l_val else "l = 1"
    lines = [
        f"由 {d['kind_label']} = {_num(d['h'])} 反解链段数（{unit}）：",
        f"· 连续解 n_exact = {_num(d['n_exact'])}",
        f"· ⌊n⌋ = {_num(d['n_floor'])}　→ 算出来的 h 不超过目标",
        f"· ⌈n⌉ = {_num(d['n_ceil'])}　→ 算出来的 h 不低于目标　← 推荐这个",
    ]
    # 必须回显是哪一种 h：四种在同一个 n 下能差一到两个数量级，
    # 不说清楚，用户就没法判断这个 n 是照哪一种标度反的。
    lines.append(
        f"注意：这算的是 **{d['kind_label']}**。"
        f"换一种 h（h_rms / ⟨h⟩ / h* / nl）答案会不一样，说清楚要哪种再问就行。"
    )
    for w in d.get("warnings", []):
        lines.append(f"注意：{w}")
    if not l_val:
        lines.append(_SOLVE_L_CAVEAT)

    return LocalReply(
        text="\n".join(lines),
        steps=[{"tool": "solve_n", "args": args, "ok": True}],
        matched=True,
    )


# --- 意图：力–伸长 --------------------------------------------------------

def _try_force(q: str, low: str) -> LocalReply | None:
    if not _FORCE_RE.search(q):
        return None

    # 目标三选一。一个都读不到就返回 None —— 认得出话题、读不出参数时，
    # 硬凑一段话不如让下面的通用拒绝把能问的格式列出来。
    lam: float | None = None
    m = _PCT_RE.search(q) or _PCT_WORD_RE.search(q)
    if m:
        raw = m.group(1)
        # 60% 走这条路；百分之八十 走中文数字那条
        lam = _cn_int(raw) if not raw[:1].isdigit() else _const(raw)
        if lam is not None and not raw[:1].isdigit():
            lam = float(lam)
        if lam is not None:
            lam /= 100.0
    else:
        for word, v in _WORD_FRACTIONS.items():
            if word in q:
                lam = v
                break
    fp_hit = _PN_RE.search(q) if lam is None else None

    if lam is None and fp_hit is None:
        return None

    l_hit = _L_HINT.search(q)
    l_val = _const(l_hit.group(1)) if l_hit else 1.0
    if l_val is None or l_val <= 0:
        return _refuse("链段长度 l 必须大于 0。", matched=True)

    if lam is not None:
        args: dict = {"lam": lam, "l": l_val}
        r = ai_tools.run("force_extension", args)
        if not r["ok"]:
            return _refuse(f"这个我算不了：{r['error']}", matched=True)
        qd = r["data"]["to_reach_lam"]
        lines = [
            f"拉到全长的 {_num(lam * 100)}%（归一化伸长 λ = {_num(lam)}）：",
            f"· 无量纲力 x = {_num(qd['x'])}　← x = f·l/(k_B·T)",
            f"· 力 f = {_num(qd['force_pn'])} pN",
            f"λ 和 x 只由你要拉到的比例决定，与链段数 n、链段长度 l 都无关；"
            f"换成 pN 这一步才用到单位，那一步 f 与 l 成**反比**、与 T 成正比。",
            f"（按 l = {_num(l_val)} nm、T = 298.15 K 换算。"
            f"只报 λ 和 x 就不报绝对伸长了 —— 那个数要乘 n，本地模式不知道界面上的 n 是多少，"
            f"替你猜一个等于编数。）",
        ]
        return LocalReply(
            text="\n".join(lines),
            steps=[{"tool": "force_extension", "args": args, "ok": True}],
            matched=True,
        )

    fp = _const(fp_hit.group(1))
    if fp is None or fp < 0:
        return _refuse(f"力要是一个非负数，收到 {fp_hit.group(1)}。", matched=True)
    args = {"force_pn": fp, "l": l_val}
    r = ai_tools.run("force_extension", args)
    if not r["ok"]:
        return _refuse(f"这个我算不了：{r['error']}", matched=True)
    qd = r["data"]["at_this_force_pn"]
    lines = [
        f"施加 {_num(fp)} pN 的力：",
        f"· 换成无量纲力 x = {_num(qd['x'])}　← 这一步才用到 l 和 T",
        f"· 归一化伸长 λ = {_num(qd['lam'])}　（全长的 {_num(qd['lam'] * 100)}%）",
        f"（按 l = {_num(l_val)} nm、T = 298.15 K 换算。同一个 pN 在 l 更长的链上"
        f"相当于更小的 x，拉伸也就更少；λ 到 1 附近意味着链快拉直了。）",
        "没报绝对伸长 —— 那个数要乘 n，本地模式不知道界面上的 n 是多少。",
    ]
    return LocalReply(
        text="\n".join(lines),
        steps=[{"tool": "force_extension", "args": args, "ok": True}],
        matched=True,
    )


# --- 意图：算统计量 ------------------------------------------------------

def _try_stats(q: str, low: str) -> LocalReply | None:
    strong = any(k in low for k in _STATS_STRONG)
    if not strong and not any(k in low for k in _STATS_WEAK):
        return None
    # 反解和力–伸长都长得像统计量问题（都爱写「末端距」「多少」），
    # 让它们先走；这里硬算一个 n 出来就是答非所问。
    if _SOLVE_RE.search(q) or _FORCE_RE.search(q):
        return None

    n_hit = _N_HINT.search(q)
    nums = _int_list(q)
    if not strong and n_hit is None and len(nums) != 1:
        # 只命中了泛词（「多少」「算」），却又有一堆数字：分不清哪个是 n。
        # 「100 到 1000 的斜率是多少」是扫描的问题，不该在这里再算一遍 n=100。
        return None

    if n_hit:
        n_val = int(n_hit.group(1))
    else:
        n_val = nums[0] if nums else None
    if n_val is None:
        return _refuse(
            "算统计量需要一个链段数 n。比如：「n=400 的 h_rms 是多少」。",
            matched=True,
        )

    l_hit = _L_HINT.search(q)
    l_val = None
    if l_hit:
        l_val = _const(l_hit.group(1))
        if l_val is None or l_val <= 0:
            return _refuse("链段长度 l 必须大于 0。", matched=True)

    args: dict = {"n": n_val}
    if l_val:
        args["l"] = l_val
    r = ai_tools.run("compute_statistics", args)
    if not r["ok"]:
        return _refuse(f"这个我算不了：{r['error']}", matched=True)
    d = r["data"]

    unit = f"l = {_num(l_val)}" if l_val else "l = 1"
    lines = [f"n = {d['n']}（{unit}）的 FJC 结果："]
    lines.append(f"· 根均方末端距 h_rms = {_num(d['h_rms'])}　← 主输出，√(nl²)")
    lines.append(f"· 最可几末端距 h* = {_num(d['h_mp'])}　← P(h) 的极大值点")
    lines.append(f"· 平均末端距 ⟨h⟩ = {_num(d['h_mean'])}")
    lines.append(f"· 分布标准差 σ = {_num(d['sigma'])}")
    lines.append(f"· 回转半径 Rg = {_num(d['Rg_rms'])}　← √(nl²/6)")
    lines.append(f"· 全伸展长度 nl = {_num(d['h_max'])}")
    lines.append(f"· 特征比 Cn = {_num(d['Cn'])}　← FJC 恒为 1")
    lines.append("序关系 h* < ⟨h⟩ < h_rms ≤ nl 成立。" if d["ordering_ok"]
                 else "⚠ 序关系异常，这不该发生，请把参数发回来看看。")
    for w in d.get("warnings", []):
        lines.append(f"注意：{w}")
    if not l_val:
        lines.append(_L_CAVEAT)

    return LocalReply(
        text="\n".join(lines),
        steps=[{"tool": "compute_statistics", "args": args, "ok": True}],
        matched=True,
    )


HANDLERS = (
    _try_help,
    _try_normalize,
    _try_set_params,
    _try_compare,
    _try_sweep,
    _try_chain_view,
    _try_freerotating,
    _try_solve_n,
    _try_force,
    _try_stats,
)


def answer(text: str) -> LocalReply:
    """按关键词认意图。**可以同时命中多个**，认不出的部分不会硬凑。"""
    q = (text or "").strip()
    if not q:
        return _refuse("你还没说话呢。")
    low = q.lower()

    parts = [r for r in (h(q, low) for h in HANDLERS) if r is not None]
    if not parts:
        return _refuse()

    # 命中一条就算认出来了 —— 关键词匹配没有「谁更重要」这回事，
    # 认得的部分全做掉，比只挑一个更符合预期。
    #
    # 已知的边界（不是没想到，是关键词匹配做不到）：一句话里「一半认得、一半认不得」时，
    # 认不得的那半会被**安静地丢掉**，不会附上一段「这半句我没懂」。要判断哪半句
    # 属于谁，就得先切分句子，那是另一套东西了。想自由对话请配模型网关。
    texts, actions, steps = [], [], []
    seen_actions = set()
    for p in parts:
        if p.text:
            texts.append(p.text)
        for a in p.actions:
            key = (a["tool"], repr(sorted(a["args"].items())))
            if key not in seen_actions:
                seen_actions.add(key)
                actions.append(a)
        for s in p.steps:
            steps.append(s)

    return LocalReply(
        text="\n\n".join(texts),
        actions=actions,
        steps=steps,
        matched=all(p.matched for p in parts),
    )


def can_handle(text: str) -> bool:
    """这个问题本地模式认不认得出来。ai.py 用它决定网关挂了时是报错还是兜底。"""
    return answer(text).matched
