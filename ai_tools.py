"""AI 助手的工具集：**定义**与**执行**。

两条边界，整个助手的形状都由它们决定：

1. **工具不碰网络。** 服务端工具直接调 fjc_core 的函数，不走 HTTP 回环 ——
   自己请求自己会占住一个 worker，还得处理端口，纯属自找麻烦。
2. **循环不懂物理。** ai.py 只管协议（谁能调、最多调几轮、怎么降级），
   物理量一律由本文件交给 fjc_core 算。所以这里**没有任何公式**，
   只是把 fjc_core 的返回值挑成模型看得懂的形状。

工具分两类：
  server   —— 有 handler，在 Flask 进程里就地执行
  frontend —— 只有 schema，由浏览器执行（改界面状态、读界面状态）。
              它们的参数**必须先校验再落地**，见 web/ai.js。

工具定义**只写一遍，渲染两处**：`openai_tools()` 给标准 function calling，
`tools_prompt()` 给系统提示词。国内网关对 `tools` 的支持不一定可靠，
正文里那份说明就是那时候的退路 —— 不是两套实现，是一个定义的两个视图。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import fjc_core
from fjc_core import FJCInputError


class ToolError(Exception):
    """工具自身的拒绝（参数越界、预算不够……）。会原样变成给模型看的中文提示。"""


# --- 助手这条路的资源上限 ------------------------------------------------
# 为什么比界面按钮那条路紧得多：界面按钮跑 3~4 秒是可接受的（有忙状态、
# 是用户主动点的），但一个聊天问题占住 CPU 四秒就是卡住了。
# 这些数字对应最坏约 1 秒。

# 一次最多几条链
ASSIST_CHAINS_MAX = 2000
# 单链构象的顶点预算 chains × (n+1)。random_chain 会把全部顶点算出来再返回，
# 内存是 O(chains × n)，所以要卡得比统计类工具紧（100 万点约 24MB/数组）。
ASSIST_POINTS_MAX = 1_000_000
# 扫描的链数上限与总计算量预算（chains × Σn）。实测约 1500 万/秒。
ASSIST_SWEEP_CHAINS_MAX = 2000
ASSIST_SWEEP_WORK_MAX = 12_000_000
# 扫描默认链数：比界面的 10000 小一个量级，够看清标度关系了
ASSIST_SWEEP_DEFAULT_CHAINS = 2000
# 一次最多对比几个 n。与 server.MAX_CURVES 一致（前端只有 5 个色槽）
ASSIST_COMPARE_MAX = 5
# 扫描一次最多几个 n。核心的 SWEEP_N_MAX 是 20，够画满一张图；
# 但一个聊天问题不需要 20 个点，8 个已经能看清斜率了
ASSIST_SWEEP_N_MAX = 8


@dataclass(frozen=True)
class Tool:
    """一个工具的完整声明。params 是 JSON Schema 的 properties。"""

    name: str
    summary: str                     # 一句话，给模型看
    params: dict[str, dict] = field(default_factory=dict)
    required: tuple[str, ...] = ()
    kind: str = "server"             # "server" | "frontend"
    handler: Callable[..., dict] | None = None
    readonly: bool = False           # 前端工具里只有 get_page_state 是只读的

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.summary,
                "parameters": {
                    "type": "object",
                    "properties": self.params,
                    "required": list(self.required),
                },
            },
        }


def _p(type_: str, desc: str, **extra) -> dict:
    """一个参数的 JSON Schema 片段。"""
    out = {"type": type_, "description": desc}
    out.update(extra)
    return out


# --- n 列表归一化 --------------------------------------------------------

def _n_list(ns) -> list:
    """把 n 归一化成列表：数组、单个数字，或「逗号/空格分隔」的字符串。

    这里故意与 `fjc_core._as_n_list` 的行为保持一致（分隔符集合也照抄），
    原因是那个函数是私有的，跨模块调它不合适，而重写一套又怕两处走偏 ——
    所以 `test_ai.py` 里有一条用例拿同一批输入逐一对拍这两个实现。
    想改分隔符的话，两边都要改，那条测试会立刻报出来。
    """
    if ns is None:
        raise ToolError("请至少给一个链段数 n")
    if isinstance(ns, (list, tuple)):
        items = list(ns)
    elif isinstance(ns, str):
        items = [t for t in re.split(r"[,，、;；\s]+", ns.strip()) if t]
    else:
        items = [ns]
    if not items:
        raise ToolError("请至少给一个链段数 n")
    return items


# --- 链数收缩 ------------------------------------------------------------

def _fit_chains(chains: int, per_chain: int, max_chains: int, work_max: int,
                unit: str = "链数") -> tuple[int, str | None]:
    """把链数降到同时满足「条数上限」和「总计算量上限」。

    只降链数，**绝不降 n 或 l** —— 链数只影响统计精度（rel_se 会如实报出来），
    而 n、l 是物理量，偷偷改掉等于给出一个错误的答案。两者性质完全不同。

    返回 (实际链数, 说明或 None)。说明会原样进模型上下文，
    所以它必须说清「从多少降到了多少」，而不是静默截断。
    """
    if per_chain <= 0:
        raise ToolError("链段数必须为正")

    # 计算量上限折算成条数；连一条都放不下就说明 n 本身太大了
    work_cap = work_max // per_chain
    if work_cap < 1:
        raise ToolError(
            f"计算量超出预算：{per_chain:,} × 1 条链已经超过 {work_max:,}，"
            f"请把 n 的总量降下来"
        )

    allowed = min(max_chains, work_cap)
    if chains <= allowed:
        return chains, None
    return allowed, (
        f"已把{unit}从 {chains:,} 降到 {allowed:,}"
        f"（上限：条数 ≤ {max_chains:,}，且 链数 × Σn ≤ {work_max:,}）；"
        f"链数只影响统计精度，n 和 l 没有改动"
    )


# --- 服务端工具 ----------------------------------------------------------

def _t_compute_statistics(n, l=1.0) -> dict:
    """一个 n 的全部解析统计量。"""
    r = fjc_core.compute(n, l)
    return {
        "n": r.n,
        "l": r.l,
        "h2": r.h2,
        "h_rms": r.h_rms,
        "h_mp": r.h_mp,
        "h_mean": r.h_mean,
        "h_max": r.h_max,
        "sigma": r.sigma,
        "Rg2": r.Rg2,
        "Rg_rms": r.Rg_rms,
        "beta": r.beta,
        "Cn": r.Cn,
        # 自检：这个序关系是模型最容易说反的地方（谁比谁大），
        # 直接算好给它，省得它去比大小比错。
        # 末位用 ≤ 而不是 <：n=1 时 h_rms 与 h_max 都等于 l（链只有一个链段，
        # 伸展长度就是它自己），写成 < 会在 n=1 误报「序关系不成立」。
        "ordering": "h_mp < h_mean < h_rms <= h_max",
        "ordering_ok": r.h_mp < r.h_mean < r.h_rms <= r.h_max,
        "warnings": r.warnings,
        # 注意：**不回传 curve**（400 点 × 2）。模型不需要那条曲线上的点，
        # 它只需要知道 P(h) 由高斯近似给出、极大值在 h_mp。
    }


def _t_compare_n(ns, l=1.0) -> dict:
    """多个 n 横向对比。"""
    if isinstance(ns, (int, float)) and not isinstance(ns, bool):
        ns = [ns]
    if not isinstance(ns, (list, tuple, str)):
        raise ToolError("ns 需要是数字数组，例如 [100, 1000]")
    raw = _n_list(ns)
    if len(raw) > ASSIST_COMPARE_MAX:
        raise ToolError(
            f"一次最多对比 {ASSIST_COMPARE_MAX} 个 n，收到 {len(raw)} 个；"
            f"要更多请分两次问"
        )
    results = fjc_core.compute_many(raw, l)

    rows = []
    for r in results:
        rows.append({
            "n": r.n,
            "h_rms": r.h_rms,
            "h_mp": r.h_mp,
            "h_mean": r.h_mean,
            "h_max": r.h_max,
            "Rg_rms": r.Rg_rms,
            "Cn": r.Cn,
        })

    # 相对于第一个 n 的倍数。h_rms ∝ √n，所以 n 翻四倍、h_rms 只翻一倍 ——
    # 这是「标度」这件事最容易被说错的地方，直接把倍数算出来
    base = rows[0]
    for row in rows:
        row["h_rms_ratio_to_first"] = row["h_rms"] / base["h_rms"]
        row["n_ratio_to_first"] = row["n"] / base["n"]

    warnings = []
    for r in results:
        for w in r.warnings:
            if w not in warnings:
                warnings.append(w)
    return {"l": results[0].l, "rows": rows, "warnings": warnings}


def _t_simulate_chain(n, l=1.0, seed=None, chains=1) -> dict:
    """随机生成若干条链，给末端距的**一次实现**（不含顶点）。"""
    # 用 validate_chain() 而不是自己 _as_int()：它一次把 n / l / seed / chains
    # 都归一化并校验了（seed=None 时还会现取一个回传），而且是公开 API。
    #
    # max_n 传 None 是**故意的**：CHAIN_N_MAX=20000 是「画出来不像一团糊」的
    # 表现层约束，报错文案也是「绘制用的链段数不能超过…」。助手这条路不画图
    # （顶点根本不回传），套用绘制上限只会给出误导性的理由，还会平白拒掉
    # n=50000 这种完全合理的物理提问。真正的护栏是下面的计算量预算，
    # 而它正好也把内存框住了：chains × n 有上限，顶点数组就不可能失控。
    n_i, l_f, seed_i, chains_i = fjc_core.validate_chain(
        n, l, seed, chains,
        max_chains=None, max_n=None, points_max=None,
    )

    chains_i, note = _fit_chains(
        chains_i, n_i + 1, ASSIST_CHAINS_MAX, ASSIST_POINTS_MAX
    )

    res = fjc_core.random_chain(
        n_i, l_f, seed_i, chains_i,
        max_chains=ASSIST_CHAINS_MAX,
        max_n=None,
        points_max=ASSIST_POINTS_MAX,
    )

    out = {
        "n": res.n,
        "l": res.l,
        "seed": res.seed,
        "chains": res.chains,
        "R_mag": [round(float(v), 6) for v in res.R_mag],
        "R2_mean": res.R2_mean,
        "R2_theory": res.R2_theory,
        "R2_rel_se": res.R2_rel_se,
        # 该有多大涨落 —— 判断「偏了 20% 算不算正常」全靠它
        "rel_se_percent": res.R2_rel_se * 100,
        "h_rms": res.h_rms,
        "h_mp": res.h_mp,
        "h_mean": res.h_mean,
        "h_max": res.h_max,
        "warnings": res.warnings,
    }
    if res.chains == 1:
        # float() 不能省：res.R_mag 是 numpy 数组，取元素得到的是 np.float64，
        # jsonify 序列化不了它（会 500）。
        out["R_over_h_rms"] = float(res.R_mag[0]) / res.h_rms
        out["note"] = (
            "这是一条链的**一次随机实现**，它偏离 h_rms 是正常涨落，不是错误。"
            f"单条链的固有相对涨落就有约 {res.R2_rel_se * 100:.1f}%。"
        )
    if note:
        out["clamped"] = note
    # **不回传顶点**：那是给画布用的（n=1000 就 40KB），塞进对话上下文纯属浪费。
    # 要画图用前端工具 set_chain_params，让浏览器自己去 /api/chain 拿。
    return out


def _t_run_sweep(ns, l=1.0, chains=ASSIST_SWEEP_DEFAULT_CHAINS, seed=None) -> dict:
    """⟨R²⟩ 对 n 的标度扫描：斜率应当 ≈ 1。"""
    # validate_sweep() 一次把 ns 归一化（数组 / 单个数 / "10, 100" 字符串都行）、
    # 校验每个 n、去重、并归一化 seed 与 chains。上限先全开，
    # 因为链数要按实际 Σn 收缩过再交给 sweep()。
    n_list, l_f, seed_i, chains_i = fjc_core.validate_sweep(
        ns, l, seed, chains,
        max_points=ASSIST_SWEEP_N_MAX,
        max_chains=None,
        work_max=None,
    )
    if len(n_list) < 2:
        # 一个 n 拟合不出斜率。sweep() 本身不会报错（它只是给个 warning、
        # 把 slope 留成 None），但那样模型会以为拿到了结果，
        # 所以这里直接拦下来，让它去补一个 n。
        raise ToolError(
            f"至少要两个不同的 n 才能拟合标度斜率，只收到 {len(n_list)} 个"
        )

    chains_i, note = _fit_chains(
        chains_i, sum(n_list), ASSIST_SWEEP_CHAINS_MAX, ASSIST_SWEEP_WORK_MAX,
        unit="每个 n 的链数",
    )

    res = fjc_core.sweep(
        n_list, l_f, seed_i, chains_i,
        max_points=ASSIST_SWEEP_N_MAX,
        max_chains=ASSIST_SWEEP_CHAINS_MAX,
        work_max=ASSIST_SWEEP_WORK_MAX,
    )

    rows = []
    for i, n_i in enumerate(res.ns):
        rows.append({
            "n": n_i,
            "R2_mean": res.R2_mean[i],
            "R2_theory": res.R2_theory[i],
            "rel_dev": res.rel_dev[i],
            "rel_dev_percent": res.rel_dev[i] * 100,
            "Cn_sim": res.Cn_sim[i],
        })

    out = {
        "l": res.l,
        "seed": res.seed,
        "chains": res.chains,
        "rel_se": res.rel_se,
        "rel_se_percent": res.rel_se * 100,
        "slope": res.slope,
        "intercept": res.intercept,
        "fit_r2": res.fit_r2,
        "rows": rows,
        "warnings": res.warnings,
        "note": (
            "slope 是 log⟨R²⟩ 对 log n 的拟合斜率，理论值是 1。"
            f"每个点的相对标准误约 {res.rel_se * 100:.2f}%，"
            "所以 rel_dev 落在 ±2~3 倍 rel_se 内都属正常。"
        ),
    }
    if note:
        out["clamped"] = note
    return out


def _t_freerotating_check(n, l=1.0, theta_deg=90.0) -> dict:
    """自由旋转链交叉验证：θ=90° 应退化为 FJC 的 nl²。"""
    n_i, l_f = fjc_core.validate(n, l)
    try:
        theta = float(theta_deg)
    except (TypeError, ValueError):
        raise ToolError("键角 theta_deg 必须是数字（角度制）") from None
    if not math.isfinite(theta) or theta <= 0.0 or theta >= 180.0:
        # θ=180° 时 1+cosθ=0，公式发散；θ=0° 时退化为 ⟨h²⟩=0
        raise ToolError("键角必须在 0° 与 180° 之间（不含两端）")

    h2_fr = fjc_core.freerotating_h2(n_i, l_f, theta)
    fjc_h2 = fjc_core.compute(n_i, l_f).h2
    out = {
        "n": n_i,
        "l": l_f,
        "theta_deg": theta,
        "h2_freerotating": h2_fr,
        "h2_fjc": fjc_h2,
        "ratio": h2_fr / fjc_h2,
        "Cn_equivalent": h2_fr / (n_i * l_f * l_f),
        "note": (
            "⟨h²⟩ = nl²(1−cosθ)/(1+cosθ)。θ=90° 时 ratio 应为 1（退化回 FJC）；"
            "聚乙烯键角 arccos(−1/3)≈109.47° 时 Cn 应为 2，正是教科书值。"
            "这条只做交叉验证，不参与主计算路径。"
        ),
    }
    return out


def _t_solve_n(h, l=1.0, kind="h_rms") -> dict:
    """由特征末端距反解链段数。kind 的合法性由 fjc_core.solve_n 自己判。"""
    r = fjc_core.solve_n(h, l, kind)
    out = r.to_dict()
    out["note"] = (
        f"kind = {r.kind_label}。正向公式与 compute() 是同一棵表达式树，"
        "所以这里反出来的 n 喂回 compute() 会得到同一个 h。"
        "n_exact 是连续解（链段数本来该是整数，这里没替你取整）；"
        "n_floor 给出的 h **不超过**目标、n_ceil 给出的 h **不低于**目标 —— "
        "推荐值 n 取的是 ⌈⌉。"
    )
    return out


def _t_force_extension(n=100, l=1.0, temperature=None, lam=None, x=None,
                       force_pn=None) -> dict:
    """λ ↔ 无量纲力 x ↔ 力（pN）。

    **不回传那 400 个曲线点** —— 和 compute_statistics 不回传 P(h) 曲线是同一条
    理由：模型要的是「拉到 λ 需要多大力」这一个数，不是一整条曲线。
    界面上那张图才是给眼睛看的。
    """
    temp = fjc_core.DEFAULT_TEMPERATURE if temperature is None else temperature
    n_i, l_f = fjc_core.validate(n, l)

    def _force(xv: float) -> float:
        return fjc_core.force_in_pn(xv, temp, l_f)

    out: dict = {
        "n": n_i,
        "l": l_f,
        "temperature": temp,
        # 单位是调用方定的，这里只声明换算时用了哪个 —— 否则会出现
        # 「按 l=1 算出来的 pN」被当成「按 l=3nm 算的」这种误会
        "unit_note": (
            "换算成 pN 时**假设 l 以 nm 计、temperature 以开尔文计**。"
            "λ = ⟨x⟩/(n·l) 与 n、l 都无关（Langevin 函数里根本没有它们），"
            "只有 absolute extension 和 pN 这两个数依赖 n、l、T。"
            "小力（x≪1）时 λ ≈ x/3，正是高斯链的熵弹性 3k_BT⟨x⟩/(n·l²) = f。"
        ),
        # 一张常查的锚点表：大多数问题问的就是这几个拉伸比例
        "anchors": [
            {
                "lam": lv,
                "x": fjc_core.inverse_langevin(lv),
                "force_pn": _force(fjc_core.inverse_langevin(lv)),
                "extension": n_i * l_f * lv,
            }
            for lv in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
        ],
    }

    if lam is not None:
        # 「要拉到 λ 需要多大力」—— 反解 Langevin，没有解析式，内核用二分解
        lv = float(lam)
        xv = fjc_core.inverse_langevin(lv)   # λ ≥ 1 或非数在这里被中文拒掉
        out["to_reach_lam"] = {
            "lam": lv,
            "x": xv,
            "force_pn": _force(xv),
            "extension": n_i * l_f * lv,
        }

    if x is not None:
        # 「这个力能把链拉到多长」—— 正向，直接 Langevin
        xv = float(x)
        lv = fjc_core.langevin(xv)
        out["at_this_x"] = {
            "x": xv,
            "lam": lv,
            "force_pn": _force(xv),   # x < 0 时 force_in_pn 会中文拒掉
            "extension": n_i * l_f * lv,
        }

    if force_pn is not None:
        # 用户问的是 pN（“0.5 pN 能拉多长”），而 Langevin 只认无量纲 x。
        # 换算是线性的，交给内核；这里只负责把结果摆成和上面一样的形状。
        fp = float(force_pn)
        xv = fjc_core.x_from_force_pn(fp, temp, l_f)
        lv = fjc_core.langevin(xv)
        out["at_this_force_pn"] = {
            "force_pn": fp,
            "x": xv,
            "lam": lv,
            "extension": n_i * l_f * lv,
        }

    return out


# --- 前端工具（只有 schema，由浏览器执行）-------------------------------

_N = _p("integer", "链段数 n，≥1")
_L = _p("number", "链段长度 l（Kuhn 长度），>0，默认 1")


TOOLS: list[Tool] = [
    # ---- 服务端 ----
    Tool(
        "compute_statistics",
        "算一个 n 下的全部 FJC 统计量：h_rms=√(nl²)（根均方末端距，主输出）、"
        "h*=l√(2n/3)（最可几）、⟨h⟩=l√(8n/(3π))、σ=l√(n/3)、Rg=√(nl²/6)、"
        "全伸展长度 nl、特征比 Cn。任何「某个 n 的 h 是多少」都走这里，"
        "不要自己算。",
        {"n": _N, "l": _L},
        ("n",),
        handler=_t_compute_statistics,
    ),
    Tool(
        "compare_n",
        "对比多个 n 的同一组统计量，并给出各自相对第一个 n 的倍数。"
        "h_rms ∝ √n，所以 n 翻四倍 h_rms 只翻一倍 —— 要讲标度关系就用这个。"
        f"最多 {ASSIST_COMPARE_MAX} 个 n。",
        {"ns": _p("array", "链段数数组，如 [100, 1000]", items={"type": "integer"}),
         "l": _L},
        ("ns",),
        handler=_t_compare_n,
    ),
    Tool(
        "simulate_chain",
        "随机生成若干条 FJC 链（一次实现的几何），返回每条链的末端距 |R|、"
        "样本 ⟨R²⟩ 与理论 nl² 的对照、以及该有的涨落幅度 R2_rel_se。"
        "用来回答「随机走一条链会走多远」「R 和 h_rms 差这么多正常吗」。"
        "**不返回顶点坐标** —— 要在界面上画出来请改用 set_chain_params。",
        {"n": _N, "l": _L,
         "seed": _p("integer", "随机种子；不给就现取一个并回传，便于复现"),
         "chains": _p("integer", "链数，默认 1，最多 2000。链数越大 ⟨R²⟩ 越接近理论值")},
        ("n",),
        handler=_t_simulate_chain,
    ),
    Tool(
        "run_sweep",
        "链长扫描：对一组 n 各采若干条独立链，拟合 log⟨R²⟩ 对 log n 的斜率"
        "（理论值 1），给出每个 n 的模拟 ⟨R²⟩ 与 nl² 的相对偏差。"
        "回答「⟨h²⟩=nl² 凭什么可信」「标度律对不对」这类问题用它。"
        "这个工具偏慢（约 1 秒），一次问清再调。",
        {"ns": _p("array", "链段数数组，至少 2 个，如 [10, 100, 1000]",
                  items={"type": "integer"}),
         "l": _L,
         "chains": _p("integer", f"每个 n 的链数，默认 {ASSIST_SWEEP_DEFAULT_CHAINS}，"
                                 f"最多 {ASSIST_SWEEP_CHAINS_MAX}"),
         "seed": _p("integer", "随机种子；不给就现取一个并回传")},
        ("ns",),
        handler=_t_run_sweep,
    ),
    Tool(
        "freerotating_check",
        "自由旋转链交叉验证：⟨h²⟩ = nl²(1−cosθ)/(1+cosθ)。"
        "θ=90° 时退化为 FJC 的 nl²，θ≈109.47°（聚乙烯键角）时 Cn=2。"
        "用来验证 FJC 结果和其它链模型一致。",
        {"n": _N, "l": _L,
         "theta_deg": _p("number", "键角（角度制），介于 0 与 180 之间，默认 90")},
        ("n",),
        handler=_t_freerotating_check,
    ),
    Tool(
        "solve_n",
        "反解：已知某个特征末端距 h，要多少个链段 n。"
        "**必须说清是哪一种 h** —— 在同一个 n 下 h_rms / ⟨h⟩ / h* / nl 能差一到两个"
        "数量级，只说「要 100」而不说哪种 100，答案没有意义。"
        "返回连续解 n_exact，以及夹住它的 ⌊⌋（算出来的 h 不超过目标）与 ⌈⌉"
        "（不低于目标）；推荐值取 ⌈⌉。注意 h 是以 l 为单位的长度，不是无量纲数。",
        {"h": _p("number", "目标末端距，**和 l 同一个单位**，必须 > 0"),
         "l": _L,
         "kind": _p("string",
                    "要反解的 h 是哪一种：h_rms 根均方（默认）| h_mean 平均 ⟨h⟩ | "
                    "h_mp 最可几 h* | h_max 全伸展 nl。"
                    "这四种在同一个 n 下差别很大，不给就默认 h_rms",
                    enum=list(fjc_core.H_KINDS))},
        ("h",),
        handler=_t_solve_n,
    ),
    Tool(
        "force_extension",
        "力–伸长：FJC 在外力下的拉伸。横轴是无量纲力 x = f·l/(k_B·T)（不是牛顿），"
        "纵轴是归一化伸长 λ = ⟨x⟩/(n·l) ∈ [0,1)。"
        "回答「拉到 60% 需要多大力」「0.5 pN 能拉多长」这类问题用它，**不要自己算**"
        "（Langevin 反函数没有解析式，硬写会错）。"
        "传 lam 就正着解出所需 x 与力，传 x 或 force_pn 就反着解出伸长，"
        "多个都传就都给；不传则只回一张常用锚点表。"
        "**用力就用 force_pn（单位 pN），不要用 x** —— x 是无量纲的 f·l/(k_BT)，"
        "直接拿牛顿/pN 的数当 x 会差出 ~243 倍（室温、l=1nm 时）。",
        {"lam": _p("number", "要达到的归一化伸长 λ，必须 ≥ 0 且 < 1（0.5 = 拉到全长的一半）"),
         "force_pn": _p("number", "力，单位 **pN**，必须 ≥ 0。答「多大的力能拉多长」用这个"),
         "x": _p("number", "无量纲力 x = f·l/(k_BT)，必须 ≥ 0。"
                           "只有对方明确说了「无量纲力」才用，否则用 force_pn"),
         "l": _L,
         "n": _p("integer", "链段数 n，≥1。只用来把 λ 换算成绝对伸长 ⟨x⟩=n·l·λ，"
                            "λ 和力本身都与 n 无关。默认 100"),
         "temperature": _p("number", "温度，开尔文，0<T<1000，默认 298.15。只影响 pN 换算")},
        (),
        handler=_t_force_extension,
    ),

    # ---- 前端：读 ----
    Tool(
        "get_page_state",
        "读当前界面状态：主参数 n / l、是否归一化、正在对比的曲线集合、"
        "3D 卡片的 n / 链数 / 种子 / 视角、扫描卡片的参数与是否已有结果。"
        "用户说「这个」「当前」的时候先调它，不要猜界面上是什么。",
        {},
        (),
        kind="frontend",
        readonly=True,
    ),

    # ---- 前端：改界面 ----
    Tool(
        "set_params",
        "改顶部参数行的主参数 n 和/或 l，等价于用户自己在输入框里改，图表会重画。"
        "用户说「把 n 改成 400」时用它。注意这只改主分布图的 n，"
        "3D 卡片的 n 是独立的，要改那边用 set_chain_params。",
        {"n": _p("integer", "新的链段数，≥1；不改就别传"),
         "l": _p("number", "新的链段长度，>0；不改就别传")},
        (),
        kind="frontend",
    ),
    Tool(
        "set_curves",
        "设置主分布图上叠加对比的曲线集合（最多 5 个 n）。"
        "「把 100 和 1000 放一起看」用这个；它只改**对比集合**，"
        "会和 n 一起决定画出哪几条曲线。",
        {"ns": _p("array", "要对比的链段数数组，最多 5 个", items={"type": "integer"})},
        ("ns",),
        kind="frontend",
    ),
    Tool(
        "set_normalize",
        "开关归一化显示。打开后横轴变 h/h_rms、纵轴变 P·h_rms，"
        "各 n 的曲线会精确重合（这正是 √n 标度律在整个分布上成立的证据）。",
        {"on": _p("boolean", "true 打开，false 关闭")},
        ("on",),
        kind="frontend",
    ),
    Tool(
        "set_chain_params",
        "设置 3D 单链构象卡片的参数：n、链数、随机种子（3D 卡片用**主界面的 l**，"
        "所以这里没有 l）。改完会重新向 /api/chain 取一条链并重画。"
        "用户说「画一条 n=500 的链」时用它。",
        {"n": _p("integer", "3D 卡片的链段数，1~20000；不改就别传"),
         "chains": _p("integer", "同时叠画几条链，1~5；不改就别传"),
         "seed": _p("integer", "随机种子，非负整数；传了就能复现同一条链")},
        (),
        kind="frontend",
    ),
    Tool(
        "set_chain_view",
        "设置 3D 视角与缩放，不重新取数、只重画。azim 是绕竖直轴的水平角、"
        "elev 是仰角，都用角度制；zoom 是与初始视角的比值。",
        {"azim": _p("number", "水平角（度），可为任意实数"),
         "elev": _p("number", "仰角（度），会被夹在 −89~89"),
         "zoom": _p("number", "缩放，0.4~4，1 是初始大小")},
        (),
        kind="frontend",
    ),
    Tool(
        "play_chain",
        "播放链生长动画（从原点一段一段长出来）。"
        "用户说「播放」「演示一下链是怎么长的」时用它。",
        {},
        (),
        kind="frontend",
    ),
    Tool(
        "set_sweep_params",
        "设置链长扫描卡片的参数（n 列表、每个 n 的链数 M、随机种子），"
        "只填参数不运行。「把扫描改成 10000 条链」用这个。",
        {"ns": _p("array", "要扫的链段数数组", items={"type": "integer"}),
         "chains": _p("integer", f"每个 n 的链数，1~{fjc_core.SWEEP_CHAINS_MAX:,}；不改就别传"),
         "seed": _p("integer", "随机种子；不改就别传")},
        (),
        kind="frontend",
    ),
    Tool(
        "run_sweep_view",
        "点扫描卡片上的「运行模拟」按钮，用当前参数跑一次并画出结果。"
        "注意这走的是界面的预算（比 run_sweep 工具大得多，最坏约 4 秒），"
        "只是想问个数值就用 run_sweep，不要用这个。",
        {},
        (),
        kind="frontend",
    ),
]

TOOL_BY_NAME: dict[str, Tool] = {t.name: t for t in TOOLS}
SERVER_TOOLS = [t for t in TOOLS if t.kind == "server"]
FRONTEND_TOOLS = [t for t in TOOLS if t.kind == "frontend"]

# 助手**不做**的动作，但要知道它们存在（用户问「怎么导出」时要答得上来）。
# 导出会落盘、复制会进剪贴板 —— 这类动作留给人点按钮。
NOT_TOOLS = {
    "导出 PNG": "3D 卡片控件行里的「PNG」按钮",
    "导出 3D 顶点 CSV": "3D 卡片控件行里的「CSV」按钮",
    "导出扫描汇总 CSV": "扫描卡片控件行里的「汇总 CSV」按钮",
    "导出扫描完整 JSON": "扫描卡片控件行里的「完整 JSON」按钮",
    "复制数值表": "数值表卡片右上角的「复制」按钮",
}


# --- 执行 ----------------------------------------------------------------

def openai_tools() -> list[dict]:
    """渲染成请求体里的 tools 数组。"""
    return [t.schema() for t in TOOLS]


def tools_prompt() -> str:
    """渲染成系统提示词里的一段人话说明。

    这是 `tools` 不被网关支持时的**退路**：模型照样知道有哪些工具、
    参数叫什么。调用时用一个 JSON 对象表达，形如
    {"tool": "compute_statistics", "args": {"n": 400}}。
    """
    lines = []
    for t in TOOLS:
        if t.kind == "frontend":
            scope = "（由浏览器执行，只读）" if t.readonly else "（由浏览器执行，会改动界面）"
        else:
            scope = ""
        sig = ", ".join(
            f"{name}" + ("" if name in t.required else "?")
            for name in t.params
        )
        lines.append(f"- {t.name}({sig}){scope}：{t.summary}")
    return "\n".join(lines)


def capability_list() -> str:
    """给本地兜底用的短能力清单（比 tools_prompt 短，是给人看的一句话）。"""
    return (
        "算统计量（h_rms、h*、⟨h⟩、σ、Rg、全伸展长度）、多个 n 横向对比、"
        "随机生成单链构象、⟨R²⟩ 对 n 的标度扫描、自由旋转链交叉验证、"
        "由末端距反解链段数、力–伸长曲线（Langevin 反函数：拉到某个比例要多大力），"
        "以及改界面上的参数（n、l、归一化、3D 卡片的 n / 链数 / 种子 / 视角、扫描参数）"
    )


def run(name: str, args: Any) -> dict:
    """执行一个**服务端**工具。

    一律返回 {"ok": bool, ...}，不向调用方抛物理错误 —— 参数错了要让模型看见
    并能自己改，而 (a) 把中文报错当工具结果回灌是最省事、也最有效的做法。
    """
    tool = TOOL_BY_NAME.get(name)
    if tool is None:
        return {"ok": False, "error": f"没有 {name} 这个工具"}
    if tool.kind != "server":
        return {"ok": False, "error": f"{name} 是界面工具，应该在浏览器里执行"}
    if tool.handler is None:
        return {"ok": False, "error": f"{name} 没有实现"}
    if not isinstance(args, dict):
        return {"ok": False, "error": "工具参数必须是一个 JSON 对象"}

    unknown = sorted(set(args) - set(tool.params))
    if unknown:
        return {
            "ok": False,
            "error": (f"不认识的参数：{'、'.join(unknown)}；"
                      f"{name} 的参数是 {'、'.join(tool.params) or '（无）'}"),
        }
    missing = [p for p in tool.required if args.get(p) is None]
    if missing:
        return {"ok": False, "error": f"缺少必填参数：{'、'.join(missing)}"}

    try:
        return {"ok": True, "data": tool.handler(**args)}
    except FJCInputError as e:
        # 物理内核的校验信息本来就是写给用户看的中文，直接转给模型
        return {"ok": False, "error": str(e)}
    except ToolError as e:
        return {"ok": False, "error": str(e)}
    except (TypeError, ValueError) as e:
        return {"ok": False, "error": f"参数类型不对：{e}"}
