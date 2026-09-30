"""链模型物理内核：自由连接链 + 它的三个邻居（自由旋转链 / 受阻旋转链 / 蠕虫状链）。

默认模型是自由连接链（FJC）：n 个长度为 l 的链段自由连接，无键角限制、无位阻，
链段取向互相独立，于是 ⟨h²⟩ = n·l² 是**精确解**（不是近似）。另外三个模型
各自加上一层更真实的约束，见下面「链模型」那一节 —— 每个模型只定义一处公式，
compute / random_chain / sweep / API / AI 工具全部从那里取。

末端距的分布 P(h) 一律用**等效高斯链**：把 ⟨h²⟩ 换成当前模型的值，分布形状不变。
这在等效 Kuhn 段数 L/b 足够大时是标准做法，小了会在 warnings 里说清楚。

本模块不 import flask，可单独 import 或测试。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

# 链段数上限：再大既没有物理意义，也会让浮点失去精度
N_MAX = 10_000_000
# 分布曲线的采样点数
CURVE_POINTS = 400
# 等效 Kuhn 段数 L/b 低于此值时，等效高斯近似不可靠（FJC 时 L/b 就是 n）
GAUSSIAN_N_MIN = 10

# --- 单链构象模拟（3D 视图）的参数 ---
# 画出来的链段数上限：再多在屏幕上就是一团结，看不出构象
CHAIN_N_MAX = 20_000
# 一次最多画几条链
MAX_CHAINS = 5
# 全部链的顶点总数预算，防止 chains × n 一起放大把 JSON 撑爆
CHAIN_POINTS_MAX = 60_000
# 顶点坐标回传时保留的小数位（l=1、链尺度 ~√n 时远够用，体积小一个量级）
CHAIN_DECIMALS = 4

# --- 链长扫描（⟨R²⟩ 对 n 的标度关系）---
# 一次最多扫几个 n（等比序列 10…1000 是 15 个，真正的护栏是下面的计算量预算）
SWEEP_N_MAX = 20
# 每个 n 最多采几条独立链。这里能比 MAX_CHAINS 大四个数量级，因为扫描
# **只要末端矢量、不要顶点**：内存是 O(chains)，与 n 无关，所以 M=10000 也吃得下。
SWEEP_CHAINS_MAX = 100_000
# 每个 n 的默认链数。⟨R²⟩ 的相对标准误 = √(2/3)/√M，M=10000 时约 0.82%
SWEEP_DEFAULT_CHAINS = 10_000
# Σ(chains × n) 的总预算，防止一次请求把 CPU 占死。
# 实测约 1500 万/秒，60M 对应最坏 ~4 秒 —— 前端会在这期间显示忙状态。
SWEEP_WORK_MAX = 60_000_000
# 单次分块的元素数上限：分块只是为了控内存，不改变统计量
_WORK_CHUNK = 2_000_000


# ---------- 链模型：FJC 与它的三个邻居 --------------------------------------
#
# 四个模型可以压成同一句话：**长链极限下 ⟨h²⟩ = L·b**（L = n·l 是轮廓长度，
# b 是等价 Kuhn 长度）。差别只有两处：
#
#   1) b 是多少 —— 它就是特征比 Cn = b/l = ⟨h²⟩/(n·l²)。
#      FJC 恒为 1；自由旋转链由键角定；受阻旋转链再乘一个内旋转因子；
#      蠕虫状链直接由持久长度给出（b = 2p）。
#   2) 短链端怎么过渡 —— 前三个是「n 个链段」的离散图景，⟨h²⟩ = n·l²·Cn
#      对一切 n 都成立；WLC 是连续杆状链，短链端退化成刚杆（⟨h²⟩ → L²），
#      所以它有自己的一条式子，不能写成 n·l²·Cn。
#
# **四个模型只在这里定义一处**：compute / random_chain / sweep / API / AI 工具
# 全部从下面这些常量与函数取，谁都不许另写一套公式 —— 写了就会改一处忘一处。

MODEL_FJC = "fjc"
MODEL_FRC = "frc"
MODEL_HINDERED = "hindered"
MODEL_WLC = "wlc"

MODEL_KEYS = (MODEL_FJC, MODEL_FRC, MODEL_HINDERED, MODEL_WLC)

MODEL_LABELS = {
    MODEL_FJC: "自由连接链 FJC",
    MODEL_FRC: "自由旋转链 FRC",
    MODEL_HINDERED: "受阻旋转链",
    MODEL_WLC: "蠕虫状链 WLC",
}

DEFAULT_MODEL = MODEL_FJC

# 参数默认值都取「教科书里最常见的那一个」：
DEFAULT_THETA_DEG = 109.47   # 聚乙烯 sp³ 键角 arccos(−1/3) —— Cn 恰好是 2
DEFAULT_COS_PHI = 0.5        # 内旋转平均余弦，聚乙烯约 0.5（位阻因子 ≈ 3）
DEFAULT_P = 10.0             # 持久长度（单位是 l）：链长 100l 时约 10 个持久长度
# 三态（trans / gauche±）分布能表达的平均余弦范围：再负就只有「全 gauche」那一种
COS_PHI_MIN = -0.5
COS_PHI_MAX = 0.99
# 数值护栏。θ 顶到 180° 时 1+cosθ→0、特征比发散；p 太大时采样里的 e^(2κ) 会溢出
THETA_EPS_DEG = 0.01
P_MAX = 300.0
# 采样器里 κ 的二分次数：区间宽 400，120 轮后远小于双精度分辨率
_KAPPA_ITER = 120


class FJCInputError(ValueError):
    """输入不合法。server 层把它转成 HTTP 400。"""


@dataclass
class FJCResult:
    """单个 n 的全部输出量。长度单位的量都以 l 的单位为准。"""

    n: int
    l: float

    h2: float       # ⟨h²⟩ = n·l²，均方末端距
    h_rms: float    # √⟨h²⟩ = l√n，根均方末端距（主输出）
    h_mp: float     # h* = l√(2n/3)，最可几末端距
    h_mean: float   # ⟨h⟩ = l√(8n/(3π))，平均末端距
    h_max: float    # n·l，全伸展长度

    Rg2: float      # ⟨Rg²⟩ = ⟨h²⟩/6
    Rg_rms: float   # √⟨Rg²⟩

    beta: float     # β = √(3/(2⟨h²⟩))，等效高斯链的分布参数
    sigma: float    # σ = √(⟨h²⟩/3)，分布的标准差
    Cn: float       # 特征比 ⟨h²⟩/(n·l²)：FJC 恒为 1，其余模型由参数定

    curve_h: np.ndarray = field(repr=False)
    curve_P: np.ndarray = field(repr=False)

    # --- 模型信息（四个链模型共用这一套结果结构）---
    # 前三个模型的 ⟨h²⟩ = n·l²·Cn 对一切 n 成立；WLC 的 Cn 依赖链长，
    # 所以 ⟨h²⟩ 一律读 h2 本身，别拿 Cn 去乘 n·l²。
    model: str = DEFAULT_MODEL   # fjc / frc / hindered / wlc
    kuhn_length: float = 0.0     # 等价 Kuhn 长度 b（长度单位跟随 l）
    n_kuhn: float = 0.0          # 等效 Kuhn 段数 L/b —— 高斯近似能不能用就看它
    params: dict = field(default_factory=dict)   # 参数回显，前端读数条直接用

    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """转成可 JSON 序列化的 dict。"""
        return {
            "n": self.n,
            "l": self.l,
            "h2": self.h2,
            "h_rms": self.h_rms,
            "h_mp": self.h_mp,
            "h_mean": self.h_mean,
            "h_max": self.h_max,
            "Rg2": self.Rg2,
            "Rg_rms": self.Rg_rms,
            "beta": self.beta,
            "sigma": self.sigma,
            "Cn": self.Cn,
            "model": self.model,
            "kuhn_length": self.kuhn_length,
            "n_kuhn": self.n_kuhn,
            "params": self.params,
            "curve": {
                "h": [float(v) for v in self.curve_h],
                "P": [float(v) for v in self.curve_P],
            },
            "warnings": self.warnings,
        }


@dataclass
class ChainResult:
    """单链构象模拟的结果：一次实现的几何 + 与之对照的理论量。

    注意这里和 FJCResult 是两件不同的事：FJCResult 全是**系综平均**（解析式），
    本类是一次**随机实现**。单条链的 R 偏离 h_rms 是正常的，只有多条链的
    ⟨R²⟩ 才收敛到 n·l² —— 这正是 3D 视图要讲的事。
    """

    n: int
    l: float
    seed: int
    chains: int

    points: np.ndarray   # (chains, n+1, 3) 各顶点坐标，第 0 个是原点
    R: np.ndarray        # (chains, 3) 末端矢量
    R_mag: np.ndarray    # (chains,) |R|

    R2_mean: float       # 这些链的 ⟨R²⟩
    R2_theory: float     # 本模型的 ⟨h²⟩（FJC 时就是 n·l²）
    R2_rel_se: float     # ⟨R²⟩ 的相对标准误，见 random_chain() 里的推导

    h_rms: float         # l√n
    h_mp: float          # l√(2n/3)
    h_mean: float        # l√(8n/(3π))
    h_max: float         # n·l

    model: str = ""                              # 采样用的链模型
    params: dict = field(default_factory=dict)   # 模型参数回显
    kuhn_length: float = 0.0                     # 等价 Kuhn 长度 b（3D 视图据此标刻度）

    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """转成可 JSON 序列化的 dict。

        顶点用**扁平**数组 [x0,y0,z0,x1,y1,z1,…] 并限位小数：
        n=1000 时约 40KB，比嵌套数组 + 全精度小一个量级。
        """
        q = CHAIN_DECIMALS
        return {
            "n": self.n,
            "l": self.l,
            "seed": self.seed,
            "chains": self.chains,
            "R2_theory": self.R2_theory,
            "R2_mean": self.R2_mean,
            "R2_rel_se": self.R2_rel_se,
            "h_rms": self.h_rms,
            "h_mp": self.h_mp,
            "h_mean": self.h_mean,
            "h_max": self.h_max,
            "model": self.model,
            "params": self.params,
            "kuhn_length": self.kuhn_length,
            "R_mag": [round(float(v), q) for v in self.R_mag],
            "R": [[round(float(v), q) for v in row] for row in self.R],
            "points": [
                [round(float(v), q) for v in row.ravel()] for row in self.points
            ],
            "warnings": self.warnings,
        }


@dataclass
class SweepResult:
    """链长扫描：一组 n 的模拟 ⟨R²⟩ 与理论 n·l² 的对照。

    和 ChainResult 一样，这里混着两种量：R2_mean / h_rms_sim / Cn_sim 来自
    `chains` 条链的**有限样本**，R2_theory / h_rms 是解析值。有限样本的偏差
    由 rel_se 定标 —— 判断「偏了多少算正常」全靠它。
    """

    l: float
    seed: int
    chains: int
    ns: list[int]

    R2_mean: list[float]     # 每个 n 的模拟 ⟨R²⟩
    R2_theory: list[float]   # n·l²
    h_rms_sim: list[float]   # √R2_mean
    h_rms: list[float]       # l√n
    Cn_sim: list[float]      # R2_mean / R2_theory（模型自己的 ⟨h²⟩），应当 ≈ 1
    rel_dev: list[float]     # (R2_mean − R2_theory)/R2_theory

    rel_se: float            # ⟨R²⟩ 的相对标准误 √(2/3)/√M，与 n 无关
    slope: float | None      # log⟨R²⟩ 对 log n 的拟合斜率
    intercept: float | None
    fit_r2: float | None     # 该拟合的决定系数
    slope_se: float | None   # 斜率自身的标准误，比 rel_se 小 —— n 排得越开越准

    model: str = ""                              # 扫描用的链模型
    params: dict = field(default_factory=dict)   # 模型参数回显
    slope_ref: float | None = None               # 本模型的理论斜率（FJC 恒为 1）

    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """转成可 JSON 序列化的 dict。只有十几个点，不必压体积。"""
        return {
            "l": self.l,
            "seed": self.seed,
            "chains": self.chains,
            "n": list(self.ns),
            "R2_mean": [float(v) for v in self.R2_mean],
            "R2_theory": [float(v) for v in self.R2_theory],
            "h_rms_sim": [float(v) for v in self.h_rms_sim],
            "h_rms": [float(v) for v in self.h_rms],
            "Cn_sim": [float(v) for v in self.Cn_sim],
            "rel_dev": [float(v) for v in self.rel_dev],
            "rel_se": self.rel_se,
            "slope": None if self.slope is None else float(self.slope),
            "intercept": None if self.intercept is None else float(self.intercept),
            "fit_r2": None if self.fit_r2 is None else float(self.fit_r2),
            "slope_se": None if self.slope_se is None else float(self.slope_se),
            "model": self.model,
            "params": self.params,
            "slope_ref": None if self.slope_ref is None else float(self.slope_ref),
            "warnings": self.warnings,
        }


def _as_int(value, name: str) -> int:
    """把输入强转成整数，失败时抛带中文信息的 FJCInputError。"""
    if isinstance(value, bool) or (isinstance(value, str) and not value.strip()):
        raise FJCInputError(f"{name} 必须是整数")
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise FJCInputError(f"{name} 必须是整数") from None
    if not math.isfinite(f):
        raise FJCInputError(f"{name} 必须是有限数")
    if f != int(f):
        raise FJCInputError(f"{name} 必须是整数，收到 {value}")
    return int(f)


def _as_seed(seed) -> int:
    """没给种子就现取一个并回传，前端显示出来即可复现。"""
    if seed is None or (isinstance(seed, str) and not seed.strip()):
        return int(np.random.default_rng().integers(0, 1 << 31))
    seed_i = _as_int(seed, "随机种子")
    if seed_i < 0:
        raise FJCInputError("随机种子必须是非负整数")
    if seed_i > (1 << 63) - 1:
        raise FJCInputError("随机种子太大")
    return seed_i

# ---------- 模型层：类型与函数 ----------------------------------------------
# （「四个模型各是什么」那段说明与全部模型常量在文件开头。）

@dataclass(frozen=True)
class ModelParams:
    """一个链模型 + 它的参数。compute / random_chain / sweep / API 共用这一个入口。

    一律用 `ModelParams.of(...)` 构造 —— 校验和填默认都在那里，是**唯一**一处：
    server 只把请求体里的几个字段喂进来，中文错误信息也从这里出去。
    p 的单位是 l（和这个软件其余部分一致：结果的长度单位就是 l 的单位）。
    """

    key: str = DEFAULT_MODEL
    theta_deg: float = DEFAULT_THETA_DEG   # 键角 θ（化学意义上的 ∠C–C–C）
    cos_phi: float = DEFAULT_COS_PHI       # 内旋转平均余弦 ⟨cosφ⟩
    p: float = DEFAULT_P                   # 持久长度，单位 = l

    @classmethod
    def of(cls, model=None, theta_deg=None, cos_phi=None, p=None) -> "ModelParams":
        key = DEFAULT_MODEL if model in (None, "") else str(model).strip().lower()
        if key not in MODEL_KEYS:
            raise FJCInputError(
                "链模型只能是 " + " / ".join(MODEL_KEYS) + f" 之一，收到 {model!r}"
            )
        return cls(
            key=key,
            theta_deg=_as_angle(theta_deg),
            cos_phi=_as_cos_phi(cos_phi),
            p=_as_persistence(p),
        )

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "label": MODEL_LABELS[self.key],
            "theta_deg": self.theta_deg,
            "cos_phi": self.cos_phi,
            "p": self.p,
        }


def _as_float(value, name: str, default: float) -> float:
    if value is None or value == "":
        return float(default)
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise FJCInputError(f"{name} 必须是数字") from None
    if not math.isfinite(f):
        raise FJCInputError(f"{name} 必须是有限数")
    return f


def _as_angle(value) -> float:
    theta = _as_float(value, "键角 θ", DEFAULT_THETA_DEG)
    if not (THETA_EPS_DEG <= theta <= 180.0 - THETA_EPS_DEG):
        raise FJCInputError(
            f"键角 θ 必须在 {THETA_EPS_DEG}° 与 {180.0 - THETA_EPS_DEG}° 之间："
            f"0° 会把链折成一点，180° 是直杆，公式在两端都失效"
        )
    return theta


def _as_cos_phi(value) -> float:
    c = _as_float(value, "内旋转平均余弦 ⟨cosφ⟩", DEFAULT_COS_PHI)
    if not (COS_PHI_MIN <= c <= COS_PHI_MAX):
        raise FJCInputError(
            f"内旋转平均余弦 ⟨cosφ⟩ 必须在 {COS_PHI_MIN} 与 {COS_PHI_MAX} 之间"
            f"（0 表示绕键完全自由，正好退回自由旋转链）"
        )
    return c


def _as_persistence(value) -> float:
    p = _as_float(value, "持久长度 p", DEFAULT_P)
    if p <= 0:
        raise FJCInputError("持久长度 p 必须大于 0")
    if p > P_MAX:
        raise FJCInputError(f"持久长度 p 不能超过 {P_MAX:g}（单位是 l）")
    return p


def coerce_params(params) -> ModelParams:
    """把「None / 字符串 / dict / ModelParams」统一成 ModelParams。

    字符串（"frc"）和 dict（API 里的请求体）都收，是为了让调用点干净：
    **校验只有 ModelParams.of 一处**，这里只做类型分派。
    """
    if params is None:
        return ModelParams()
    if isinstance(params, ModelParams):
        return params
    if isinstance(params, str):
        return ModelParams.of(params)
    if isinstance(params, dict):
        return ModelParams.of(
            params.get("model", params.get("key")),
            params.get("theta_deg"),
            params.get("cos_phi"),
            params.get("p"),
        )
    raise FJCInputError("链模型参数只能是 ModelParams、dict 或模型名")


def model_kuhn_over_l(mp: ModelParams) -> float:
    """等价 Kuhn 长度 b/l。

    前三个模型它同时就是特征比 Cn（与 n 无关）；WLC 的 b = 2p，但 Cn 还依赖链长
    （⟨h²⟩ 在短链端退化成 L²），要用 model_cn() 拿。
    """
    if mp.key == MODEL_FJC:
        return 1.0
    if mp.key == MODEL_HINDERED:
        c = math.cos(math.radians(mp.theta_deg))
        return ((1.0 - c) / (1.0 + c)) * ((1.0 + mp.cos_phi) / (1.0 - mp.cos_phi))
    if mp.key == MODEL_FRC:
        c = math.cos(math.radians(mp.theta_deg))
        return (1.0 - c) / (1.0 + c)
    # WLC：持久长度 p（单位 l）→ Kuhn 长度 b = 2p
    return 2.0 * mp.p


def model_h2(mp: ModelParams, n: int, l: float) -> float:
    """本模型的均方末端距 ⟨h²⟩（长度单位跟随 l）。

    键角 θ 用**化学意义**上的那个：∠C–C–C。聚乙烯是 109.47°，两个相邻**键矢量**
    之间的夹角是它的补角 70.53°（余弦 +1/3）—— 两者只差一个负号，
    而 (1−cosθ)/(1+cosθ) 在 θ→180°−θ 下正好互换，别把 θ 当成键矢量夹角填。

    **自由旋转链与受阻旋转链用的是有限 n 的精确离散式，不是教科书的渐近式。**
    教科书那条 ⟨h²⟩ = n·l²·Cn 是 n → ∞ 的极限，差一个与 n 无关的常数项；
    n=100、Cn=6 时这个差别有 4%，正好会被「模拟 vs 理论」那张对照图当成异常。
    离散链的精确值就是 ⟨h²⟩ = l²·[n + 2Σₖ(n−k)·⟨u₀·u_k⟩]，
    而 ⟨u₀·u_k⟩ 由方向关联的几何级数给出（自由旋转链只有一项、受阻旋转链两项），
    于是那两重求和能写成闭式。渐近式仍然有用（做交叉验证、报 C∞），见
    freerotating_h2() 与 model_cn_infinite()。
    """
    if mp.key == MODEL_WLC:
        L = n * l
        p = mp.p * l
        # 连续蠕虫状链（Kratky–Porod）：⟨h²⟩ = 2pL[1 − (p/L)(1 − e^(−L/p))]
        # (1 − e^(−x))/x 用 −expm1(−x)/x 算：x 很小时 e^(−x)≈1，直接相减会把
        # 刚杆极限的 ⟨h²⟩ 算成 0（而那正是差得最狠的地方）。
        x = L / p
        return 2.0 * p * L * (1.0 + math.expm1(-x) / x)
    if mp.key == MODEL_FJC:
        return n * l * l

    # c = 相邻键矢量的平均余弦 = −cos(键角)；受阻旋转链再带上 ⟨cosφ⟩。
    # 自由旋转链就是 g=0 的那个特例，两条走同一个和。
    c = -math.cos(math.radians(mp.theta_deg))
    g = mp.cos_phi if mp.key == MODEL_HINDERED else 0.0
    return n * l * l + 2.0 * l * l * _bond_series(c, g, n)


def _series(r, n: int):
    """T(r) = Σ_{k=1}^{n−1} (n−k)·r^k —— 方向关联求和里的那条几何级数。

    闭式 T(r) = r[n(1−r) − (1−r^n)]/(1−r)²。r 可以取复数（受阻旋转链在
    ⟨cosφ⟩<0 时关联是振荡衰减的，特征值成共轭复根）。
    |r| 接近 1 时改走展开式 T ≈ r·n(n−1)/2·[1 − (n−2)(1−r)/3]：
    闭式在那里是「大数相减」，会掉一半有效位；展开式只留一阶，
    而这一带 |1−r| 只有 1e-6，二阶项在 1e-12 量级，够用。
    """
    if n <= 1:
        return 0.0
    if abs(1.0 - r) < 1e-6:
        return r * n * (n - 1) / 2.0 * (1.0 - (n - 2) * (1.0 - r) / 3.0)
    return r * (n * (1.0 - r) - (1.0 - r**n)) / (1.0 - r) ** 2


def _bond_series(c: float, g: float, n: int) -> float:
    """Σ_{k=1}^{n−1}(n−k)·⟨u₀·u_k⟩ —— 离散链 ⟨h²⟩ = l²[n + 2·这个] 的那个和。

    方向关联由 2×2 转移矩阵递推（s² = 1−c²）：

        A_k = c·A_{k−1} + s·g·B_{k−1}          A_k = ⟨u₀·u_k⟩
        B_k = s·A_{k−1} − c·g·B_{k−1}          B_k = ⟨u₀·e₁⁽ᵏ⁾⟩（平面朝向的投影）

    起点 A₀=1、B₀=0。**第二行那个「−c·g·B」项是关键**：它来自
    e₁⁽ᵏ⁾ = s·u_{k−1} − c·f_k 这个几何关系（f_k 是那一步的平面内方向），
    少了它整条链就少了一大截关联（错的那一版会把聚乙烯算成 Cn≈2.9 而不是 6）。

    矩阵的迹是 c(1−g)、**行列式恰好是 −g**，于是特征值
        λ± = [c(1−g) ± √(c²(1−g)² + 4g)] / 2
    配上 A₀=1、A₁=c 定出两个系数，两重求和就化成两条几何级数。
    g=0（绕键自由旋转）时 λ₋=0、系数为 0，只剩 λ=c 一项 —— 正好退回自由旋转链，
    所以自由旋转链也走这一条；g<0 时判别式可能为负，用复数算完取实部。
    """
    if n <= 1:
        return 0.0
    lam1, a1, lam2, a2 = _bond_eigen(c, g)
    return complex(a1 * _series(lam1, n) + a2 * _series(lam2, n)).real


def _bond_eigen(c: float, g: float):
    """方向关联的两个特征根与系数：⟨u₀·u_k⟩ = α₊λ₊^k + α₋λ₋^k。

    转移矩阵 [[c, s·g], [s, −c·g]] 的迹是 c(1−g)、**行列式恰好是 −g**；
    系数由 A₀=1、A₁=c 定出来。g=0（绕键自由旋转）时退化成单个 λ=c。
    ⟨cosφ⟩<0 时判别式可能为负 → 复共轭根（关联是振荡衰减的），调用侧取实部。
    """
    tr = c * (1.0 - g)
    disc = tr * tr + 4.0 * g
    if disc < 0.0:
        root = complex(0.0, math.sqrt(-disc))
        lam1 = (tr + root) / 2.0
        lam2 = (tr - root) / 2.0
    else:
        root = math.sqrt(disc)
        lam1 = (tr + root) / 2.0
        lam2 = (tr - root) / 2.0
    if abs(lam1 - lam2) < 1e-12:
        # 重根：A_k = (1 + kβ)λ^k，需要另一个闭式，不值当。
        # 这是参数空间里一条零测度的线，把参数挪 1e-9 求值即可（函数连续）。
        lam2 = lam1 + 1e-9
    a1 = (c - lam2) / (lam1 - lam2)
    return lam1, a1, lam2, 1.0 - a1


def _corr_k(c: float, g: float, k: int) -> float:
    """⟨u₀·u_k⟩：隔 k 个链段之后，两个键的方向还一致多少。"""
    if k == 0:
        return 1.0
    lam1, a1, lam2, a2 = _bond_eigen(c, g)
    return complex(a1 * lam1 ** k + a2 * lam2 ** k).real


def _n_grid(n_max: int, points: int) -> list[int]:
    """1 … n_max 的对数网格（取整去重），用来画 Cn(n) 的收敛曲线。"""
    out = {1, n_max}
    top = math.log10(max(2, n_max))
    for i in range(points):
        out.add(max(1, int(round(10 ** (i / max(1, points - 1) * top)))))
    return sorted(out)


def model_fingerprint(mp: ModelParams, n_max: int = 1_000_000,
                      points: int = 24, k_max: int = 40) -> dict:
    """模型的「指纹」：**方向关联怎么衰减**、**Cn 怎么随链长收敛到 C∞**。

    两张纯理论曲线，不含任何采样 —— 给页面上「模型指纹」那张卡片画图用。
    它们回答的是同一个问题的两面：四个模型的差别从哪来（关联衰减快慢），
    以及教科书那个 Cn 是怎么在长链极限下才出现的（有限 n 时 Cn < C∞）。
    """
    mp = coerce_params(mp)
    c = -math.cos(math.radians(mp.theta_deg))
    g = mp.cos_phi if mp.key == MODEL_HINDERED else 0.0
    ks = list(range(k_max + 1))

    def corr_of(key: str) -> list[float]:
        if key == MODEL_FJC:
            return [1.0] + [0.0] * k_max          # 链段之间毫无关系
        if key == MODEL_WLC:
            r = math.exp(-1.0 / mp.p)             # ⟨u₀·u_k⟩ = e^(−k·l/p)
            return [r ** k for k in ks]
        if key == MODEL_FRC:
            return [c ** k for k in ks]
        return [_corr_k(c, g, k) for k in ks]     # 受阻旋转：两根之和

    ns = _n_grid(n_max, points)
    compare = []
    for key in MODEL_KEYS:
        other = ModelParams.of(key, theta_deg=mp.theta_deg, cos_phi=mp.cos_phi, p=mp.p)
        compare.append({
            "key": key,
            "label": MODEL_LABELS[key],
            "corr": corr_of(key),
            "cn": [model_cn(other, n, 1.0) for n in ns],
            "cn_infinite": model_cn_infinite(other),
        })
    return {
        "model": mp.key,
        "params": mp.to_dict(),
        "k": ks,
        "ns": ns,
        "corr": corr_of(mp.key),
        "cn": [model_cn(mp, n, 1.0) for n in ns],
        "cn_infinite": model_cn_infinite(mp),
        "compare": compare,
    }


def model_cn_infinite(mp: ModelParams) -> float:
    """长链极限的特征比 C∞ = b/l（教科书里那个数：聚乙烯自由旋转链 2、受阻旋转链 6）。

    有限 n 下的特征比要小一点（差 O(1/n)），拿 model_cn() 取。
    """
    return model_kuhn_over_l(mp)


def model_cn(mp: ModelParams, n: int, l: float) -> float:
    """特征比 ⟨h²⟩/(n·l²)。前三个模型与 n 无关，WLC 依赖链长。"""
    return model_h2(mp, n, l) / (n * l * l)


def model_kuhn_length(mp: ModelParams, l: float) -> float:
    """等价 Kuhn 长度 b（长度单位跟随 l）。"""
    return model_kuhn_over_l(mp) * l


def model_formula(mp: ModelParams) -> str:
    """给界面/助手用的一行公式（纯文本）。"""
    if mp.key == MODEL_FJC:
        return "⟨h²⟩ = n·l²　Cn ≡ 1"
    if mp.key == MODEL_FRC:
        return "⟨h²⟩ = n·l²·(1−cosθ)/(1+cosθ)"
    if mp.key == MODEL_HINDERED:
        return "⟨h²⟩ = n·l²·(1−cosθ)/(1+cosθ)·(1+⟨cosφ⟩)/(1−⟨cosφ⟩)"
    return "⟨h²⟩ = 2pL[1−(p/L)(1−e^(−L/p))]，L = n·l，b = 2p"


def model_note(mp: ModelParams) -> str:
    """给界面用的一句「这个模型在说什么」（含它近似的部分）。"""
    if mp.key == MODEL_FJC:
        return "链段自由连接：无键角限制、无位阻，链段取向互相独立。⟨h²⟩ = n·l² 是精确解。"
    if mp.key == MODEL_FRC:
        return (
            f"键角固定为 {mp.theta_deg:g}°，绕键自由旋转、无位阻。"
            f"方向关联按 cosθ 逐段衰减，求和得 Cn = (1−cosθ)/(1+cosθ)；θ=90° 时退回 FJC。"
        )
    if mp.key == MODEL_HINDERED:
        # 「受阻」两个字的量化说法：三态里 gauche 相对 trans 的能量差。读数就是
        # 能量图景那张卡画的东西 —— 两个入口共用 hindered_barrier() 这一份换算。
        sigma, delta_e, _ = hindered_barrier(mp.cos_phi)
        ratio = "∞（gauche 完全占优）" if not math.isfinite(sigma) else f"{sigma:.3g}"
        barrier = "−∞" if not math.isfinite(delta_e) else f"{delta_e:.2f} kT"
        return (
            f"键角固定为 {mp.theta_deg:g}°，再叠加内旋转势垒（平均余弦 ⟨cosφ⟩ = {mp.cos_phi:g}）。"
            f"三态模型里 trans 在 φ=0°、gauche± 在 φ=±120°；gauche 与 trans 的布居比 "
            f"σ = {ratio}，对应能量差 ΔE = {barrier}（σ = e^(−ΔE/kT)）——ΔE = 0 时三态等权重，"
            f"正好退回自由旋转链。"
            f"经典近似下就是几何因子再乘 (1+⟨cosφ⟩)/(1−⟨cosφ⟩) —— 聚乙烯 θ=109.47°、⟨cosφ⟩≈0.5 "
            f"时 Cn ≈ 6，正好对上实测的 C∞ ≈ 6.7。"
        )
    return (
        f"连续杆状链（Kratky–Porod），持久长度 p = {mp.p:g}l、Kuhn 长度 b = 2p = {2 * mp.p:g}l。"
        f"短链端退化成刚杆（⟨h²⟩ → L²），长链端回到 ⟨h²⟩ ≈ 2pL —— 所以它的特征是"
        f"「先 L²、后 L¹」的过渡，不像前三个模型那样一条斜率 1 的直线到底。"
    )


def _langevin(k: float) -> float:
    """Langevin 函数 L(k) = coth k − 1/k。就是力–伸长那张图用的同一个函数。

    单独包一层是为了让 WLC 采样器读起来自洽：κ 是弯曲刚度，L(κ) 是相邻键矢量的
    平均余弦 —— 同一个 L，两处用。
    """
    return langevin(k)


def _wlc_kappa(corr: float) -> float:
    """解 L(κ) = corr，返回离散蠕虫状链的弯曲刚度 κ。

    L 单调递增（0→1），所以二分一定收敛，而且不需要初值猜测。目标相关取
    corr = e^(−l/p)：这样链的 ⟨u₀·u_s⟩ 随弧长指数衰减、衰减长度正好是 p，
    也就是持久长度的定义本身。
    """
    lo, hi = 1e-9, 400.0
    for _ in range(_KAPPA_ITER):
        mid = 0.5 * (lo + hi)
        if _langevin(mid) < corr:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _torsion_steps(rng, shape, cos_phi: float) -> np.ndarray:
    """受阻旋转链的内旋转角（弧度），取值 {0, ±120°} 的三态分布。

    trans(φ=0) 与 gauche±(φ=±120°) 按概率配：
        ⟨cosφ⟩ = p·1 + (1−p)·(−1/2)  ⇒  p = (2⟨cosφ⟩ + 1)/3
    ⟨cosφ⟩ = 0.5 对应 p = 2/3 —— 正是聚乙烯那个经典的 trans 占比。

    说清它是什么：经典近似里 ⟨h²⟩ 只依赖 ⟨cosφ⟩，**任何**同均值的独立分布都给出
    同一条 ⟨h²⟩，但构象细节不同。三态是其中最经典、也最容易画出来的那一种。
    """
    p_trans = (2.0 * cos_phi + 1.0) / 3.0
    u = rng.random(shape)
    gauche = (1.0 - p_trans) / 2.0
    return np.where(
        u < p_trans, 0.0,
        np.where(u < p_trans + gauche, 2.0 * math.pi / 3.0, -2.0 * math.pi / 3.0),
    )


def _perp_basis(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """给一批单位矢量 u（…, 3）各配一组与它正交的单位基。

    参考轴避开与 u 平行的方向（u 靠近 z 轴就改用 x 轴），否则叉乘会退化成零矢量。
    """
    ref = np.where((np.abs(u[..., 2]) > 0.9)[..., None],
                   np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    e1 = np.cross(u, ref)
    e1 = e1 / np.linalg.norm(e1, axis=-1, keepdims=True)
    return e1, np.cross(u, e1)


def _step_vectors_model(mp: ModelParams, rng, chains: int, n: int) -> np.ndarray:
    """按模型生成 (chains, n, 3) 的步进矢量，每段长度为 1。

    随机流的用法对四个模型是同一套：先抽第一步（球面均匀），再逐步抽转角。
    random_chain 和 _sweep_endpoints 共用这一个函数 —— 测试钉着「同一条随机流上
    两者逐位相同」，所以随机数消耗的**顺序和形状**不能按模型改。
    """
    if mp.key == MODEL_FJC:
        # 自由连接链：每一步的方向都独立、球面均匀 —— 这是它快的原因（整批一次抽完）
        return _step_vectors(rng, (chains, n))

    if mp.key == MODEL_WLC:
        # 离散蠕虫状链：cos(键矢量夹角) 服从 p(x) ∝ e^(κx)（x ∈ [−1,1]）。
        # 逆变换抽样一行出结果：u = (e^(κ(x+1)) − 1)/(e^(2κ) − 1)。
        kappa = _wlc_kappa(math.exp(-1.0 / mp.p))
        u = rng.random((chains, n))
        cos_step = -1.0 + np.log1p(u * math.expm1(2.0 * kappa)) / kappa
    else:
        # 自由旋转链 / 受阻旋转链：相邻**键矢量**的夹角固定 = 180° − 键角。
        # 第一步由球面均匀给，占位写 1 不影响（下面从 i=1 才开始用）。
        cos_step = np.full((chains, n), -math.cos(math.radians(mp.theta_deg)))
        cos_step[:, 0] = 1.0
    sin_step = np.sqrt(np.maximum(0.0, 1.0 - cos_step * cos_step))

    if mp.key == MODEL_HINDERED:
        phi = _torsion_steps(rng, (chains, n), mp.cos_phi)
        # 第一步还没有参考平面（二面角在只有一条链段时没有定义），方位角取均匀分布。
        # 这只影响「第一条链段朝哪边」，不影响任何统计量 —— 但少了它，
        # u₁ 相对 u₀ 的方位角会只剩三个离散值，链在起点的各向同性就没了。
        phi[:, 1] = rng.uniform(0.0, 2.0 * math.pi, size=chains)
    else:
        # 绕键自由旋转 = 方位角均匀。WLC 没有内旋转势，也走这里。
        phi = rng.uniform(0.0, 2.0 * math.pi, size=(chains, n))

    out = np.empty((chains, n, 3), dtype=float)
    out[:, 0] = _step_vectors(rng, (chains,))
    # 参考平面怎么选：
    #   * **受阻旋转链必须用「前两段张成的平面」** —— 这样 φ 才是真正的二面角，
    #     三态分布才有化学含义。它的键矢量夹角是固定的（聚乙烯 70.53°），
    #     投影不会退化。
    #   * 自由旋转链与蠕虫状链的方位角是均匀的，用哪组正交基都一样，
    #     所以直接用 _perp_basis 取一组：不经过「减投影再归一化」，
    #     链很长、转角很小时就不会累积浮点误差。WLC 的键角可以贴近 0，
    #     投影法在那一带退化成噪声，再沿链放大 —— 实测 n=400 时步长已经跑到 2.1。
    use_plane = mp.key == MODEL_HINDERED
    for i in range(1, n):
        u_prev = out[:, i - 1]
        if use_plane and i >= 2:
            u_ref = out[:, i - 2]
            e1 = u_ref - (u_ref * u_prev).sum(axis=1, keepdims=True) * u_prev
            norm = np.linalg.norm(e1, axis=1, keepdims=True)
            # 链几乎拉直时这个分量趋于 0（那时 φ 本来也没有意义），退回任意正交基
            bad = (norm < 1e-9)[:, 0]
            if bad.any():
                fallback, _ = _perp_basis(u_prev)
                e1 = np.where(bad[:, None], fallback, e1)
                norm = np.linalg.norm(e1, axis=1, keepdims=True)
            e1 = e1 / norm
        else:
            e1, _ = _perp_basis(u_prev)
        e2 = np.cross(u_prev, e1)
        c = cos_step[:, i][:, None]
        s = sin_step[:, i][:, None]
        ph = phi[:, i][:, None]
        out[:, i] = c * u_prev + s * (np.cos(ph) * e1 + np.sin(ph) * e2)
    return out


# --- 能量图景 -----------------------------------------------------------------
#
# 四个解析模型里**只有两个有真正的能量自由度**：
#
#   * 受阻旋转链 —— 内旋转势垒。trans(φ=0) 与 gauche±(φ=±120°) 的能量不同，
#     这个差就是 `_torsion_steps()` 那行 p_trans = (2⟨cosφ⟩+1)/3 的全部来源。
#   * 蠕虫状链   —— 弯曲刚度 κ。相邻键对齐与否能量不同，`_step_vectors_model()`
#     的 WLC 分支抽的 p(x) ∝ e^(κx) 正是势 U/kT = κ(1−cosθ) 的玻尔兹曼分布。
#
# 另外两个没有：自由连接链除固定键长外没有任何约束（势恒为 0），自由旋转链的
# 键角是**硬约束**不是势（绕键自由旋转）。给它们画能量图只能画出平线或 δ 峰，
# 零信息，所以 energy_figure() 对它们返回 kind="none" 与一句中文理由 ——
# 这条分支不是死代码，它是「为什么这两个模型没有能量图」在后端的**唯一**落点：
# 测试钉它、README 引它，前端不必为它写 if。
#
# 两条式子都是**精确**的，不是拟合：
#
#   ΔE/kT = ln((1 + 2c)/(1 − c))        c ≡ ⟨cosφ⟩，gauche 相对 trans 的能量差
#   κ 由 L(κ) = e^(−l/p) 定出，大 p/l 端 κ ≈ p/l + 1/2
#
# 受阻旋转链那张图上画的 U(φ) 是一条**曲线**，而采样器抽的是三个**离散**值 ——
# 这两件事别混：曲线画的是**势**（井底落在 0°/±120° 三个态上，井深差就是 ΔE），
# 三根柱画的是**布居**。这条曲线的**井底位置与井深**由模型定死，**势垒高度**则
# 纯属画法约定（TORSION_BARRIER_RATIO）—— 模型只给了 ΔE 一个数，画一条曲线还需要
# 形状。而且对这条连续曲线积出来的玻尔兹曼权重**不等于**三态权重（三态是离散近似）。
#
# 两条都有一处**接缝**，是这套图最值得画的地方：c = 0 ⇒ ΔE = 0 ⇒ 受阻旋转链与
# 自由旋转链精确重合（U(φ) 在那里退化成一条平线）；κ → 0 ⇒ 键角分布变均匀 ⇒
# 蠕虫状链退回自由连接链。

# Cn–位垒那张图的横轴下限。ΔE → −∞（cos_phi → COS_PHI_MIN，全部 gauche）画不出来，
# 截断在 −3：那里 Cn 与它的下界 Cn_FRC/3 只差 0.3%（(1+2e⁻³)/3 = 0.3665 对 1/3）。
ENERGY_CN_DELTA_MIN = -3.0
ENERGY_CN_POINTS = 240

# 内旋转势 U(φ) 那条曲线的取样点数：−180°…180°，1.5° 一格。
# 0°/±60°/±120°/±180° 全都**正好落在格点上**（360/240 × 整数），所以
# 井底与几个极值点不需要插值 —— 图上那三个点与曲线的极小点是同一个数。
ENERGY_TORSION_POINTS = 241

# **约定**，不是物理：两个 gauche 井之间的势垒取成 |ΔE| 的多少倍。
# 模型本身只定死一个数 —— 井深差 ΔE（由 ⟨cosφ⟩ 精确换算）—— 而一条曲线还需要
# **形状**，也就是势垒得多高；这不是模型的预言，必须另外约定一个。
# 基准取 gauche↔gauche 那一处，是因为它连接的是**两个等价的井**，「比更高的那个
# 井高多少」在那里没有歧义；trans↔gauche 那处连接的两个井不等高（相差 ΔE），
# 拿它做基准还得再约定一次以谁为准。
# 取 4 的一个旁证：聚乙烯（ΔE = ln 4 = 1.386 kT）由此得到 5.545 kT = 13.7 kJ/mol，
# 与聚乙烯真实的内旋转势垒（约 3 kcal/mol ≈ 12.6 kJ/mol）同量级。
TORSION_BARRIER_RATIO = 4.0

# 摩尔气体常数，J/(mol·K)。2019 年 SI 修订之后 k_B 与 N_A 都是定义精确值，
# 所以 R = k_B·N_A 也是精确的 —— 与力–伸长那节的 K_B 同源，不是测量值。
# 只用在曲线纵轴的 kJ/mol 换算说明上。
R_GAS = 8.31446261815324

# 持久长度–刚度那张图的 κ 扫描范围。上限取 _wlc_kappa 二分上界的一半以上，
# 覆盖到 p = P_MAX = 300（那时 κ ≈ 300.5）；下限 0.01 对应 p/l ≈ 0.175。
ENERGY_KAPPA_MIN = 0.01
ENERGY_KAPPA_MAX = 400.0
ENERGY_KAPPA_POINTS = 240


def hindered_barrier(cos_phi: float) -> tuple[float, float, float]:
    """受阻旋转链的三态位垒：(σ, ΔE/kT, p_trans)。

    σ 是**单个** gauche 态相对 trans 态的玻尔兹曼因子。gauche 有两个（±120°），
    把 ⟨cosφ⟩ 按三态展开就能解出它：

        ⟨cosφ⟩ = p_trans·1 + (1 − p_trans)·(−1/2)
        p_trans = 1/(1 + 2σ)   ⇒   σ = (1 − ⟨cosφ⟩)/(1 + 2⟨cosφ⟩)

    ΔE = −kT·ln σ 就是 gauche 相对 trans 的能量差 —— 位垒滑块上那个 (kT) 数。
    这里算出的 p_trans 与 `_torsion_steps()` 用的 (2c+1)/3 是**同一个数**
    （代换上恒等，测试逐点钉着）—— 所以图上画的就是采样器真正在用的分布。

    cos_phi = COS_PHI_MIN = −0.5 时 σ → ∞、ΔE → −∞（全部链段走 gauche）。
    这里如实返回 inf，**由调用方**格式化成「−∞（全 gauche）」，
    绝不把 inf 塞进 JSON（`json.dumps` 会写成 `-Infinity`，JS 的 JSON.parse 直接报错）。
    """
    c = float(cos_phi)
    denom = 1.0 + 2.0 * c
    if denom <= 0.0:
        return math.inf, -math.inf, 0.0
    sigma = (1.0 - c) / denom
    p_trans = 1.0 / (1.0 + 2.0 * sigma)
    delta_e = -math.log(sigma) if sigma > 0.0 else math.inf
    if delta_e == 0.0:
        delta_e = 0.0                     # c = 0 时 -log(1.0) 是 -0.0，收拾干净
    return sigma, delta_e, p_trans


def hindered_potential(delta_e_over_kt: float) -> tuple[list[float], list[float]]:
    """内旋转势那条曲线：(φ/°, U/kT)，各 ENERGY_TORSION_POINTS 个点。

    取的是**已截断过**的 ΔE（`all_gauche` 时是 ENERGY_CN_DELTA_MIN）—— 截断判断
    在 `energy_figure()` 里只做一次，这里不重复。

    三态模型只说了「φ 取 0°（trans）与 ±120°（gauche±）三个值，井深差 ΔE」。
    要把这条能量画成一条曲线，就得造一个**以这三个值为井底**的三井势：

        U(φ) = α·(cos φ + ½·cos 2φ) + a₃·cos 3φ − (3α/2 + a₃)，   α = −4ΔE/9

    α 是**解出来的、不是选的**：要求极值正好落在 0° 与 ±120° 上，
    求导 dU/dφ = −sinφ·(a₁ + 2a₂cosφ) − 3a₃·sin3φ 在 120° 那一点要求
    a₁ + 2a₂·cos120° = 0，即 a₁ = 2a₂（0° 在 a₁ = 2a₂ 下自动满足）；
    再用 U(±120°) − U(0°) = ΔE 定标，就得到 α = a₁ = 2a₂ = −4ΔE/9。
    常数项把 U(0°) 钉在 0 —— trans 是能量零点。

    **这条曲线不是 120° 周期的**（别被「三井」骗了）：井底间距确实是 120°、三口井
    一样深，但两条势垒不一样高（见下），而 120° 周期会强制它们一样高。整条曲线是
    **360° 周期**的：三个井在 0°/±120°，极大点在 ±60° 附近与 ±180°。

    a₃ 是**唯一自由的形状参数**（势垒高度），按 TORSION_BARRIER_RATIO 那条约定取：

        a₃ = −ΔE/18 − 2|ΔE|   ⇒   两个 gauche 井之间的势垒 = TORSION_BARRIER_RATIO·|ΔE|

    里面那个绝对值是必需的：⟨cosφ⟩ < 0 时 gauche 比 trans **低**，井深差是负的，
    而势垒高度不能跟着变负 —— 写成 −20ΔE/9 的话，ΔE < 0 时 0°/±120° 会从极小点
    翻成极大点，**井底就跑掉了**。取 |ΔE| 之后两边都对：三个井仍都在 0°/±120°，
    只是 ΔE < 0 时 0° 那个井被抬到两个 gauche 井**上面** —— 正是「gauche 更稳」的样子。
    **但不是把曲线整体翻过来**：极小点位置一个没动，变的是三个井的相对高度。

    一个结构性的推论（不是选择）：**trans↔gauche 那条势垒总比 gauche↔gauche 那条低**，
    ΔE > 0 时约低 |ΔE|/3（实测 ≈ 3.68|ΔE| 对 4|ΔE|），ΔE < 0 时约低 2|ΔE|/3
    （≈ 3.35|ΔE|）。两者的差由 α 唯一决定，换任何 a₃ 都消不掉。
    **只有 gauche↔gauche 那条有精确值 4|ΔE|** —— 它是构造出来的；trans↔gauche 那条
    的顶点由 −α(sinφ + sin2φ) = 3a₃·sin3φ 定，是个超越方程，没有闭式，
    所以那个数由 `energy_figure()` **从画出来的曲线上直接量**，不在这里假造一个公式。
    顺带：两条势垒的顶点都落在 ±60° 附近但**不正好在** ±60°（实测偏到 57°~63°）——
    被 a₁ = 2a₂ 钉死的是 0°/±120° 那三个**井底**，顶点位置是自由的。

    两个接缝：
      ΔE = 0 ⇒ α = a₃ = 0 ⇒ **U ≡ 0**，一条平线 —— 这正是「绕键自由旋转」的诚实
        画法，也正是 `model_h2()` 里受阻旋转链精确退回自由旋转链的那一格。
      ΔE < 0 ⇒ 0° 那个井被抬到两个 gauche 井上面（见上）。

    **这条曲线的玻尔兹曼分布不等于三态权重**：三态是**离散**近似，对连续 U(φ)
    积分出来的权重与 p_trans = (2c+1)/3 不是同一个数。这条曲线说的是「ΔE 是从哪
    来的」，不是「采样器在按它抽」—— 画布上那张分布图照旧只画三根柱。
    """
    d_e = float(delta_e_over_kt)
    alpha = -4.0 * d_e / 9.0
    a3 = -d_e / 18.0 - 2.0 * abs(d_e)
    const = -(1.5 * alpha + a3)
    phis = [-180.0 + 360.0 * i / (ENERGY_TORSION_POINTS - 1)
            for i in range(ENERGY_TORSION_POINTS)]
    us = []
    for p in phis:
        t = math.radians(p)
        us.append(alpha * (math.cos(t) + 0.5 * math.cos(2.0 * t))
                  + a3 * math.cos(3.0 * t) + const)
    # ΔE = 0 时三项正好抵消成 0，但可能出现 −0.0，JSON 里会写成 "-0.0"
    # （hindered_barrier 里对 delta_e 做过同一件事）。加一个 0.0 掰回 +0.0，
    # IEEE 下 −0.0 + 0.0 = +0.0，非零值不受影响。
    return phis, [u + 0.0 for u in us]


def wlc_bending(p: float) -> tuple[float, float, float]:
    """蠕虫状链的弯曲刚度：(κ, p/l, ⟨cosθ⟩)。

    κ 解自 L(κ) = e^(−l/p)，用的就是采样器那个 `_wlc_kappa()`。所以
    ⟨cosθ⟩ = L(κ) = e^(−l/p) 是**构造出来的恒等式**而不是近似 ——
    页面上模型指纹卡的 corr[1] 是这个同一个数，两处能对上。

    能量读法：采样器抽的 p(x) ∝ e^(κx) 就是势 U(cosθ)/kT = κ(1 − cosθ) 的
    玻尔兹曼分布，值域 0（相邻键完全对齐）… 2κ（完全反向）。
    """
    corr = math.exp(-1.0 / float(p))
    return _wlc_kappa(corr), float(p), corr


def _wlc_density(kappa: float, x: np.ndarray) -> np.ndarray:
    """p(x) ∝ e^(κx) 在 [−1,1] 上的归一化密度：p = κ·e^(κx) / (2 sinh κ)。

    走 log 空间算：κ 到 300 时 e^(κx) 与 sinh κ 都贴着 double 的上限
    （e^300 ≈ 2×10¹³⁰），而 log 形式只要 κx − log(2 sinh κ / κ)，全程安全；
    小 κ 端 2 sinh κ / κ → 2，log 也不退化。x = ±1 两端正好是 0 与 κ 的密度。
    """
    log_z = math.log(2.0) if kappa < 1e-9 else math.log(2.0 * math.sinh(kappa) / kappa)
    return np.exp(kappa * x - log_z)


def energy_figure(mp: ModelParams) -> dict:
    """「能量图景」那张卡片要的全部数据。**唯一一份**，/api/energy 原样吐出去。

    前端一个公式都不写 —— 和 model_fingerprint() / force_extension() 同一条规矩。
    本函数只算「这个模型的能量长什么样」，不碰 ⟨h²⟩ 的公式层。
    """
    if mp.key == MODEL_HINDERED:
        sigma, delta_e, p_trans = hindered_barrier(mp.cos_phi)
        # 指数一过 709 就溢出，这里也顺手把 ΔE 封顶到 COS_PHI_MAX 对应的那个值
        delta_max = math.log((1.0 + 2.0 * COS_PHI_MAX) / (1.0 - COS_PHI_MAX))

        # Cn 随位垒：逐点回 model_kuhn_over_l() 现算，不在这里抄第二遍位垒公式。
        # σ = e^(−ΔE) ⇒ ⟨cosφ⟩ = (1−σ)/(1+2σ) 是 hindered_barrier 的逆变换。
        ds = np.linspace(ENERGY_CN_DELTA_MIN, delta_max, ENERGY_CN_POINTS)
        cn_curve = [
            model_kuhn_over_l(ModelParams.of(
                MODEL_HINDERED, theta_deg=mp.theta_deg,
                # 端点会因浮点落在 COS_PHI_MIN/MAX 外一点点，夹回去；
                # 夹的是网格端点而不是物理，误差在 1e-15 量级
                cos_phi=min(max((1.0 - math.exp(-d)) / (1.0 + 2.0 * math.exp(-d)),
                                COS_PHI_MIN), COS_PHI_MAX),
            ))
            for d in ds
        ]
        cn_frc = model_kuhn_over_l(ModelParams.of(MODEL_FRC, theta_deg=mp.theta_deg))
        p_gauche = (1.0 - p_trans) / 2.0
        # σ = ∞ 正是「全部走 gauche」那一个端点。写成 `not isfinite(delta_e)` 也能跑，
        # 但 ΔE 在另一端（σ = 0，全 trans）同样不是有限数 —— 那一点被 COS_PHI_MAX = 0.99
        # 挡在参数范围外，而按 σ 判就不会在这两件事上留下混同的余地。
        all_gauche = sigma == math.inf
        u_gauche = ENERGY_CN_DELTA_MIN if all_gauche else delta_e

        # 那条 U(φ) 曲线。传进去的是**已经截断过**的 u_gauche，不重复上面那次判断。
        phi_deg, u_potential = hindered_potential(u_gauche)
        # 两条势垒**从画出来的曲线上直接量**，读数与图必然一致：
        #   gauche↔gauche：120°…180° 段的最高点 − U(120°)
        #   trans↔gauche：0°…120° 段的最高点 − 两者中**较高**的那个井
        #     （ΔE > 0 时较高的是 gauche，ΔE < 0 时是 trans）
        # 两个都恒有限：u_gauche 是截断值，all_gauche 时是 −3 而不是 −∞。
        u_trans = u_potential[phi_deg.index(0.0)]
        gg_barrier = max(u for p, u in zip(phi_deg, u_potential)
                         if p >= 120.0) - u_gauche
        tg_barrier = max(u for p, u in zip(phi_deg, u_potential)
                         if 0.0 <= p <= 120.0) - max(u_trans, u_gauche)

        return {
            "model": mp.key,
            "kind": "torsion",
            "theta_deg": mp.theta_deg,
            "cos_phi": mp.cos_phi,
            "barrier": {
                # None 表示 −∞（全 gauche）：JSON 里不能出现 Infinity
                "sigma": None if not math.isfinite(sigma) else sigma,
                "delta_e_over_kt": None if not math.isfinite(delta_e) else delta_e,
                "p_trans": p_trans,
                "all_gauche": all_gauche,
                # 势垒高度是**约定**（TORSION_BARRIER_RATIO），不是模型的预言 ——
                # 模型只定死井深差 ΔE。all_gauche 时这两个数建在截断值 −3 上，
                # 读数行要注明，别让 12 kT 看着像真值。
                "gg_over_kt": gg_barrier,
                "tg_over_kt": tg_barrier,
            },
            # 三井势 U(φ)/kT（**360° 周期、不是 120°**，见 hindered_potential()），
            # 井底正好落在下面 levels 那三个 φ 上。
            # 这是**势**（模型的输入），不是布居 —— 右栏那三根柱才是布居（模型的输出）。
            # 对这条曲线做玻尔兹曼积分**不等于**那三根柱，见 hindered_potential()。
            "phi_deg": phi_deg,
            "potential": u_potential,
            "u_max": max(u_potential),
            "u_min": min(u_potential),
            # 纵轴的 kJ/mol 换算（R·T，T 用 DEFAULT_TEMPERATURE）。只用来说明刻度 ——
            # **不做第二根纵轴**，换算写进轴标题的文字里。
            "kt_in_kj_per_mol": R_GAS * DEFAULT_TEMPERATURE / 1000.0,
            # 三态，**离散**的三个值 —— 采样器抽的就是这三个。上面那条 U(φ) 曲线画的
            # 是**势**（井底落在这三个 φ 上），**不是** φ 的分布；两者别混。
            "levels": [
                {"phi_deg": 0.0, "cos_phi": 1.0, "name": "trans",
                 "u_over_kt": 0.0, "weight": p_trans},
                {"phi_deg": 120.0, "cos_phi": -0.5, "name": "gauche+",
                 "u_over_kt": u_gauche, "weight": p_gauche},
                {"phi_deg": -120.0, "cos_phi": -0.5, "name": "gauche−",
                 "u_over_kt": u_gauche, "weight": p_gauche},
            ],
            "cn": {
                "now": model_kuhn_over_l(mp),
                "frc": cn_frc,
                "delta_e": ds.tolist(),
                "values": cn_curve,
                # 聚乙烯那个点：θ = 109.47°、⟨cosφ⟩ = 0.5
                "pe_delta_e": math.log(4.0),
                "pe_cn": model_kuhn_over_l(ModelParams.of(
                    MODEL_HINDERED, theta_deg=DEFAULT_THETA_DEG, cos_phi=0.5)),
            },
            "delta_max": delta_max,
        }

    if mp.key == MODEL_WLC:
        kappa, p_over_l, corr = wlc_bending(mp.p)
        xs = np.linspace(-1.0, 1.0, CURVE_POINTS)
        # 持久长度随弯曲刚度：改用 log 均匀的 κ 网格，小 κ 那一端的弯折才看得见
        ks = np.logspace(math.log10(ENERGY_KAPPA_MIN),
                         math.log10(ENERGY_KAPPA_MAX), ENERGY_KAPPA_POINTS)
        # p/l = −1/ln L(κ)：就是 wlc_bending 的反函数，大 κ 端趋于 κ − 1/2
        ks_list = ks.tolist()
        pl = [-1.0 / math.log(_langevin(k)) for k in ks_list]
        return {
            "model": mp.key,
            "kind": "bending",
            "p": mp.p,
            "kappa": kappa,
            "p_over_l": p_over_l,
            "corr": corr,
            "cos": xs.tolist(),
            # U/kT = κ(1 − cosθ)：0 是相邻键对齐，2κ 是完全反向
            "potential": (kappa * (1.0 - xs)).tolist(),
            "density": _wlc_density(kappa, xs).tolist(),
            "u_max": 2.0 * kappa,
            "kappa_scan": ks_list,
            "pl_scan": pl,
            # 渐近 κ ≈ p/l + 1/2（由 l/p = −ln L(κ) ≈ 1/κ + 1/(2κ²) 得来）。
            # 只给 κ ≥ 1 那一段：再往左它就给不出正数了，而那正是渐近本身失效的地方
            # —— 画在**对数**纵轴上会变成 NaN，不如后端就不发。
            "pl_asymptote": [[k, k - 0.5] for k in ks_list if k >= 1.0],
        }

    # fjc / frc：没有能量自由度。理由写在后端一处，前端不必为它写 if
    reason = (
        "自由连接链除了固定键长之外没有任何约束：链段取向各向同性、彼此独立，"
        "等价于势能恒为 0。没有能量自由度，画不出能量图。"
        if mp.key == MODEL_FJC else
        "自由旋转链的键角是**硬约束**不是势：相邻键矢量的夹角被钉死，绕键的旋转完全自由。"
        "约束不贡献能量自由度（它只在 ⟨h²⟩ 里给出 (1−cosθ)/(1+cosθ) 那个几何因子），"
        "所以画不出能量图。"
    )
    return {"model": mp.key, "kind": "none", "reason": reason}


def validate(n, l) -> tuple[int, float]:
    """校验并归一化输入。不合法时抛 FJCInputError（中文信息，直接给用户看）。"""
    n_i = _as_int(n, "链段数 n")
    if n_i < 1:
        raise FJCInputError("链段数 n 至少为 1")
    if n_i > N_MAX:
        raise FJCInputError(f"链段数 n 不能超过 {N_MAX:,}")

    try:
        l_f = float(l)
    except (TypeError, ValueError):
        raise FJCInputError("链段长度 l 必须是数字") from None
    if not math.isfinite(l_f):
        raise FJCInputError("链段长度 l 必须是有限数")
    if l_f <= 0:
        raise FJCInputError("链段长度 l 必须大于 0")

    return n_i, l_f


def validate_chain(n, l, seed=None, chains=1,
                   max_chains: int | None = MAX_CHAINS,
                   max_n: int | None = CHAIN_N_MAX,
                   points_max: int | None = CHAIN_POINTS_MAX) -> tuple[int, float, int, int]:
    """单链构象模拟的输入校验。比 validate() 多一层**绘制预算**的约束。

    三个上限都是「画布和 JSON 不被撑爆」这类的表现层约束，不是物理约束，
    所以每个都能单独传 None 关掉 —— 统计验证需要几千条链才能让 ⟨R²⟩ 收敛，
    超出绘制预算是故意的。写法与 compute(..., points=CURVE_POINTS) 一致。
    """
    n_i, l_f = validate(n, l)
    if max_n is not None and n_i > max_n:
        raise FJCInputError(
            f"绘制用的链段数不能超过 {max_n:,}（再多画出来就是一团结）"
        )

    chains_i = _as_int(chains, "链数")
    if chains_i < 1:
        raise FJCInputError("链数至少为 1")
    if max_chains is not None and chains_i > max_chains:
        raise FJCInputError(f"最多同时画 {max_chains} 条链，收到 {chains_i}")

    total = chains_i * (n_i + 1)
    if points_max is not None and total > points_max:
        raise FJCInputError(
            f"链段数 × 链数 超出绘制预算（{total:,} > {points_max:,}），"
            f"请降低 n 或链数"
        )

    seed_i = _as_seed(seed)

    return n_i, l_f, seed_i, chains_i


def beta_from_h2(h2: float) -> float:
    """等效高斯链的宽度参数 β = √(3/(2⟨h²⟩))。

    **四个模型共用这一条**：分布形状一律是等效高斯，模型只改 ⟨h²⟩。
    """
    return math.sqrt(3.0 / (2.0 * h2))


def beta_of(n: int, l: float) -> float:
    """FJC 的宽度参数 β = √(3/(2nl²))（= beta_from_h2(n·l²)）。"""
    return beta_from_h2(n * l * l)


def P_of_h(h, beta: float):
    """三维空间中的末端距径向分布 P(h) = (β/√π)³·4πh²·e^(−β²h²)。

    ∫₀^∞ P(h)dh = 1，∫₀^∞ h²P(h)dh = 3/(2β²) = n·l²。
    """
    h = np.asarray(h, dtype=float)
    return (beta / math.sqrt(math.pi)) ** 3 * 4.0 * math.pi * h**2 * np.exp(-((beta * h) ** 2))


def compute(n, l=1.0, points: int = CURVE_POINTS, params=None) -> FJCResult:
    """算出一个 n 下的全部输出量（按 params 指定的链模型）。

    n       链段数（≥1 的整数）
    l       链段长度（>0），默认 1，此时结果以链段长度为单位
    params  链模型与参数（ModelParams / dict / 模型名），默认自由连接链 FJC

    只有 ⟨h²⟩ 本身由模型给出，其余量都从它推出来：h*、⟨h⟩、σ 与 P(h) 用的是
    **等效高斯链**（换掉 ⟨h²⟩、分布形状不变）。前三个模型的 ⟨h²⟩ 是精确的，
    分布形状在等效 Kuhn 段数 L/b 足够大时才可靠 —— 不够时 warnings 里会说清。
    """
    mp = coerce_params(params)
    n, l = validate(n, l)

    h2 = model_h2(mp, n, l)
    b = model_kuhn_length(mp, l)
    n_kuhn = n * l / b            # 等价 Kuhn 段数 L/b：高斯近似能不能用就看它
    beta = beta_from_h2(h2)
    h_rms = math.sqrt(h2)
    h_mp = 1.0 / beta
    h_mean = 2.0 / (beta * math.sqrt(math.pi))
    h_max = n * l
    Rg2 = h2 / 6.0
    sigma = math.sqrt(h2 - h_mp * h_mp)

    # 采样区间：由分布宽度决定，但不能超过全伸展长度
    # （高斯分布在 h>n·l 的尾巴没有物理意义）
    h_hi = min(h_max, h_mp + 5.0 * sigma)
    curve_h = np.linspace(0.0, h_hi, points)
    curve_P = P_of_h(curve_h, beta)

    warnings: list[str] = []
    if n_kuhn < GAUSSIAN_N_MIN:
        if mp.key == MODEL_FJC:
            warnings.append(
                f"n={n} 太小，高斯近似不可靠：真实自由连接链在小 n 时偏离高斯分布，"
                f"且在 h→nl 处有硬截断。上方的 ⟨h²⟩=nl² 仍然精确，但分布曲线仅供参考。"
            )
        else:
            warnings.append(
                f"等效 Kuhn 段数只有 {n_kuhn:.3g} 段（L/b = {n * l:.6g}/{b:.6g}），"
                f"高斯近似不可靠：⟨h²⟩ 按{MODEL_LABELS[mp.key]}仍然精确，"
                f"但 P(h)、h*、⟨h⟩、σ 都只是长链估算，仅供参考。"
            )
    if h_rms > h_max:
        warnings.append(
            f"特征比 Cn={h2 / (n * l * l):.4g} 太大：h_rms={h_rms:.6g} 已经超过全伸展长度 "
            f"nl={h_max:.6g}，链接近刚杆，等效高斯近似在这一带没有意义"
            f"（⟨h²⟩ 本身仍按{MODEL_LABELS[mp.key]}给出）。"
        )
    if h_max < h_mp + 5.0 * sigma:
        warnings.append(
            f"全伸展长度 nl={h_max:.4g} 小于分布的自然宽度，曲线在右端被截断。"
        )

    return FJCResult(
        n=n,
        l=l,
        h2=h2,
        h_rms=h_rms,
        h_mp=h_mp,
        h_mean=h_mean,
        h_max=h_max,
        Rg2=Rg2,
        Rg_rms=math.sqrt(Rg2),
        beta=beta,
        sigma=sigma,
        Cn=h2 / (n * l * l),
        model=mp.key,
        kuhn_length=b,
        n_kuhn=n_kuhn,
        params=mp.to_dict(),
        curve_h=curve_h,
        curve_P=curve_P,
        warnings=warnings,
    )


def compute_many(ns, l=1.0, points: int = CURVE_POINTS, params=None) -> list[FJCResult]:
    """多个 n（用于曲线叠加对比）。逐个校验，任一不合法即整体报错。"""
    if not ns:
        raise FJCInputError("请至少输入一个链段数 n")
    return [compute(n, l, points, params) for n in ns]


def _step_vectors(rng, shape) -> np.ndarray:
    """在单位球面上**均匀**取 shape 个方向，返回 (…, 3) 的步进矢量。

    正确做法是 z ~ U(−1,1) + φ ~ U(0,2π)。
    注意**不是**「θ、φ 都均匀」—— 那样方向会在两极堆积，⟨u_z²⟩ 从 1/3 抬到 1/2，
    链会系统性变形。random_chain 与 sweep 共用这一个采样器，正是为了不让两处走偏。
    """
    z = rng.uniform(-1.0, 1.0, size=shape)
    phi = rng.uniform(0.0, 2.0 * math.pi, size=shape)
    rho = np.sqrt(1.0 - z * z)
    return np.stack([rho * np.cos(phi), rho * np.sin(phi), z], axis=-1)


def random_chain(n, l=1.0, seed=None, chains=1,
                 max_chains: int | None = MAX_CHAINS,
                 max_n: int | None = CHAIN_N_MAX,
                 points_max: int | None = CHAIN_POINTS_MAX,
                 params=None) -> ChainResult:
    """随机生成 `chains` 条单链构象（一次实现的几何），按 params 指定的模型。

    FJC 的定义是「每一步走一个长度为 l、方向在单位球面上**均匀分布**的矢量」，
    链段方向互相独立 —— 这正是 ⟨h²⟩ = n·l² 成为精确解的原因。另外三个模型只在
    「下一步往哪走」上加约束：自由旋转链固定相邻键矢量的夹角、受阻旋转链再给二面角
    一个三态分布、蠕虫状链按弯曲刚度抽夹角。**理论值一律来自 compute()**，
    采样器自己不另算一遍（那就会变成第二套公式）。

    返回的 R（本次实现的末端距）与 h_rms（理论 l√n）的差异是**正常涨落**，
    不是错误 —— 只有多条链的 ⟨R²⟩ 才收敛到该模型的 ⟨h²⟩。

    注意非 FJC 的三个模型**必须一步步递推**（下一步依赖上一步的方向），
    不像 FJC 能整批一次抽完 —— n 上万时这一步会有几百毫秒。

    三个上限参数见 validate_chain()：传 None 即解除对应的绘制预算。
    只要统计量、不要几何时用 sweep()：那个不需要顶点，能吃下上万条链。
    """
    mp = coerce_params(params)
    n, l, seed, chains = validate_chain(
        n, l, seed, chains, max_chains=max_chains, max_n=max_n, points_max=points_max
    )

    rng = np.random.default_rng(seed)
    u = _step_vectors_model(mp, rng, chains, n)

    # 累积求和即得各顶点；首项补一个原点，所以共 n+1 个顶点
    points = np.concatenate(
        [np.zeros((chains, 1, 3), dtype=float), np.cumsum(u, axis=1)], axis=1
    ) * l

    R = points[:, -1, :]
    R_mag = np.linalg.norm(R, axis=1)

    # 理论量复用解析内核，保证与分布图那一侧完全同源（不是第二套公式）
    base = compute(n, l, params=mp)

    warnings: list[str] = []
    if n > 10_000:
        warnings.append(
            f"n={n} 的链画出来非常密集，构象细节基本看不出来；"
            f"单链构象通常取 n ≤ 2000 更可读。"
        )

    # R 的涨落有多大？高斯链有 ⟨R⁴⟩ = (5/3)⟨R²⟩²，故 Var(R²) = (2/3)(n·l²)²，
    # 于是 k 条链的 ⟨R²⟩ 相对标准误 = √(2/3)/√k。
    # 页面上用它来判断「偏差 −20%」到底是离谱还是正常 ——
    # k=5 时固有涨落就有 ±36.5%，所以那种偏差完全是正常范围。
    # （n 很小时高斯结果只是近似，但这里只用来给个量级，够用。）
    rel_se = math.sqrt(2.0 / 3.0) / math.sqrt(chains)

    return ChainResult(
        n=n,
        l=l,
        seed=seed,
        chains=chains,
        points=points,
        R=R,
        R_mag=R_mag,
        R2_mean=float(np.mean(R_mag ** 2)),
        R2_theory=base.h2,
        R2_rel_se=rel_se,
        h_rms=base.h_rms,
        h_mp=base.h_mp,
        h_mean=base.h_mean,
        h_max=base.h_max,
        model=mp.key,
        params=mp.to_dict(),
        kuhn_length=base.kuhn_length,
        warnings=warnings,
    )


def _as_n_list(ns) -> list:
    """把扫描的 n 归一化成列表。

    允许数组、单个数字，或「用逗号或空格分隔」的字符串 —— 后者是为了让 API
    直接贴网页输入框里的内容也能用。
    """
    if ns is None:
        raise FJCInputError("请至少输入一个链段数 n")
    if isinstance(ns, (list, tuple)):
        items = list(ns)
    elif isinstance(ns, str):
        items = [t for t in re.split(r"[,，、;；\s]+", ns.strip()) if t]
    else:
        items = [ns]
    if not items:
        raise FJCInputError("请至少输入一个链段数 n")
    return items


def validate_sweep(ns, l, seed, chains,
                   max_points: int | None = SWEEP_N_MAX,
                   max_chains: int | None = SWEEP_CHAINS_MAX,
                   work_max: int | None = SWEEP_WORK_MAX) -> tuple[list[int], float, int, int]:
    """链长扫描的输入校验。

    和 validate_chain() 一样，三个上限都是**表现/资源约束**而非物理约束，
    每个都能单独传 None 关掉（统计验证会需要更大的样本）。
    """
    n_list = []
    l_f = 1.0
    for raw in _as_n_list(ns):
        n_i, l_f = validate(raw, l)
        n_list.append(n_i)

    if max_points is not None and len(n_list) > max_points:
        raise FJCInputError(f"一次最多扫 {max_points} 个 n，收到 {len(n_list)} 个")
    if len(set(n_list)) != len(n_list):
        raise FJCInputError("链段数 n 有重复，请去掉重复值")

    chains_i = _as_int(chains, "每个 n 的链数")
    if chains_i < 1:
        raise FJCInputError("每个 n 的链数至少为 1")
    if max_chains is not None and chains_i > max_chains:
        raise FJCInputError(f"每个 n 最多采 {max_chains:,} 条链，收到 {chains_i:,}")

    work = chains_i * sum(n_list)
    if work_max is not None and work > work_max:
        raise FJCInputError(
            f"计算量超出预算（链数 × Σn = {work:,} > {work_max:,}），"
            f"请降低链数或减少 n 的个数"
        )

    return n_list, l_f, _as_seed(seed), chains_i


def _sweep_endpoints(rng, n: int, chains: int, params=None) -> np.ndarray:
    """(chains, 3) 的末端矢量，单位步长（按 params 指定的模型采样）。

    与 random_chain 的区别是**只累加、不存顶点**：内存是 O(chains) 而与 n 无关，
    所以这里能采上万条链，random_chain 不能。分块只是为了控内存峰值，
    分块大小只由 n 决定，因此给定种子后结果完全可复现。

    两条路走**同一个** _step_vectors_model：同一条随机流上两者逐位相同（测试钉着）。
    """
    mp = coerce_params(params)
    chunk = max(1, min(chains, _WORK_CHUNK // n))
    out = np.empty((chains, 3), dtype=float)
    for i in range(0, chains, chunk):
        j = min(i + chunk, chains)
        out[i:j] = _step_vectors_model(mp, rng, j - i, n).sum(axis=1)
    return out


def sweep(ns, l=1.0, seed=None, chains=SWEEP_DEFAULT_CHAINS,
           max_points: int | None = SWEEP_N_MAX,
           max_chains: int | None = SWEEP_CHAINS_MAX,
           work_max: int | None = SWEEP_WORK_MAX,
           params=None) -> SweepResult:
    """扫一组 n，每个 n 采 `chains` 条独立链，得到 ⟨R²⟩ 对 n 的标度关系。

    这是把模型的 ⟨h²⟩(n) 从一条公式变成一条**可验证的标度律**：对 log⟨R²⟩ 与
    log n 做最小二乘拟合，再和**本模型的理论斜率**比 —— FJC/自由旋转/受阻旋转
    恒为 1，蠕虫状链则不是（短链端是 L²、长链端才过渡到 L¹），所以它的参考斜率
    由模型自己的 ⟨h²⟩(n) 曲线在同一组 n 上拟合出来，而不是硬写 1。

    每个 n 用 SeedSequence 派生出独立子种子，所以整条序列给定 seed 后可复现，
    且加密强度上各 n 之间不相关。
    """
    mp = coerce_params(params)
    ns, l, seed, chains = validate_sweep(
        ns, l, seed, chains,
        max_points=max_points, max_chains=max_chains, work_max=work_max,
    )

    children = np.random.SeedSequence(seed).spawn(len(ns))
    R2_mean, R2_theory, Cn_sim, rel_dev = [], [], [], []
    for n_i, child in zip(ns, children):
        R = _sweep_endpoints(np.random.default_rng(child), n_i, chains, mp)
        r2 = float(np.mean(np.sum(R * R, axis=1))) * l * l
        theory = model_h2(mp, n_i, l)        # 理论量复用模型层，不是第二套公式
        R2_mean.append(r2)
        R2_theory.append(theory)
        Cn_sim.append(r2 / theory)
        rel_dev.append((r2 - theory) / theory)

    # 相对标准误：高斯链 ⟨R⁴⟩ = (5/3)⟨R²⟩² ⇒ Var(R²) = (2/3)(n·l²)²，
    # 于是 M 条链的 ⟨R²⟩ 相对标准误 = √(2/3)/√M。M=10000 时约 0.82%。
    rel_se = math.sqrt(2.0 / 3.0) / math.sqrt(chains)

    slope = intercept = None
    fit_r2 = None
    slope_se = None
    slope_ref = None
    if len(ns) >= 2:
        ln_n = np.log(np.asarray(ns, dtype=float))
        ln_y = np.log(np.asarray(R2_mean, dtype=float))
        slope_f, intercept_f = np.polyfit(ln_n, ln_y, 1)
        slope, intercept = float(slope_f), float(intercept_f)
        ss_res = float(np.sum((ln_y - (slope * ln_n + intercept)) ** 2))
        ss_tot = float(np.sum((ln_y - ln_y.mean()) ** 2))
        fit_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        # 斜率自身的标准误**不等于**单个点的标准误：最小二乘下
        # SE(slope) = σ_ln y / √(Σ(ln nᵢ − mean(ln n))²)，而 σ_ln y ≈ rel_se
        # （ln y 的扰动就是 ⟨R²⟩ 的相对扰动）。n 排得越开，斜率越稳 ——
        # 拿 3·rel_se 当门槛会两头都错：10…3000 这种铺开的序列宽了十几倍（真出事也不报），
        # 300 和 400 这种挤在一起的又太紧（明明还在正常涨落里也报警）。
        sxx = float(np.sum((ln_n - ln_n.mean()) ** 2))
        if sxx > 0:
            slope_se = rel_se / math.sqrt(sxx)
        # 参考斜率：FJC / 自由旋转 / 受阻旋转恒为 1（⟨h²⟩ ∝ n），
        # 蠕虫状链得由它自己的曲线定 —— 拿 1 去卡它会天天误报。
        if mp.key in (MODEL_FJC, MODEL_FRC, MODEL_HINDERED):
            slope_ref = 1.0
        else:
            ln_t = np.log(np.asarray([model_h2(mp, n_i, l) for n_i in ns], dtype=float))
            slope_ref = float(np.polyfit(ln_n, ln_t, 1)[0])

    warnings: list[str] = []
    if len(ns) < 2:
        warnings.append("只给了一个 n，拟合不出标度斜率；至少要两个不同的 n 才能看 ⟨R²⟩ ∝ n。")
    if len(ns) == 2:
        warnings.append(
            "只有两个 n：任何两点都能连成一条直线，所以这里决定系数 R² 必然是 1，"
            "它说明不了什么；斜率的误差也比点多时大得多（见下面那条三倍标准误）。"
        )
    if chains < 1000:
        warnings.append(
            f"每个 n 只采了 {chains:,} 条链，⟨R²⟩ 的相对标准误约 {rel_se * 100:.1f}%，"
            f"点时会在理论线上下明显跳动。样本调大（如 10000）会稳得多。"
        )
    if any(n_i < GAUSSIAN_N_MIN for n_i in ns):
        warnings.append(
            f"有 n < {GAUSSIAN_N_MIN} 的扫描点：那里的 √(2/3)/√M 只是量级参考，"
            f"因为高斯链的 ⟨R⁴⟩ = (5/3)⟨R²⟩² 在小 n 并不成立。⟨R²⟩ = n·l² 本身仍然精确。"
        )
    if slope_se is not None and abs(slope - slope_ref) > 3.0 * slope_se:
        ref_txt = "1" if slope_ref == 1.0 else f"本模型的理论斜率 {slope_ref:.4f}"
        warnings.append(
            f"拟合斜率 {slope:.4f} 偏离 {ref_txt} 超过三倍标准误（3σ = {3 * slope_se:.4f}），"
            f"这不该发生 —— 请把参数发回来看看。"
        )

    return SweepResult(
        l=l,
        seed=seed,
        chains=chains,
        ns=ns,
        R2_mean=R2_mean,
        R2_theory=R2_theory,
        h_rms_sim=[math.sqrt(v) for v in R2_mean],
        h_rms=[math.sqrt(v) for v in R2_theory],
        Cn_sim=Cn_sim,
        rel_dev=rel_dev,
        rel_se=rel_se,
        slope=slope,
        intercept=intercept,
        fit_r2=fit_r2,
        slope_se=slope_se,
        model=mp.key,
        params=mp.to_dict(),
        slope_ref=slope_ref,
        warnings=warnings,
    )


def freerotating_h2(n: int, l: float, theta_deg: float) -> float:
    """自由旋转链的均方末端距 ⟨h²⟩ = nl²(1−cosθ)/(1+cosθ)。

    只用于交叉验证：θ=90° 时应退化为 FJC 的 nl²。

    **这是教科书那条渐近式**（n → ∞）。模型层（model_h2）用的是有限 n 的精确
    离散式，两者差一个 O(1/n) 的常数项：聚乙烯键角、n=100 时分别是 Cn=2.000
    与 Cn=1.985。两个都要留着 —— 前者是「教科书值 2」这条交叉验证的锚点，
    后者才是「模拟出来的链」该对上的那条。别把其中一个当成笔误去「修正」。
    """
    c = math.cos(math.radians(theta_deg))
    return n * l * l * (1.0 - c) / (1.0 + c)


def _model_param_spec(key: str | None) -> dict:
    """三个模型参数在目录里的说明。`key=None` 表示「哪个都不用」。

    单独抽出来是为了不给自回避行走抄第二份默认值 —— 它不吃键角、内旋转、
    持久长度这三样（在格点上走，没有键角也没有持久长度），但它那一项照样得带
    `default`：前端的 syncModelFields() 会拿这三个默认值做字符串替换
    （把说明里的「键角固定为 109.47°」换成输入框里的当前值），缺了会抛。
    """
    return {
        "theta_deg": {
            "label": "键角 θ", "unit": "°", "default": DEFAULT_THETA_DEG,
            "min": THETA_EPS_DEG, "max": 180.0 - THETA_EPS_DEG,
            "used": key in (MODEL_FRC, MODEL_HINDERED),
        },
        "cos_phi": {
            "label": "内旋转平均余弦", "unit": "", "default": DEFAULT_COS_PHI,
            "min": COS_PHI_MIN, "max": COS_PHI_MAX,
            "used": key == MODEL_HINDERED,
        },
        "p": {
            "label": "持久长度", "unit": "链段长度", "default": DEFAULT_P,
            "min": 0.01, "max": P_MAX,
            "used": key == MODEL_WLC,
        },
    }


def model_catalog() -> list[dict]:
    """链模型的目录：前端拿它建选择器，并据此决定**哪些卡片适用**。

    和 KUHN_PRESETS 一样，**唯一的一份在这里**：/api/models 原样吐出去，
    前端不再抄第二份（抄了就会改一处忘一处，两处各说各话）。

    目录里有**两类**模型，用 `analytic` 这一个位区分：

    · `analytic: True` —— 四个解析模型（fjc / frc / hindered / wlc），⟨h²⟩ 有闭式解。
      力–伸长、h→n 反解、模型指纹、数值表、P(h) 分布、主结果全都成立；
    · `analytic: False` —— 自回避行走。**没有闭式解**（不是「还没推出来」，
      是它属于另一个普适类），只能实空间采样，上面那些「按公式算」的功能
      一条都不适用。

    前端就靠这一位把不适用的卡片收起来。别在各自的模块里再写一遍
    `model === 'saw'` —— 那正是这份目录要消掉的东西。
    """
    out = []
    for key in MODEL_KEYS:
        mp = ModelParams.of(key)
        out.append({
            "key": key,
            "label": MODEL_LABELS[key],
            "analytic": True,
            "formula": model_formula(mp),
            "note": model_note(mp),
            "params": _model_param_spec(key),
        })

    # 自回避行走：**不在 MODEL_KEYS 里**（理由见下面「自回避行走」那一节的开头 ——
    # 进了 MODEL_KEYS 就要给四个解析模型的公式层加一堆「本模型不适用」的分支），
    # 但它是用户能选的一个模型，所以要出现在同一个选择器、同一份目录里。
    out.append({
        "key": "saw",
        "label": "自回避行走（SAW）",
        "analytic": False,
        "formula": f"⟨R²⟩ ∝ n^(2ν)，ν = {SAW_NU:.7f}（3D 立方格）",
        "note": (
            "本模型没有闭式解 —— 这不是「还没推出来」，是它属于另一个普适类："
            "⟨R²⟩ ∝ n^(2ν) 里的 ν 只能数值定出来。所以这一页只做实空间采样，"
            f"在立方格点上严格拒绝采样（撞到自己就整条丢弃）。n ≤ {SAW_N_MAX} 是"
            "**算力**上限不是物理上限：存活率每加一个链段约乘 0.78，"
            "n = 30 比 n = 12 贵约 250 倍。"
            "力–伸长、h→n 反解、模型指纹、数值表、末端距分布这些「按公式算」的功能"
            "本模型一条都不适用，已经收起来了 —— 要看它们就切回上面四个解析模型。"
        ),
        "params": _model_param_spec(None),
    })
    return out


# --- 自回避行走（SAW）----------------------------------------------------
#
# 上面四个模型的 ⟨h²⟩ 都有闭式解，所以它们能共用同一套公式层。自回避行走
# **没有** —— 「同一个格点不许重复访问」这条约束把它推进了另一个普适类，
# 末端距只能测、不能推：
#
#     ⟨R²⟩ ∝ n^(2ν)，ν = 0.5875970（3D 立方格），2ν = 1.175194
#
# 所以它**不做成 MODEL_KEYS 的第五项**：一旦进去，force_extension / solve_n /
# model_fingerprint / model_catalog / AI 工具全都得加「本模型无解析式，不适用」
# 的分支，四个解析模型的公式层就被污染了。它单独成节、单独成卡，并自带一条
# 理想链对照 —— 于是「扫描卡验证斜率是不是 1」在这里变成「实测 ν 是不是 0.588」。
#
# 格点约定：立方格，6 个单位方向，步长 = l = 格点常数。

# 3D 立方格的普适指数。这不是「还没推出来的闭式解」，而是**不存在**闭式解 ——
# 它由数值/共形自举定出来，是这一节唯一的理论锚点。
SAW_NU = 0.5875970
SAW_SCALE_REF = 2.0 * SAW_NU        # ⟨R²⟩ 的标度指数 2ν = 1.175194

# 链段数上限。严格拒绝采样的存活率每加一步大约乘 0.79（渐近到 μ/6 = 0.7807），
# 所以代价随 n **指数**增长：n=30 比 n=10 贵两个数量级。这是**算力**上限，
# 不是物理上限。
SAW_N_MAX = 30
SAW_CHAINS_MAX = 20_000
SAW_DEFAULT_CHAINS = 1_000

# 试投预算。实测吞吐约 4×10⁶ 元素运算/秒，而每次试投平均约 16 个元素运算，
# 所以 150 万次试投 ≈ 6 秒（整次扫描），单点 120 万 ≈ 5 秒 —— 前端在这期间
# 显示忙状态。默认那组 n 加 1000 条链大致要 110 万次试投，正好在预算内。
SAW_TRIALS_PER_N = 1_200_000
SAW_TRIALS_TOTAL = 1_500_000
# 分块的内存上限（只管 keys 那张表，walk 是它的 1/8）。按**内存**反算块大小，
# 不按行数：M=2×10⁷ 行 × (n+1) 个 int64 键要 2~6 GB，这是踩过的坑。
_SAW_MEM_BYTES = 64_000_000

# 立方格的连接常数 μ（存活率渐近 ≈ (μ/6)^n）与指数修正 n^(11/32)，γ = 43/32 是
# 3D 的「磁化率」指数。**这两个只用于估算批量与预算**，不作为结果、也不进任何
# 报给用户的量 —— 页面上报的存活率一律是实测的 accept_rate。
_SAW_MU = 4.68404
_SAW_SUSCEPT_EXP = 11.0 / 32.0
_SAW_AMP = 0.81927            # 由下面表里 n=8 的精确值定出

# n ≤ 8 的**精确**一步自回避行走条数 c_n（立方格，OEIS A001412）。
# 用途有二：估批量/预算时给出准确值；以及校验实测存活率是否等于 c_n/6^n ——
# 这正是「采样器有没有退化成 Rosenbluth 采样」的判据（那个 bug 的现象就是存活率
# 恒为 1）。测试用**现场 DFS 枚举**把这张表逐项钉住。
#
# 只到 8 是有意的：n=9、10 的枚举要一两千万次访问，测试跑不动，而收进来的
# 未验证常量比少一点精度更糟。n ≥ 9 走渐近式（偏高约 4%），估批量够用。
_SAW_C_N = {
    0: 1, 1: 6, 2: 30, 3: 150, 4: 726, 5: 3534, 6: 16926, 7: 81390, 8: 387966,
}

# 位置都是 [-n, n] 内的小整数，用固定宽度打包成唯一整数键：成员判定就退化成
# 一次整数相等比较，不用哈希、不会碰撞，比集合快一个量级。
_SAW_OFF = SAW_N_MAX + 1
_SAW_W = 2 * SAW_N_MAX + 3
_SAW_ORIGIN = _SAW_OFF * (1 + _SAW_W + _SAW_W * _SAW_W)

SAW_DIRECTIONS = np.array(
    [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)],
    dtype=np.int64,
)


@dataclass
class SawChainResult:
    """自回避行走的一次实现 + 同格点理想链的对照。

    和 ChainResult 最大的不同：这里**没有解析式**可用。理想链那一侧的
    R2_ideal = n·l² 仍然精确（与 FJC 恒等），但 SAW 这一侧只有实测值，
    所以 rel_se 是**实测的** std/√M，而不是高斯公式 √(2/3)/√M —— SAW 的 R²
    分布比高斯窄，套高斯公式会把误差报大。swelling = R2_mean/(n·l²)。
    """

    n: int
    l: float
    seed: int
    chains: int              # 实际采到的链数（受试投上限约束，可能少于请求值）
    trials: int              # 试投了多少条链
    accept_rate: float       # 实测存活率 = chains/trials

    points: np.ndarray       # (chains, n+1, 3) SAW 顶点，第 0 个是原点
    R: np.ndarray            # (chains, 3) SAW 末端矢量
    R_mag: np.ndarray        # (chains,) |R|

    ideal_points: np.ndarray  # (chains, n+1, 3) 同 n 的理想链顶点
    ideal_R2_mean: float      # 理想链的实测 ⟨R²⟩

    R2_mean: float           # SAW 的实测 ⟨R²⟩
    R2_ideal: float          # n·l²，理想链的解析 ⟨h²⟩（与 FJC 恒等）
    R2_rel_se: float         # 实测相对标准误 std/(mean·√M)
    swelling: float          # R2_mean / R2_ideal，> 1 就是溶胀

    n_max: int = SAW_N_MAX       # 上限回显，前端据此提示而不是自己抄一份
    chains_max: int = SAW_CHAINS_MAX
    nu: float = SAW_NU
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """扁平化顶点数组并限位小数，和 ChainResult.to_dict() 同一套写法。"""
        q = CHAIN_DECIMALS

        def flat(pts):
            return [[round(float(v), q) for v in row.ravel()] for row in pts]

        return {
            "n": self.n,
            "l": self.l,
            "seed": self.seed,
            "chains": self.chains,
            "trials": self.trials,
            "accept_rate": self.accept_rate,
            "R2_mean": self.R2_mean,
            "R2_ideal": self.R2_ideal,
            "R2_rel_se": self.R2_rel_se,
            "ideal_R2_mean": self.ideal_R2_mean,
            "swelling": self.swelling,
            "R_mag": [round(float(v), q) for v in self.R_mag],
            "R": [[round(float(v), q) for v in row] for row in self.R],
            "points": flat(self.points),
            "ideal_points": flat(self.ideal_points),
            "n_max": self.n_max,
            "chains_max": self.chains_max,
            "nu": self.nu,
            "warnings": self.warnings,
        }


@dataclass
class SawSweepResult:
    """SAW 的 ⟨R²⟩ 对 n 的标度关系，附带理想链（斜率 1）与 2ν 两条参考线。

    和 SweepResult 的区别集中在「没有解析式」这一件事上：
    - rel_se 是**每个 n 一个**的实测值（SAW 的 R² 相对涨落随 n 变），
      所以 slope_se 用逐点误差传播，而不是「一个 rel_se 除以 √Sxx」。
    - 参考线有**两条**：理想链斜率 1、文献值 2ν = 1.175194。
    - 两条线都**锚在数据质心**上（不引任何无法自证的振幅）。
    """

    l: float
    seed: int
    chains: int               # 每个 n 请求的链数
    ns: list[int]

    R2_mean: list[float]      # 每个 n 的 SAW 实测 ⟨R²⟩
    R2_ideal: list[float]     # 每个 n 的 n·l²
    h_rms_sim: list[float]
    h_rms_ideal: list[float]
    swelling: list[float]     # R2_mean / R2_ideal
    rel_se: list[float]       # 每个 n 实测的相对标准误
    trials: list[int]         # 每个 n 试投了多少条链
    chains_used: list[int]    # 每个 n 实际采到多少条链

    slope: float | None       # log⟨R²⟩ 对 log n 的拟合斜率
    intercept: float | None
    fit_r2: float | None
    slope_se: float | None    # 逐点误差传播，见 saw_sweep()
    nu_eff: float | None      # slope/2，这片 n 区间的**有效** ν
    slope_ref: float = SAW_SCALE_REF   # 文献值 2ν
    intercept_ref: float | None = None  # 斜率 2ν、过质心的参考线
    intercept_ideal: float | None = None  # 斜率 1、过质心的参考线

    n_max: int = SAW_N_MAX
    chains_max: int = SAW_CHAINS_MAX
    nu: float = SAW_NU
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        def opt(v):
            return None if v is None else float(v)

        return {
            "l": self.l,
            "seed": self.seed,
            "chains": self.chains,
            "n": list(self.ns),
            "R2_mean": [float(v) for v in self.R2_mean],
            "R2_ideal": [float(v) for v in self.R2_ideal],
            "h_rms_sim": [float(v) for v in self.h_rms_sim],
            "h_rms_ideal": [float(v) for v in self.h_rms_ideal],
            "swelling": [float(v) for v in self.swelling],
            "rel_se": [float(v) for v in self.rel_se],
            "trials": [int(v) for v in self.trials],
            "chains_used": [int(v) for v in self.chains_used],
            "slope": opt(self.slope),
            "intercept": opt(self.intercept),
            "fit_r2": opt(self.fit_r2),
            "slope_se": opt(self.slope_se),
            "nu_eff": opt(self.nu_eff),
            "slope_ref": float(self.slope_ref),
            "intercept_ref": opt(self.intercept_ref),
            "intercept_ideal": opt(self.intercept_ideal),
            "nu": self.nu,
            "n_max": self.n_max,
            "chains_max": self.chains_max,
            "warnings": self.warnings,
        }


def _saw_survival(n: int) -> float:
    """存活率 c_n/6^n 的**估算值**。只用于批量与预算，不进结果。

    n ≤ 10 用精确表；再大用渐近式 A·(μ/6)^n·n^(11/32) —— 单用 (μ/6)^n 在有限 n
    偏得很厉害（n=8 时给 0.138，实际 0.231，差 1.7 倍），因为还差一个 n^(11/32)
    的修正和一个振幅。加上之后 n=11…30 的误差在 10% 出头，估批量足够。

    **这不是物理**：真值只有数值/枚举能给，页面上报的存活率一律是实测值。
    """
    if n in _SAW_C_N:
        return _SAW_C_N[n] / 6.0 ** n
    return _SAW_AMP * (_SAW_MU / 6.0) ** n * n ** _SAW_SUSCEPT_EXP


def _saw_next_batch_size(n: int, remaining: int, trials_left: int) -> int:
    """下一批投多少条：按存活率估算「还差多少试投」，再由内存和预算封顶。

    按内存上限整批投是不行的：n=10 时一批 72 万次试投会活下来十万条，而只要
    一千条 —— 一个点白烧 1.7 秒（这是实测到的）。估算偏了也不要紧，下一轮会
    拿**实测**存活率重新估，自己收敛。
    """
    est = _saw_survival(n)
    need = math.ceil(remaining / est) if est > 0 else trials_left
    batch_max = max(1, _SAW_MEM_BYTES // (8 * (n + 1)))
    return int(max(1, min(batch_max, trials_left, need)))


def _saw_endpoints(rng, n: int, want: int, trial_max: int):
    """采 want 条自回避行走，只要末端矢量（格点坐标）。

    返回 (R, trials, produced)。`produced` 是**试投中活下来的总条数**，于是
    produced/trials 才是实测存活率 —— 必须按「活下来多少」统计，不能按
    「返回了多少条」：最后一批会随手截断，那个比值会假得离谱（第一版就报出
    0.0014 这种数，而真值 0.146）。

    试投上限兜住时会**少给几条**，由调用方转成一条 warning ——「算力不够」
    不是「参数非法」，不该报错。
    """
    chunks, produced, trials = [], 0, 0
    while produced < want and trials < trial_max:
        m = _saw_next_batch_size(n, want - produced, trial_max - trials)
        pos, _ = _saw_batch(rng, n, m)
        trials += m
        produced += pos.shape[0]
        if pos.shape[0]:
            chunks.append(pos)
    R = (np.concatenate(chunks) if chunks
         else np.zeros((0, 3), dtype=np.int64))
    # 截断是无偏的：同一批里的存活链彼此可交换，取前 want 条即可。
    return R[:want], trials, produced


def _saw_key(xyz: np.ndarray) -> np.ndarray:
    """(k, 3) 的格点坐标 → (k,) 的唯一整数键。

    宽度 _SAW_W = 2·SAW_N_MAX+3 保证 ±n 内的坐标打包后互不重叠，所以键相等
    ⇔ 格点相同，成员判定可以当成纯整数比较来做。
    """
    return (xyz[:, 0] + _SAW_OFF
            + (xyz[:, 1] + _SAW_OFF) * _SAW_W
            + (xyz[:, 2] + _SAW_OFF) * _SAW_W * _SAW_W)


def _saw_batch(rng, n: int, m: int, keep_paths: bool = False):
    """一次试投 m 条链，返回 (存活的末端坐标 (k,3), 存活者走出的方向序列)。

    每一步在**全部 6 个方向**里均匀选一个；撞到已访问的格点就整条链作废。
    这个「撞上就丢」是严格均匀的关键 —— 若改成只在**空闲**邻居里均匀选
    （Rosenbluth 采样），链几乎不会死（要 6 个邻居全被占才死），但每条构象的
    权重不同、于是**有偏**，存活率也不再等于精确的 c_n/6^n。第一版就栽在这里，
    现象是每个 n 的存活率都是 1.00000。

    方向序列只在 keep_paths 时要（顶点由它累加出来，int8 比直接存 (m,n+1,3)
    的坐标省 24 倍内存，而内存正是这里分块的依据）。逐步剔除已死的链是为了
    **提前剪枝**：成员判定的代价是 O(k)，只给活着的链做，总量才是 Σ_k k·存活率
    ≈ 16.1·试投数，而不是 n·试投数。
    """
    pos = np.zeros((m, 3), dtype=np.int64)
    keys = np.empty((m, n + 1), dtype=np.int64)
    keys[:, 0] = _SAW_ORIGIN          # 原点必须占位，否则第一步就能走回来
    walk = np.empty((m, n), dtype=np.int8) if keep_paths else None
    live = np.ones(m, dtype=bool)

    for k in range(1, n + 1):
        rows = np.flatnonzero(live)
        if rows.size == 0:
            break
        sel = rng.integers(0, len(SAW_DIRECTIONS), size=rows.size)
        cand = pos[rows] + SAW_DIRECTIONS[sel]
        ck = _saw_key(cand)
        hit = (keys[rows, :k] == ck[:, None]).any(axis=1)
        good = rows[~hit]
        live[rows[hit]] = False
        if good.size == 0:
            continue
        if walk is not None:
            walk[good, k - 1] = sel[~hit]
        keys[good, k] = ck[~hit]
        pos[good] = cand[~hit]

    return pos[live], (walk[live] if walk is not None else None)


def _walk_to_points(walk: np.ndarray, l: float) -> np.ndarray:
    """方向下标序列 (k, n) → 顶点坐标 (k, n+1, 3)，首点是原点。"""
    steps = SAW_DIRECTIONS[walk.astype(np.int64)]      # (k, n, 3)
    pts = np.concatenate(
        [np.zeros((walk.shape[0], 1, 3), dtype=np.int64), np.cumsum(steps, axis=1)],
        axis=1,
    )
    return pts.astype(float) * l


def validate_saw(n, l, seed=None, chains=1,
                 max_chains: int | None = SAW_CHAINS_MAX,
                 max_n: int | None = SAW_N_MAX):
    """SAW 的输入校验：先过通用的 validate()，再卡两条**算力**上限。

    上限的理由和别的 validate_* 不一样，所以错误信息必须把它说清楚 ——
    这里既不是「画不下」也不是「物理上没意义」，而是严格拒绝采样太贵。
    """
    n_i, l_f = validate(n, l)
    if max_n is not None and n_i > max_n:
        raise FJCInputError(
            f"自回避行走的链段数不能超过 {max_n}：严格拒绝采样的存活率按 "
            f"(μ/6)^n ≈ 0.78^n 衰减，n={max_n} 比 n=10 贵约 250 倍。"
            f"这是算力上限，不是物理上限。"
        )
    chains_i = _as_int(chains, "链数")
    if chains_i < 1:
        raise FJCInputError("链数至少为 1")
    if max_chains is not None and chains_i > max_chains:
        raise FJCInputError(f"自回避行走最多采 {max_chains:,} 条链，收到 {chains_i:,}")
    return n_i, l_f, _as_seed(seed), chains_i


def saw_chain(n, l=1.0, seed=None, chains=1,
              max_chains: int | None = SAW_CHAINS_MAX,
              max_n: int | None = SAW_N_MAX,
              trial_max: int | None = SAW_TRIALS_PER_N) -> SawChainResult:
    """采 `chains` 条 n 步自回避行走，并给出同 n、同格点的理想链做对照。

    SAW 那一侧的顶点用严格拒绝采样（见 _saw_batch）；理想链那一侧是同一格点上
    「每步 6 选 1、不管走没走过」的普通随机行走，⟨R²⟩ = n·l² 精确成立（与 FJC
    恒等）。两者画在一起时，**唯一的差别就是那条排除约束**，这正是这张卡要讲的。

    实测 ⟨R²⟩ 应当**大于** n·l²（溶胀），比值就是 swelling。n=4 时约 1.39，
    n=12 时约 1.75 —— 短链端离渐近值还很远。

    三个上限都只影响「能跑多大」，传 None 即解除 —— 和 validate_chain() 一样，
    是为了让统计验证能开更大的样本。
    """
    n, l, seed, chains = validate_saw(
        n, l, seed, chains, max_chains=max_chains, max_n=max_n
    )
    if trial_max is None:
        trial_max = SAW_TRIALS_PER_N

    # SAW 与理想链用各自独立的子种子：既能整体复现，两条链也不共享随机流。
    ss = np.random.SeedSequence(seed)
    child_saw, child_ideal = ss.spawn(2)
    rng = np.random.default_rng(child_saw)

    want = chains
    got, trials = 0, 0
    paths, endpoints = [], []
    while got < want and trials < trial_max:
        m = _saw_next_batch_size(n, want - got, trial_max - trials)
        pos, walk = _saw_batch(rng, n, m, keep_paths=True)
        trials += m
        if pos.shape[0]:
            paths.append(walk)
            endpoints.append(pos)
            got += pos.shape[0]

    # 三条路都对齐到同一个链数，否则 points 和 R 的第一维对不上。
    k = min(want, got)
    if k <= 0:
        raise FJCInputError(
            f"n={n} 时试投 {trials:,} 条链一条都没活下来，请减小 n 或提高试投预算"
        )
    points = _walk_to_points(np.concatenate(paths)[:k], l)
    R = np.concatenate(endpoints)[:k].astype(float) * l

    ideal = _walk_to_points(
        np.random.default_rng(child_ideal).integers(
            0, len(SAW_DIRECTIONS), size=(k, n)
        ).astype(np.int8),
        l,
    )

    R_mag = np.linalg.norm(R, axis=1)
    R2_mean = float(np.mean(R_mag ** 2))
    ideal_R2_mean = float(np.mean(np.sum(ideal[:, -1, :] ** 2, axis=1)))
    R2_ideal = n * l * l
    # 实测的相对标准误：std(R²)/(⟨R²⟩·√M)。不用高斯的 √(2/3)/√M ——
    # SAW 的 R² 分布比高斯窄（实测 std(R²)/⟨R²⟩ ≈ 0.50~0.63 对 0.816），
    # 而且这个比值随 n 变，套高斯公式会把误差报大。
    rel_se = (float(np.std(R_mag ** 2, ddof=1)) / R2_mean / math.sqrt(k)
              if k > 1 and R2_mean > 0 else 0.0)
    # 存活率按**活下来的总条数**算，不是按返回的条数 —— 最后一批会截断。
    accept_rate = got / trials if trials else 0.0

    warnings: list[str] = []
    if k < want:
        warnings.append(
            f"试投上限 {trial_max:,} 兜住了：请求 {want:,} 条，只采到 {k:,} 条"
            f"（n={n} 的实测存活率 {accept_rate * 100:.3f}%）。要更多样本请减小 n。"
        )
    # 存活率对不对得上精确的 c_n/6^n，是「采样器有没有退化成 Rosenbluth 采样」
    # 的判据 —— 那个 bug 的现象是存活率恒等于 1。只在 _SAW_C_N 覆盖到的 n 上比，
    # 因为那里有**精确**的 c_n；更大的 n 只有渐近估计（差 10% 出头），
    # 撑不起这么紧的判据。
    if n in _SAW_C_N and trials >= 20_000:
        exact = _saw_survival(n)
        if abs(accept_rate - exact) / exact > 0.10:
            warnings.append(
                f"实测存活率 {accept_rate:.5f} 与精确的 c_n/6^n = {exact:.5f} "
                f"对不上（差 {abs(accept_rate - exact) / exact * 100:.0f}%），"
                f"这不正常 —— 请把参数发回来看看。"
            )
    if n < 5:
        warnings.append(
            f"n={n} 太短，⟨R²⟩ 还看不出标度行为；自回避的普适指数要到 n 几十以上才显现。"
        )

    return SawChainResult(
        n=n, l=l, seed=seed, chains=k, trials=trials, accept_rate=accept_rate,
        points=points, R=R, R_mag=R_mag,
        ideal_points=ideal, ideal_R2_mean=ideal_R2_mean,
        R2_mean=R2_mean, R2_ideal=R2_ideal, R2_rel_se=rel_se,
        swelling=R2_mean / R2_ideal,
        warnings=warnings,
    )


def saw_sweep(ns, l=1.0, seed=None, chains=SAW_DEFAULT_CHAINS,
              max_points: int | None = SWEEP_N_MAX,
              max_chains: int | None = SAW_CHAINS_MAX,
              trials_per_n: int | None = SAW_TRIALS_PER_N,
              trials_total: int | None = SAW_TRIALS_TOTAL) -> SawSweepResult:
    """扫一组 n，每个 n 采 `chains` 条自回避行走，拟合 ⟨R²⟩ ∝ n^(2ν)。

    这是把「自回避行走」从一句话变成一次**可验证的测量**：对 log⟨R²⟩ 与 log n
    做最小二乘拟合，再和 2ν = 1.175194 比。

    **要如实预期**：n ≤ 30 这片区间测出来的斜率约 1.20~1.22，**高于** 2ν ——
    有限尺寸修正的符号在这里是正的，渐近值要 n 很大才到。所以这里报的是
    「这片 n 区间的**有效** ν = slope/2」，不是「测出了 0.588」。

    slope_se 用逐点误差传播（每个 n 的实测 rel_se 不同），当各点误差相同时
    退化成 SweepResult 里那条 rel_se/√Sxx。
    """
    n_list, l = _as_n_list(ns), 1.0
    # 逐项校验（沿用 validate_saw 的中文信息），再单独看链数与预算
    checked = []
    for raw in n_list:
        n_i, l = validate(raw, l)
        checked.append(n_i)
    if not checked:
        raise FJCInputError("请至少输入一个链段数 n")
    if max_points is not None and len(checked) > max_points:
        raise FJCInputError(f"一次最多扫 {max_points} 个 n，收到 {len(checked)} 个")
    if len(set(checked)) != len(checked):
        raise FJCInputError("链段数 n 有重复，请去掉重复值")
    for n_i in checked:
        if n_i > SAW_N_MAX:
            raise FJCInputError(
                f"自回避行走的链段数不能超过 {SAW_N_MAX}（收到 {n_i}）："
                f"每加一个链段，存活率大约乘 0.78，代价是指数增长的，"
                f"再大就跑不完。这是算力上限，不是物理上限。"
            )
    ns = checked

    chains_i = _as_int(chains, "每个 n 的链数")
    if chains_i < 1:
        raise FJCInputError("每个 n 的链数至少为 1")
    if max_chains is not None and chains_i > max_chains:
        raise FJCInputError(
            f"自回避行走每个 n 最多采 {max_chains:,} 条链，收到 {chains_i:,}"
        )
    seed = _as_seed(seed)

    # 预算：每个 n 要 trials ≈ chains / 存活率，封顶在 trials_per_n。
    # 提前拦比跑到一半才失败好，但**这个估计只用来拦**：真正花掉多少次试投由
    # 采样器实测，结果里的 trials 是实测值，不是这里的估算。
    cap = trials_per_n if trials_per_n is not None else SAW_TRIALS_PER_N
    est_total = sum(min(cap, math.ceil(chains_i / _saw_survival(n_i)))
                    for n_i in ns)
    if trials_total is not None and est_total > trials_total:
        raise FJCInputError(
            f"计算量超出预算（估算需试投 {est_total:,} 次 > {trials_total:,}），"
            f"请降低链数或减少大 n 的个数"
        )

    children = np.random.SeedSequence(seed).spawn(len(ns))
    R2_mean, R2_ideal, rel_se, trials_used, chains_used, accept = [], [], [], [], [], []
    for n_i, child in zip(ns, children):
        R, t, produced = _saw_endpoints(
            np.random.default_rng(child), n_i, chains_i, cap
        )
        m = R.shape[0]
        if m == 0:
            raise FJCInputError(
                f"n={n_i} 时试投 {t:,} 条链一条都没活下来，请减小 n 或提高预算"
            )
        r2s = np.sum(R.astype(float) ** 2, axis=1) * l * l
        r2 = float(np.mean(r2s))
        R2_mean.append(r2)
        R2_ideal.append(n_i * l * l)
        trials_used.append(int(t))
        chains_used.append(int(m))
        accept.append(produced / t if t else 0.0)
        # 实测 R² 的相对标准误（ddof=1），逐点不同 —— 这是和 SweepResult 的关键差异
        rel_se.append(float(np.std(r2s, ddof=1) / r2 / math.sqrt(m)) if m > 1 else 0.0)

    slope = intercept = fit_r2 = slope_se = nu_eff = None
    intercept_ref = intercept_ideal = None
    if len(ns) >= 2:
        ln_n = np.log(np.asarray(ns, dtype=float))
        ln_y = np.log(np.asarray(R2_mean, dtype=float))
        slope_f, intercept_f = np.polyfit(ln_n, ln_y, 1)
        slope, intercept = float(slope_f), float(intercept_f)
        ss_res = float(np.sum((ln_y - (slope * ln_n + intercept)) ** 2))
        ss_tot = float(np.sum((ln_y - ln_y.mean()) ** 2))
        fit_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        # 逐点误差传播：Var(slope) = Σ(xᵢ−x̄)²σᵢ² / Sxx²。各点 σᵢ 相同时
        # 正是 SweepResult 用的 σ/√Sxx。
        xbar = float(ln_n.mean())
        sxx = float(np.sum((ln_n - xbar) ** 2))
        if sxx > 0:
            num = float(np.sum(((ln_n - xbar) ** 2) * np.asarray(rel_se) ** 2))
            slope_se = math.sqrt(num) / sxx
            nu_eff = slope / 2.0
        # 两条参考线都过**数据质心**，只有斜率是外部信息 —— 不引振幅。
        ybar = float(ln_y.mean())
        intercept_ideal = ybar - 1.0 * xbar
        intercept_ref = ybar - SAW_SCALE_REF * xbar

    warnings: list[str] = []
    if len(ns) < 2:
        warnings.append("只给了一个 n，拟合不出标度斜率；至少要两个不同的 n。")
    elif len(ns) == 2:
        warnings.append(
            "只有两个 n：任何两点都能连成一条直线，决定系数 R² 必然是 1，说明不了什么。"
        )
    short = [n_i for n_i, c in zip(ns, chains_used) if c < chains_i]
    if short:
        warnings.append(
            "这些 n 没采满请求的链数（试投上限兜住）："
            + "、".join(f"n={n_i}" for n_i in short)
            + f"。它们的点会明显地跳，判断斜率时请把这一条算进去。"
        )
    if any(n_i < 10 for n_i in ns):
        warnings.append(
            "有 n < 10 的扫描点：那里离渐近区还很远，会把拟合斜率推高。"
            "自回避行走的 2ν = 1.175194 是 n → ∞ 的极限。"
        )
    # 阈值对着 slope_ref 而不是 1.0：这条说的是「比 2ν 高得太多」，
    # 拿 1.0 当基准的话，斜率落在 1 和 2ν 之间（不可能由采样误差产生）
    # 也会被判成「高于 2ν」，那条消息就成了假话。
    #
    # 而且**只在全是 n ≥ 10 时才判**：带了小 n 的点时，有限尺寸修正本来就会把
    # 斜率推高 —— 默认那组（含 5、8）实测稳定落在 2ν 之上 1.5~5σ，是**正常**
    # 结果。那样的话这条既不是异常信号、又和上一条说同一件事，默认参数一跑就
    # 弹两条黄条、其中一条还说「高 3.9 倍标准误」，读起来像出了问题。
    # 小 n 的解释交给上一条；这条留给「全是 n ≥ 10、斜率却仍然高得离谱」——
    # 实测这种情况下 10 个种子最大也只到 1.8σ，真越过 3σ 就值得看一眼。
    if (slope is not None and slope_se is not None
            and all(n_i >= 10 for n_i in ns)
            and slope > SAW_SCALE_REF + 3.0 * slope_se):
        # 这里**不是**异常告警：偏高是有限尺寸修正的正常表现，要说清方向。
        warnings.append(
            f"拟合斜率 {slope:.4f}（有效 ν = {nu_eff:.4f}）比 2ν = "
            f"{SAW_SCALE_REF:.4f} 高 {(slope - SAW_SCALE_REF) / slope_se:.1f} 倍标准误。"
            f"这片 n 区间偏高的有限尺寸修正是预期之内的，不是采样出错 —— "
            f"渐近值要到 n 很大才显现，而 n > {SAW_N_MAX} 跑不完。"
        )

    return SawSweepResult(
        l=l, seed=seed, chains=chains_i, ns=ns,
        R2_mean=R2_mean, R2_ideal=R2_ideal,
        h_rms_sim=[math.sqrt(v) for v in R2_mean],
        h_rms_ideal=[math.sqrt(v) for v in R2_ideal],
        swelling=[a / b for a, b in zip(R2_mean, R2_ideal)],
        rel_se=rel_se, trials=trials_used, chains_used=chains_used,
        slope=slope, intercept=intercept, fit_r2=fit_r2, slope_se=slope_se,
        nu_eff=nu_eff, intercept_ref=intercept_ref, intercept_ideal=intercept_ideal,
        warnings=warnings,
    )


# --- 力–伸长（Langevin）-------------------------------------------------
#
# FJC 在外力 f 下，单个链段的取向分布是 exp(x·cosθ)，x ≡ f·l/(k_B·T) 是
# **无量纲力**。链段之间互相独立（正是 FJC 的定义），所以整条链沿力方向的
# 归一化伸长就是单链段的平均投影：
#
#     λ ≡ ⟨x⟩/(n·l) = L(x) = coth(x) − 1/x
#
# 两条重要的极限，页面和测试都盯着它们：
#   · x → 0：L(x) ≈ x/3，代回去就是 ⟨x⟩ ≈ n·l²·f/(3k_BT)，正是高斯链的
#     熵弹性 3k_BT⟨x⟩/(n·l²) = f。**自由连接链在小力下必须退化成高斯链**，
#     这是两条独立推导的链模型互相咬合的地方。
#   · x → ∞：L(x) → 1，即链被完全拉直；但要到 λ=1 需要无穷大的力。
#
# **λ = L(x) 的反解没有解析解**（L 本身是初等函数，它的反函数不是）。
# 文献里有 Marko–Siggia 一类插值式，但它们在 λ→1 处明显偏离，而且是
# 近似式的近似 —— 这里宁可直接解：二分到机器精度，正确性只依赖 L 单调。
#
# 无量纲量为主、不掺单位，和本模块其余部分一致（l 的单位由调用方定）。
# 只有最后那个「换算成 pN」是唯一碰 SI 单位的地方，见 force_in_pn()。

# λ 必须 < 1。留 1e-12 的余量：λ 顶到 1 时反解要发散到无穷。
LAMBDA_MAX = 1.0 - 1e-12
# 横轴（无量纲力）的护栏。L(100) = 0.99，再往右曲线已经贴着 λ=1 看不出变化。
FORCE_X_MAX = 100.0
# 力–伸长曲线的采样点数，与 P(h) 一致
FORCE_CURVE_POINTS = CURVE_POINTS
# 玻尔兹曼常数，J/K。2019 年 SI 修订后它已是**定义精确值**，不是测量值。
K_B = 1.380649e-23
# 常温，开尔文。给 force_in_pn() 一个不至于离谱的默认。
DEFAULT_TEMPERATURE = 298.15


def langevin(x: float) -> float:
    """Langevin 函数 L(x) = coth(x) − 1/x。

    x 是无量纲力 f·l/(k_B·T)。返回值就是归一化伸长 λ = ⟨x⟩/(n·l)，落在 [0,1)。

    **x → 0 处必须走级数**：coth 和 1/x 各自 → ∞，直接相减是 ∞ − ∞ = nan。
    级数 L(x) = x/3 − x³/45 + 2x⁵/945 − x⁷/4725 + … 在 |x| < 0.01 时
    截断误差在 1e-14 量级（x⁷ 项相对 x/3 已是 6e-15）。

    而 |x| ≥ 0.01 时用 1/tanh(x) − 1/x 这个写法（= (x − tanh x)/(x·tanh x) 的
    等价形）：x = 0.01 处 x 与 tanh x 相减只损失约 1e-11 的相对精度，
    比级数在此处的表现好，也不用再加项。
    """
    xf = float(x)
    if not math.isfinite(xf):
        raise FJCInputError("无量纲力必须是有限数")
    if abs(xf) < 1e-2:
        x2 = xf * xf
        return xf * (1.0 / 3.0 - x2 * (1.0 / 45.0 - x2 * (2.0 / 945.0 - x2 / 4725.0)))
    return 1.0 / math.tanh(xf) - 1.0 / xf


def inverse_langevin(lam: float) -> float:
    """解 L(x) = λ，返回无量纲力 x ≥ 0。λ ∈ [0,1)。

    没有解析解，用二分法解到机器精度。

    **上界的选取是证明过的，不是试出来的**：对 x > 0 有
    coth(x) − 1 = 2/(e^{2x} − 1) > 0，故
        L(x) = 1 + (coth x − 1) − 1/x > 1 − 1/x。
    于是取 x_hi = 1/(1−λ) 就有 L(x_hi) > λ —— 区间一定夹住真解。
    再留一个翻倍的兜底，防的是浮点在 λ 极接近 1 时把那点不等式磨平。

    二分次数取 120：区间宽度每轮减半，120 轮后远超双精度分辨率，
    继续迭代也只会来回横跳。**这是正确性护栏，不是性能调优** ——
    曲线最多取 FORCE_CURVE_POINTS 个点，多花这点时间无所谓。
    """
    lv = float(lam)
    if not math.isfinite(lv):
        raise FJCInputError("伸长比 λ 必须是有限数")
    if lv < 0:
        raise FJCInputError(f"伸长比 λ 不能为负，收到 {lv}")
    if lv >= 1.0:
        raise FJCInputError(
            f"伸长比 λ 必须小于 1（λ=1 意味着完全拉直，需要无穷大的力），收到 {lv}"
        )
    if lv == 0.0:
        return 0.0

    lo, hi = 0.0, 1.0 / (1.0 - lv)
    # 兜底：真解一定 ≤ hi（上面证过），但浮点可能让 L(hi) 差一点点不够
    for _ in range(8):
        if langevin(hi) >= lv:
            break
        hi *= 2.0
    else:
        raise FJCInputError(f"解不出 λ = {lv} 对应的力，请把这个值报回来")

    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if langevin(mid) < lv:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _pn_scale(temperature: float, l_nm: float) -> float:
    """单位换算的那**一个**乘子：`x × 它 = 力（pN）`。

    抽出来是因为两个方向共用 —— 各写一份的话，改了单位链忘改另一边，
    pN 和 x 就会对不上，而且那种错从图上看不出来。
    校验也放这里，两条路吃到的中文报错完全一致。
    """
    try:
        tf, lf = float(temperature), float(l_nm)
    except (TypeError, ValueError):
        raise FJCInputError(
            f"温度和链段长度必须是数字，收到 T={temperature!r} l={l_nm!r}"
        ) from None
    if not math.isfinite(tf) or not math.isfinite(lf):
        raise FJCInputError(f"温度和链段长度必须是有限数，收到 T={tf} l={lf}")
    if not (0.0 < tf < 1000.0):
        raise FJCInputError(f"温度 T 要在 0 到 1000 K 之间，收到 {tf}")
    if not (0.0 < lf < 1e6):
        raise FJCInputError(f"链段长度（nm）要大于 0，收到 {lf}")
    return K_B * tf / (lf * 1e-9) * 1e12


def force_in_pn(x: float, temperature: float = DEFAULT_TEMPERATURE,
                l_nm: float = 1.0) -> float:
    """把无量纲力 x = f·l/(k_BT) 换算成飞牛…… 毫牛？—— 换算成 **pN**。

    **这里假设 l 以 nm 计、T 以开尔文计**，两者都是调用方给的，函数不猜单位。
    单位链：f = x·k_B·T/l，l 用 m = l_nm×1e⁻⁹，再乘 1e12 换成 pN：

        f[pN] = x · 1.380649e-23 · T / (l_nm × 1e-9) × 1e12
              = x · 0.01380649 · T / l_nm

    量级自检：T=300 K、l=1 nm、x=1 → 4.14 pN，正是室温 k_BT/nm 的标准值。
    反方向是 `x_from_force_pn()`。
    """
    try:
        xf = float(x)
    except (TypeError, ValueError):
        raise FJCInputError(f"无量纲力必须是数字，收到 {x!r}") from None
    if not math.isfinite(xf):
        raise FJCInputError("无量纲力必须是有限数")
    if xf < 0:
        raise FJCInputError(f"无量纲力不能为负，收到 {xf}")
    return xf * _pn_scale(temperature, l_nm)


def x_from_force_pn(force_pn: float, temperature: float = DEFAULT_TEMPERATURE,
                    l_nm: float = 1.0) -> float:
    """`force_in_pn()` 的反函数：**pN → 无量纲力 x**。

    为什么要它：用户问的是「0.5 pN 能拉多长」，用的是 pN 这个单位，
    而 Langevin 函数只认 x。中间这一步换算是单位问题，属于物理，
    所以放在内核里 —— `ai_tools` 一行公式都不该有。

    换算是**线性**的（f = x·k_BT/l），所以没有近似，也不用迭代；
    需要迭代的只有下一步 λ = L(x)。
    """
    try:
        fp = float(force_pn)
    except (TypeError, ValueError):
        raise FJCInputError(f"力必须是数字，收到 {force_pn!r}") from None
    if not math.isfinite(fp):
        raise FJCInputError("力必须是有限数")
    if fp < 0:
        raise FJCInputError(f"力不能为负，收到 {fp}")
    return fp / _pn_scale(temperature, l_nm)


@dataclass
class ForceResult:
    """力–伸长曲线：横轴无量纲力 x = f·l/(k_BT)，纵轴归一化伸长 λ = L(x)。

    **这条曲线与 n、l 都无关** —— L(x) 里根本没有它们。这正是它值得单独画的
    地方：像 P(h) 的归一化视图一样，它讲的是一条与具体链长无关的普适关系。
    n、l 只在「绝对伸长是多少个长度单位」那一步才进来（curve_ext）。
    """

    l: float
    n: int
    temperature: float
    x_max: float

    curve_x: np.ndarray      # 无量纲力 f·l/(k_BT)
    curve_lambda: np.ndarray # λ = L(x)，归一化伸长，恒在 [0,1)
    curve_ext: np.ndarray    # ⟨x⟩ = n·l·λ，绝对伸长（以 l 的单位计）
    curve_force_pn: np.ndarray  # 同一个 x 对应的 f，按 temperature 与 l=1nm 换算

    # 高斯链（熵弹性）那条直线 λ_lin = x/3，作为对照画在同一张图上。
    # **只截到 λ_lin ≤ 1 为止**（即 x ≤ 3）：再往右它会预测出超过全伸展的伸长，
    # 那已经不是「近似得不好」而是无意义。这条线要在整段 x 上画出来、
    # 不要只画到差 2% 的地方 —— 图要讲的正是**两条线在哪分开**，
    # 把直线在分叉处剪掉等于把结论藏起来。
    curve_linear_x: np.ndarray
    curve_linear_lambda: np.ndarray

    lambda_at_x_max: float   # 曲线右端的 λ
    x_linear_end: float      # 直线与 Langevin 相差 2% 的位置，给页面标注用

    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "l": self.l,
            "temperature": self.temperature,
            "x_max": self.x_max,
            "curve": {
                "x": [float(v) for v in self.curve_x],
                "lambda": [float(v) for v in self.curve_lambda],
                "ext": [float(v) for v in self.curve_ext],
                "force_pn": [float(v) for v in self.curve_force_pn],
            },
            "curve_linear": {
                "x": [float(v) for v in self.curve_linear_x],
                "lambda": [float(v) for v in self.curve_linear_lambda],
            },
            "lambda_at_x_max": self.lambda_at_x_max,
            "x_linear_end": self.x_linear_end,
            "warnings": self.warnings,
        }


def _lin_dev(x: float) -> float:
    """L(x)/x 相对高斯链近似 1/3 的偏差。x=0 处定义为 0（级数首项就是 1/3）。"""
    if x <= 0.0:
        return 0.0
    return abs(langevin(x) / x - 1.0 / 3.0) / (1.0 / 3.0)


def force_extension(n, l=1.0, x_max=10.0, points: int = FORCE_CURVE_POINTS,
                    temperature: float = DEFAULT_TEMPERATURE) -> ForceResult:
    """算 FJC 的力–伸长曲线 λ(x) = L(x)，x ∈ [0, x_max]。

    n、l 用的是和主图同一套 validate()，所以单位、上限、错误文案都一致。
    """
    n_i, l_f = validate(n, l)

    try:
        xm = float(x_max)
    except (TypeError, ValueError):
        raise FJCInputError("横轴上限 x_max 必须是数字") from None
    if not math.isfinite(xm) or xm <= 0:
        raise FJCInputError(f"横轴上限 x_max 必须是正数，收到 {x_max}")
    if xm > FORCE_X_MAX:
        raise FJCInputError(f"横轴上限 x_max 不能超过 {FORCE_X_MAX:g}，收到 {xm:g}")

    p = _as_int(points, "采样点数")
    if p < 2 or p > 2000:
        raise FJCInputError(f"采样点数要落在 2 到 2000 之间，收到 {p}")
    # 温度只在 force_in_pn 里用，但校验提前做，免得画到一半才炸
    force_in_pn(1.0, temperature, l_f)

    xs = np.linspace(0.0, xm, p)
    # langevin 是标量函数，400 点直接逐个算：np.vectorize 反而更慢，而这点量
    # 逐点循环不到 1 毫秒 —— 关键是每个点都走同一套 x→0 分支，不会串味
    lams = np.array([langevin(v) for v in xs], dtype=float)

    # 线性区上限：λ ≈ x/3 与真值差 2% 的位置。先算它，下面两条提示都要引用。
    # 偏差 dev(x) = |L(x)/x − 1/3| / (1/3) 随 x **单调增**（L/x 从 1/3 往下掉），
    # 所以 dev ≤ 阈值的区间是 [0, x*]，二分时 lo 吃满足的那半、hi 吃不满足的那半。
    # 上界从 1 起倍增到夹住为止 —— 不写死 5.0，免得将来改了 2% 这个数却忘了同步。
    thr = 0.02
    lo, hi = 0.0, 1.0
    while _lin_dev(hi) < thr and hi < 64.0:
        hi *= 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _lin_dev(mid) <= thr:
            lo = mid
        else:
            hi = mid
    x_linear_end = 0.5 * (lo + hi)

    # 高斯链直线截到 λ=1（x=3）为止，见 curve_linear_x 的字段说明。
    # x ≤ 3 时 x/3 ≤ 1，所以这个掩码就是「直线还在物理意义内」的那段。
    lin_mask = xs <= 3.0

    warnings: list[str] = []
    lam_max = float(lams[-1])
    # 0.3：再窄就只剩原点附近一小截，看不出曲线的形状（默认 x_max=10 给到 λ=0.90，
    # 不该在默认值上就报警 —— 默认配置应当是干净的）。
    if lam_max < 0.3:
        warnings.append(
            f"横轴只到 x={xm:g}，伸长还不到 {lam_max * 100:.0f}% —— "
            f"想看接近拉直的行为，把 x_max 调到 30 以上。"
        )
    # 0.95（x ≈ 20）才报，理由不是 Langevin 在 FJC 内部不准 —— **它在 FJC 内部是精确的**，
    # 链段刚性正是 h_max = nl 的来源。报的是另一件事：拉到这个程度时，
    # 真实材料早就不满足「n 个刚性小球自由铰接」这个前提了
    # （键角受限、链段会被拉长、甚至先断），所以把这段当材料的实测极限是错的。
    if lam_max > 0.95:
        warnings.append(
            f"右端已经到 λ={lam_max:.3f}。**这个区间在 FJC 内部是精确的**，"
            f"但真实链在大伸长下不再满足前提 —— 刚性链段会被拉长、键角会转，"
            f"材料通常先断。别把这段外推当成实测极限。"
        )

    return ForceResult(
        n=n_i,
        l=l_f,
        temperature=float(temperature),
        x_max=xm,
        curve_x=xs,
        curve_lambda=lams,
        curve_ext=lams * (n_i * l_f),
        curve_force_pn=np.array(
            [force_in_pn(v, temperature, l_f) for v in xs], dtype=float
        ),
        curve_linear_x=xs[lin_mask],
        curve_linear_lambda=xs[lin_mask] / 3.0,
        lambda_at_x_max=lam_max,
        x_linear_end=x_linear_end,
        warnings=warnings,
    )


# --- h → n 反解 ---------------------------------------------------------
#
# 主图给的是「输入 n 得到 h」，反过来「想要某个 h 需要多少个链段」同样常用：
# 选一段真实链，测到末端距，问该切成几个 Kuhn 链段。
#
# 四个特征量各自的反解式，与 compute() 里的正向公式一一对应：
#   h_rms  = l√n                ⟹ n = (h/l)²
#   ⟨h⟩    = l√(8n/(3π))        ⟹ n = (3π/8)(h/l)²
#   h*     = l√(2n/3)           ⟹ n = (3/2)(h/l)²
#   nl     = n·l                ⟹ n = h/l        ← 唯一不是平方关系的
#
# 反解是**连续**的，而链段数是整数，所以必然要取整：
#   ⌊⌋ 保证末端距**不超过**目标，⌈⌉ 保证**不低于**目标。
# 两个都回传，页面上并排显示 —— 由用户按「要够」还是「别超」来挑，
# 不替他选。h 随 n 单调增（四种都是），所以这两个结论是稳的。

H_KINDS = ("h_rms", "h_mean", "h_mp", "h_max")

H_KIND_LABELS = {
    "h_rms": "根均方末端距 h_rms",
    "h_mean": "平均末端距 ⟨h⟩",
    "h_mp": "最可几末端距 h*",
    "h_max": "全伸展长度 nl",
}


def _h_to_n_exact(h: float, l: float, kind: str) -> float:
    """连续反解：给定 h 求（允许非整数的）n。"""
    r = h / l
    if kind == "h_rms":
        return r * r
    if kind == "h_mean":
        return 3.0 * math.pi / 8.0 * r * r
    if kind == "h_mp":
        return 1.5 * r * r
    return r  # h_max = n·l


def _n_to_h(n: float, l: float, kind: str) -> float:
    """正向公式。**刻意写成和 compute() 同一棵表达式树**，不是代数等价的另一版。

    原先这里是 `l * math.sqrt(8n/(3π))`，代数上对，但浮点求值顺序不同，
    与 compute() 的 `2/(beta·√π)` 会差 1 ulp（n=1, l=1 时 h_mean 就差在
    最后一位）。差 1 ulp 本身不致命，可它说明反解的正向**根本没走内核那条路** ——
    「不是第二套公式」这句话就落空了。所以这里一律复用 beta_of() 和
    compute() 里的同一串运算，逐位相同。
    """
    if kind == "h_max":
        return n * l
    h2 = n * l * l
    if kind == "h_rms":
        return math.sqrt(h2)
    beta = beta_of(n, l)
    if kind == "h_mp":
        return 1.0 / beta
    return 2.0 / (beta * math.sqrt(math.pi))


@dataclass
class SolveResult:
    """h → n 反解的结果。n_exact 是连续解，n_floor / n_ceil 是可用的整数解。"""

    kind: str
    kind_label: str
    h: float
    l: float
    n_exact: float
    n_floor: int
    n_ceil: int
    h_floor: float
    h_ceil: float
    warnings: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        """推荐的整数解：⌈⌉ 保证末端距不低于目标。"""
        return self.n_ceil

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "kind_label": self.kind_label,
            "h": self.h,
            "l": self.l,
            "n_exact": self.n_exact,
            "n": self.n,
            "n_floor": self.n_floor,
            "n_ceil": self.n_ceil,
            "h_floor": self.h_floor,
            "h_ceil": self.h_ceil,
            "warnings": self.warnings,
        }


def solve_n(h, l=1.0, kind: str = "h_rms") -> SolveResult:
    """给定某个特征末端距 h，反解需要多少个链段。

    kind 选哪个特征量（默认主输出 h_rms）—— 「要 100」指的是哪一种 100，
    这件事必须问清楚，否则 h_rms=100 和 nl=100 差着一到两个数量级。
    """
    if kind not in H_KINDS:
        raise FJCInputError(
            f"kind 只能是 {'、'.join(H_KINDS)} 之一，收到 {kind!r}"
        )
    # l 借用 validate() 的校验（顺带把 n 的上限也看一眼，这里用不到但无妨）
    _, l_f = validate(1, l)

    try:
        hv = float(h)
    except (TypeError, ValueError):
        raise FJCInputError("末端距 h 必须是数字") from None
    if not math.isfinite(hv):
        raise FJCInputError("末端距 h 必须是有限数")
    if hv <= 0:
        raise FJCInputError(f"末端距 h 必须大于 0，收到 {hv}")

    n_exact = _h_to_n_exact(hv, l_f, kind)

    # 贴着整数时把浮点噪声抹掉。
    #
    # 反解是「先开方再平方」走出来的（h = l√(8n/(3π)) ⟹ n = (3π/8)(h/l)²），
    # 一个 ulp 的舍入会被平方放大，于是由 n=64 造出来的 h 反解回去会得到
    # 64.00000000000001 —— 直接取 ⌈⌉ 就成了 65。**用户要的是 64 段，不是 65 段**，
    # 这个 +1 是纯噪声，不是信息。
    #
    # 容差取相对 1e-13：比浮点往返的几个 ulp（~1e-15 相对）宽两个数量级，
    # 同时远小于任何有意义的小数部分（n=1e6 上要小于 1e-7 才会被抹，
    # 而真的差 1e-7 个链段本来就说明不了什么）。
    nearest = round(n_exact)
    if abs(n_exact - nearest) <= 1e-13 * max(1.0, abs(n_exact)):
        n_exact = float(nearest)

    warnings: list[str] = []
    if n_exact < 1.0:
        # 四种都是 n 的增函数，n<1 意味着连一个链段都用不满 —— h 比 l 还小
        raise FJCInputError(
            f"反解出来的 n = {n_exact:.4g} 小于 1：目标 h={hv:.4g} "
            f"比一个链段还短（l={l_f:.4g}），凑不出这样一条链"
        )
    if n_exact > N_MAX:
        raise FJCInputError(
            f"反解出来的 n = {n_exact:.4g} 超过上限 {N_MAX:,}"
        )

    n_floor = max(1, int(math.floor(n_exact)))
    n_ceil = max(1, int(math.ceil(n_exact)))
    h_floor = _n_to_h(n_floor, l_f, kind)
    h_ceil = _n_to_h(n_ceil, l_f, kind)

    # 取整什么时候要紧？只在 n 很小时。判据直接拿「两个整数解给出的 h 差多少」：
    #   √ 型（h_rms / ⟨h⟩ / h*）：h ∝ √n，比值 = √(⌈n⌉/⌊n⌋)
    #   h_max：                h ∝ n， 比值 =  ⌈n⌉/⌊n⌋
    # n 大时这个比值必然贴近 1（相邻整数的相对差 ~1/n），所以 5% 这个门槛
    # 恰好只在「取哪个整数真的会影响结论」的地方才响 —— 不是按 n 的绝对大小拍的。
    if h_floor > 0 and h_ceil / h_floor > 1.05:
        warnings.append(
            f"n 只有 {n_exact:.4g}，⌊⌋={n_floor} 与 ⌈⌉={n_ceil} 给出的 h "
            f"差了 {(h_ceil / h_floor - 1) * 100:.1f}%（{h_floor:.4g} → {h_ceil:.4g}）—— "
            f"这个目标下取哪个整数是有感的，不是随便凑的。"
        )

    # h* 和 ⟨h⟩ 来自 P(h) 的高斯近似，小 n 时不可靠（h_rms 与 nl 是精确的，
    # 所以只有这两个 kind 要提醒）。判据复用 GAUSSIAN_N_MIN，不另立一个数。
    if kind in ("h_mp", "h_mean") and min(n_floor, n_ceil) < GAUSSIAN_N_MIN:
        warnings.append(
            f"{H_KIND_LABELS[kind]} 来自末端距分布的高斯近似，"
            f"而 n={min(n_floor, n_ceil)} < {GAUSSIAN_N_MIN} 时高斯近似不可靠 —— "
            f"反解出的 n 只能当量级看。均方末端距 h_rms = l√n 本身是精确的，"
            f"换 kind 为 h_rms 就没有这个问题。"
        )

    return SolveResult(
        kind=kind,
        kind_label=H_KIND_LABELS[kind],
        h=hv,
        l=l_f,
        n_exact=n_exact,
        n_floor=n_floor,
        n_ceil=n_ceil,
        h_floor=h_floor,
        h_ceil=h_ceil,
        warnings=warnings,
    )


# --- Kuhn 长度预置 -------------------------------------------------------
#
# 换算关系里**没有余地**的只有一条：持久长度与 Kuhn 长度 b = 2p（定义）。
# 至于各聚合物的 b 取多少，**文献里同一种材料能差一倍** —— 温度、立构规整度、
# 溶剂/熔体、以及作者给的到底是 p 还是 b，都会挪动结果。
#
# 所以这张表**刻意不装权威**：每条都带一个「可信度」和一句来源说明，
# 页面上还有一条固定提示。它的用途是把 l 填到正确的量级，不是替代文献。
# conf 字段：hard = 教科书级共识；derived = 由定义式从别的量推出；nominal = 常见引值、散布大。
KUHN_PRESETS = (
    {
        "name": "DNA（双链，B 型）",
        "b_nm": 100.0,
        "p_nm": 50.0,
        "conf": "hard",
        "note": "持久长度 ≈ 50 nm（约 150 bp），这是表里最硬的一条；b = 2p ≈ 100 nm。",
    },
    {
        "name": "聚乙烯 PE",
        "b_nm": 0.66,
        "p_nm": 0.33,
        "conf": "derived",
        "note": "由定义式 b = a√C∞ 推出：重复单元轮廓长度 a ≈ 0.254 nm（两个 CH2），"
                "C∞ ≈ 6.7，故 b ≈ 0.254×√6.7 ≈ 0.66 nm。文献常写 0.67 nm。",
    },
    {
        "name": "聚苯乙烯 PS",
        "b_nm": 1.8,
        "p_nm": 0.9,
        "conf": "nominal",
        "note": "常见引值 1.5～2.0 nm，侧基大、链更僵；不同来源差约 30%。",
    },
    {
        "name": "聚甲基丙烯酸甲酯 PMMA",
        "b_nm": 1.5,
        "p_nm": 0.75,
        "conf": "nominal",
        "note": "常见引值 1.0～1.8 nm；等规/无规差别也不小。",
    },
    {
        "name": "聚二甲基硅氧烷 PDMS",
        "b_nm": 1.5,
        "p_nm": 0.75,
        "conf": "nominal",
        "note": "主链 Si–O 很柔顺，但按 b 计常见引值仍在 1.5 nm 附近；有些来源给到 0.6 nm。",
    },
    {
        "name": "顺式聚异戊二烯（天然橡胶）",
        "b_nm": 1.8,
        "p_nm": 0.9,
        "conf": "nominal",
        "note": "常见引值 0.8～1.8 nm，是这张表里散布最大的一条。",
    },
    {
        "name": "聚氧化乙烯 PEO",
        "b_nm": 0.7,
        "p_nm": 0.35,
        "conf": "nominal",
        "note": "常见引值 0.5～0.8 nm，链柔顺。",
    },
    {
        "name": "蛋白质主链（无规卷曲）",
        "b_nm": 1.5,
        "p_nm": 0.75,
        "conf": "nominal",
        "note": "完全伸展约 0.36 nm/残基，无规卷曲下 b 常引 1.5 nm 一带；"
                "有二级结构时远不止。",
    },
)

KUHN_CONF_LABELS = {
    "hard": "教科书级共识",
    "derived": "由定义式推出",
    "nominal": "常见引值，散布大",
}


def kuhn_presets() -> list[dict]:
    """Kuhn 长度预置表，附可信度中文标签。给 /api 与页面下拉用。"""
    return [
        {**p, "conf_label": KUHN_CONF_LABELS[p["conf"]]}
        for p in KUHN_PRESETS
    ]
