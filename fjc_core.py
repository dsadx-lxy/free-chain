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
        return (
            f"键角固定为 {mp.theta_deg:g}°，再叠加内旋转势垒（平均余弦 ⟨cosφ⟩ = {mp.cos_phi:g}）。"
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


def model_catalog() -> list[dict]:
    """四个模型的目录：前端拿它建选择器、AI 助手拿它知道自己能算什么。

    和 KUHN_PRESETS 一样，**唯一的一份在这里**：/api/models 原样吐出去，
    前端不再抄第二份（抄了就会改一处忘一处，两处各说各话）。
    """
    out = []
    for key in MODEL_KEYS:
        mp = ModelParams.of(key)
        out.append({
            "key": key,
            "label": MODEL_LABELS[key],
            "formula": model_formula(mp),
            "note": model_note(mp),
            "params": {
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
            },
        })
    return out


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
