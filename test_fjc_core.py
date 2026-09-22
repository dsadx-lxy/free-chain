"""fjc_core 的单元测试。

核心思路：解析公式 vs scipy 数值积分。解析式是「声称」，数值积分是「独立算一遍」。
用 unittest 写，pytest 也能直接收集。

    D:\\anaconda3\\python.exe -m unittest test_fjc_core -v
"""

import math
import unittest

import numpy as np
from scipy.integrate import quad

import fjc_core as fjc


class TestAnalyticVsNumeric(unittest.TestCase):
    """解析式必须和数值积分对得上。"""

    def test_distribution_normalised(self):
        """∫₀^∞ P(h)dh = 1"""
        for n in (10, 100, 1000):
            beta = fjc.beta_of(n, 1.0)
            total, _ = quad(lambda h: float(fjc.P_of_h(h, beta)), 0.0, np.inf)
            self.assertAlmostEqual(total, 1.0, places=9, msg=f"n={n}")

    def test_second_moment_equals_nl2(self):
        """∫₀^∞ h²P(h)dh 必须等于解析的 ⟨h²⟩ = nl²"""
        for n, l in ((10, 1.0), (100, 1.0), (1000, 2.5), (50, 0.38)):
            beta = fjc.beta_of(n, l)
            numeric, _ = quad(lambda h: h * h * float(fjc.P_of_h(h, beta)), 0.0, np.inf)
            self.assertAlmostEqual(numeric, n * l * l, places=9,
                                   msg=f"n={n}, l={l}")

    def test_mean_end_to_end_matches(self):
        """⟨h⟩ 解析值 = l√(8n/(3π))，且与数值积分一致"""
        for n, l in ((10, 1.0), (100, 1.0), (1000, 0.5)):
            beta = fjc.beta_of(n, l)
            numeric, _ = quad(lambda h: h * float(fjc.P_of_h(h, beta)), 0.0, np.inf)
            analytic = l * math.sqrt(8.0 * n / (3.0 * math.pi))
            self.assertAlmostEqual(numeric, analytic, places=9, msg=f"n={n}")
            self.assertAlmostEqual(fjc.compute(n, l).h_mean, analytic, places=12)

    def test_most_probable_is_a_maximum(self):
        """h* 处 P(h) 的一阶导为 0，且确实是极大值"""
        n, l = 100, 1.0
        r = fjc.compute(n, l)
        eps = r.h_mp * 1e-6
        d = (float(fjc.P_of_h(r.h_mp + eps, r.beta))
             - float(fjc.P_of_h(r.h_mp - eps, r.beta))) / (2 * eps)
        self.assertAlmostEqual(d, 0.0, places=6)
        self.assertGreater(float(fjc.P_of_h(r.h_mp, r.beta)),
                           float(fjc.P_of_h(r.h_mp * 1.5, r.beta)))

    def test_sigma_consistent_with_moments(self):
        """σ² 必须等于 ⟨h²⟩ − (h*)²，即分布方差的定义"""
        for n, l in ((10, 1.0), (100, 2.0), (5000, 0.7)):
            r = fjc.compute(n, l)
            self.assertAlmostEqual(r.h2 - r.h_mp ** 2, r.sigma ** 2,
                                   places=9, msg=f"n={n}")


class TestKnownValues(unittest.TestCase):
    """对死数：n=100, l=1 时应为教科书值。"""

    def test_n100_l1(self):
        r = fjc.compute(100, 1.0)
        self.assertAlmostEqual(r.h_rms, 10.0, places=12)
        self.assertAlmostEqual(r.h2, 100.0, places=12)
        self.assertAlmostEqual(r.h_mp, math.sqrt(200.0 / 3.0), places=12)   # 8.16497
        self.assertAlmostEqual(r.h_mean, 9.2132, places=4)
        self.assertAlmostEqual(r.h_max, 100.0, places=12)
        self.assertAlmostEqual(r.Rg_rms, math.sqrt(100.0 / 6.0), places=12)
        self.assertAlmostEqual(r.Cn, 1.0, places=12)

    def test_sqrt_n_scaling(self):
        """核心结论：h 正比于 √n。n 翻 4 倍 → h 翻 1 倍。"""
        self.assertAlmostEqual(fjc.compute(100, 1.0).h_rms, 10.0, places=12)
        self.assertAlmostEqual(fjc.compute(400, 1.0).h_rms, 20.0, places=12)
        self.assertAlmostEqual(fjc.compute(25, 2.0).h_rms, 10.0, places=12)

    def test_ordering_of_characteristic_lengths(self):
        """h* < ⟨h⟩ < h_rms，这是该分布应有的序关系"""
        r = fjc.compute(100, 1.0)
        self.assertLess(r.h_mp, r.h_mean)
        self.assertLess(r.h_mean, r.h_rms)
        self.assertLess(r.h_rms, r.h_max)

    def test_scaling_collapse(self):
        """归一化后各 n 的曲线必须完全重合。

        以 x = h/h_rms 为自变量时，h_rms·P(h) = C·x²·e^(−1.5x²)，
        C = 4·(3/2)^1.5/√π ≈ 4.14624，与 n 和 l 都无关。
        这条恒等式成立说明 √n 标度律在整个分布上都对，不只是均方值。
        """
        const = 4.0 * 1.5 ** 1.5 / math.sqrt(math.pi)

        # 先确认这个常数本身是对的：∫₀^∞ C·x²e^(−1.5x²)dx 必须等于 1
        # （做代换 h = h_rms·x 后，归一化条件就是这个积分）
        norm, _ = quad(lambda x: const * x * x * math.exp(-1.5 * x * x), 0.0, np.inf)
        self.assertAlmostEqual(norm, 1.0, places=9)

        for n in (10, 100, 1000, 100000):
            r = fjc.compute(n, 1.0)
            for x in (0.3, 0.8, 1.0, 1.7, 2.5):
                h = x * r.h_rms
                lhs = r.h_rms * float(fjc.P_of_h(h, r.beta))
                rhs = const * x * x * math.exp(-1.5 * x * x)
                self.assertAlmostEqual(lhs, rhs, places=8, msg=f"n={n}, x={x}")


class TestValidation(unittest.TestCase):
    """非法输入必须被拦下，且信息是中文、能直接给用户看。"""

    def test_rejects_non_integer(self):
        with self.assertRaises(fjc.FJCInputError):
            fjc.compute(2.5, 1.0)

    def test_rejects_zero_and_negative(self):
        with self.assertRaises(fjc.FJCInputError):
            fjc.compute(0, 1.0)
        with self.assertRaises(fjc.FJCInputError):
            fjc.compute(-5, 1.0)

    def test_rejects_bad_l(self):
        with self.assertRaises(fjc.FJCInputError):
            fjc.compute(100, 0.0)
        with self.assertRaises(fjc.FJCInputError):
            fjc.compute(100, -1.0)

    def test_rejects_over_limit(self):
        with self.assertRaises(fjc.FJCInputError):
            fjc.compute(fjc.N_MAX + 1, 1.0)

    def test_rejects_garbage(self):
        for bad in (None, "abc", float("nan"), float("inf")):
            with self.assertRaises(fjc.FJCInputError):
                fjc.compute(bad, 1.0)

    def test_accepts_numeric_string(self):
        """表单传来的是字符串，'100' 应当能用。"""
        self.assertEqual(fjc.compute("100", "1.0").n, 100)

    def test_error_message_mentions_the_problem(self):
        try:
            fjc.compute(0, 1.0)
        except fjc.FJCInputError as e:
            self.assertIn("n", str(e))
        else:
            self.fail("应当抛 FJCInputError")


class TestWarnings(unittest.TestCase):
    """小 n 时高斯近似不成立，必须提示；大 n 时不应误报。"""

    def test_small_n_warns(self):
        self.assertTrue(fjc.compute(5, 1.0).warnings)
        self.assertTrue(fjc.compute(1, 1.0).warnings)

    def test_large_n_silent(self):
        self.assertEqual(fjc.compute(100, 1.0).warnings, [])
        self.assertEqual(fjc.compute(1000000, 1.0).warnings, [])

    def test_truncation_warning_at_boundary(self):
        """n=10 时 nl=10 仍小于 h*+5σ≈11.7，曲线被截断，应当提示。"""
        self.assertTrue(any("截断" in w for w in fjc.compute(10, 1.0).warnings))
        self.assertFalse(any("截断" in w for w in fjc.compute(100, 1.0).warnings))


class TestEdgeCases(unittest.TestCase):

    def test_n1_degrades_gracefully(self):
        """n=1 时 ⟨h²⟩=l² 仍精确，曲线也必须画得出来（不能全 0 或 NaN）。"""
        r = fjc.compute(1, 1.0)
        self.assertAlmostEqual(r.h_rms, 1.0, places=12)
        self.assertTrue(np.all(np.isfinite(r.curve_P)))
        self.assertGreater(float(r.curve_P.max()), 0.0)

    def test_curve_shape_and_finiteness(self):
        for n in (1, 2, 10, 100, 10**7):
            r = fjc.compute(n, 1.0)
            self.assertEqual(len(r.curve_h), fjc.CURVE_POINTS)
            self.assertEqual(len(r.curve_P), fjc.CURVE_POINTS)
            self.assertTrue(np.all(np.isfinite(r.curve_P)), msg=f"n={n}")
            # 曲线区间右端不得超过全伸展长度
            self.assertLessEqual(float(r.curve_h[-1]), r.h_max + 1e-9, msg=f"n={n}")

    def test_curve_starts_and_ends_near_zero(self):
        """h=0 处 P=0；右端取在 5σ 外，应当已经衰减到很小。"""
        r = fjc.compute(100, 1.0)
        self.assertEqual(float(r.curve_P[0]), 0.0)
        self.assertLess(float(r.curve_P[-1]), 0.02 * float(r.curve_P.max()))

    def test_extreme_n_stays_finite(self):
        r = fjc.compute(10**7, 1.0)
        self.assertTrue(math.isfinite(r.h_rms))
        self.assertTrue(math.isfinite(r.beta))
        self.assertAlmostEqual(r.h_rms, math.sqrt(10**7), places=6)

    def test_units_scale_linearly(self):
        """长度类输出对 l 线性；无量纲量不变。"""
        a, b = fjc.compute(100, 1.0), fjc.compute(100, 3.0)
        for attr in ("h_rms", "h_mp", "h_mean", "h_max", "Rg_rms", "sigma"):
            self.assertAlmostEqual(getattr(b, attr), 3.0 * getattr(a, attr),
                                   places=9, msg=attr)
        self.assertAlmostEqual(a.Cn, b.Cn, places=12)


class TestFreeRotatingChainConsistency(unittest.TestCase):
    """交叉验证：自由旋转链在 θ=90° 时应退化为自由连接链。

    这是对 FJC 公式的独立体检 —— 两条不同的推导路径必须给出同一个 ⟨h²⟩。
    """

    def test_theta_90_equals_fjc(self):
        for n, l in ((10, 1.0), (100, 1.0), (250, 2.2)):
            self.assertAlmostEqual(fjc.freerotating_h2(n, l, 90.0),
                                   fjc.compute(n, l).h2, places=9, msg=f"n={n}")

    def test_frc_angle_dependence(self):
        """θ 是相邻键矢量的夹角，方向不能弄反。

        θ<90° → 锯齿更紧、链更紧凑，⟨h²⟩ 小于 FJC；
        θ>90° → 链更伸展，⟨h²⟩ 大于 FJC。
        （聚乙烯 θ=109.5° 给出 Cn=(1+1/3)/(1−1/3)=2，正是教科书值。）
        """
        n, l = 100, 1.0
        fjc_h2 = fjc.compute(n, l).h2
        for theta in (60.0, 70.0, 80.0):
            self.assertLess(fjc.freerotating_h2(n, l, theta), fjc_h2)
        for theta in (100.0, 120.0, 109.5):
            self.assertGreater(fjc.freerotating_h2(n, l, theta), fjc_h2)

    def test_polyethylene_characteristic_ratio(self):
        """聚乙烯键角 arccos(−1/3)≈109.47° 的自由旋转链特征比恰好为 2。

        (1−cosθ)/(1+cosθ) 代入 cosθ=−1/3 得 (4/3)/(2/3)=2。这是教科书标准值。
        """
        n, l = 100, 1.0
        theta = math.degrees(math.acos(-1.0 / 3.0))
        self.assertAlmostEqual(fjc.freerotating_h2(n, l, theta) / (n * l * l),
                               2.0, places=12)

    def test_free_rotation_limit(self):
        """θ→90° 的邻域里 FRC 连续趋近 FJC。

        d/dθ[(1−cosθ)/(1+cosθ)] 在 90° 处等于 −2，所以 0.001° 的偏差
        约为 2·(0.001·π/180) ≈ 3.49e-5 —— 容差按这个量级取。
        """
        n, l = 100, 1.0
        fjc_h2 = fjc.compute(n, l).h2
        self.assertLess(abs(fjc.freerotating_h2(n, l, 90.001) - fjc_h2) / fjc_h2, 1e-4)


class TestBatch(unittest.TestCase):

    def test_compute_many(self):
        rs = fjc.compute_many([10, 100, 1000], 1.0)
        self.assertEqual([r.n for r in rs], [10, 100, 1000])
        self.assertAlmostEqual(rs[1].h_rms, 10.0, places=12)

    def test_compute_many_rejects_empty(self):
        with self.assertRaises(fjc.FJCInputError):
            fjc.compute_many([], 1.0)

    def test_compute_many_propagates_error(self):
        with self.assertRaises(fjc.FJCInputError):
            fjc.compute_many([10, 0, 100], 1.0)


class TestSerialisation(unittest.TestCase):

    def test_to_dict_is_json_serialisable(self):
        import json
        d = fjc.compute(100, 1.0).to_dict()
        s = json.dumps(d)          # numpy 类型没转干净的话这里会炸
        back = json.loads(s)
        self.assertEqual(back["n"], 100)
        self.assertEqual(len(back["curve"]["h"]), fjc.CURVE_POINTS)
        self.assertEqual(len(back["curve"]["P"]), fjc.CURVE_POINTS)
        self.assertEqual(back["warnings"], [])


class TestRandomChain(unittest.TestCase):
    """单链构象模拟。这里验的是「采样器有没有忠实实现 FJC」，不是分布公式。"""

    def test_unit_step_length(self):
        """每一步的长度必须精确等于 l（含首项为原点）"""
        for l in (1.0, 2.5, 0.38):
            p = fjc.random_chain(37, l, seed=7).points[0]
            self.assertEqual(p.shape, (38, 3))
            np.testing.assert_allclose(p[0], np.zeros(3))
            steps = np.linalg.norm(np.diff(p, axis=0), axis=1)
            np.testing.assert_allclose(steps, l, rtol=0, atol=1e-12)

    def test_directions_uniform_on_sphere(self):
        """方向必须在单位球面上**均匀**分布。

        这条专抓「θ、φ 都均匀」那个经典错误：那样方向会往两极堆，
        ⟨u_z²⟩ = 1/2 而不是 1/3，链会被系统性拉长/压扁。
        20 万个样本下，均匀采样的标准误约 1/√(3N) ≈ 0.0013，容差 0.01 很宽裕。
        """
        n, chains = 2000, 100
        p = fjc.random_chain(n, 1.0, seed=2024, chains=chains,
                             max_chains=None, points_max=None).points
        u = np.diff(p, axis=1).reshape(-1, 3)

        np.testing.assert_allclose(np.linalg.norm(u, axis=1), 1.0, atol=1e-12)
        for k, axis in enumerate("xyz"):
            self.assertAlmostEqual(float(np.mean(u[:, k])), 0.0, delta=0.01,
                                   msg=f"⟨u_{axis}⟩ 应≈0")
            self.assertAlmostEqual(float(np.mean(u[:, k] ** 2)), 1.0 / 3.0, delta=0.01,
                                   msg=f"⟨u_{axis}²⟩ 应≈1/3")

    def test_ensemble_recovers_nl2(self):
        """系综平均必须收敛到 ⟨R²⟩ = n·l² —— 采样器是否忠实于模型的硬证据。

        单条链的 R² 离 nl² 可以很远，2000 条的平均才收敛。
        Var(R²) = (2/3)(nl²)²（高斯链），2000 条的相对标准误 ≈ 1.8%，
        容差取 5%；seed 写死以保证确定性 —— 否则会偶发失败。

        这里显式解除绘制预算：5 条链的上限是给画布用的，统计验证必须超出它。
        """
        n, l = 50, 1.0
        r = fjc.random_chain(n, l, seed=99991, chains=2000,
                             max_chains=None, points_max=None)
        self.assertEqual(r.points.shape, (2000, n + 1, 3))
        self.assertLess(abs(r.R2_mean - r.R2_theory) / r.R2_theory, 0.05)

    def test_ensemble_mean_length_grows_as_sqrt_n(self):
        """⟨R²⟩/n 应近似不随 n 变（√n 标度的另一半）"""
        vals = []
        for n in (25, 100, 400):
            vals.append(fjc.random_chain(n, 1.0, seed=4242, chains=800,
                                         max_chains=None, points_max=None).R2_mean / n)
        spread = (max(vals) - min(vals)) / (sum(vals) / len(vals))
        self.assertLess(spread, 0.15, msg=f"⟨R²⟩/n 随 n 漂移过大：{vals}")

    def test_seed_reproducible(self):
        """同种子逐点一致；不同种子不同"""
        a = fjc.random_chain(64, 1.0, seed=12345).points
        b = fjc.random_chain(64, 1.0, seed=12345).points
        c = fjc.random_chain(64, 1.0, seed=12346).points
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))

    def test_seed_none_is_reported_and_usable(self):
        """不给种子时要回传一个可用的种子，用它必须能复现同一条链"""
        r = fjc.random_chain(32, 1.0, seed=None)
        self.assertIsInstance(r.seed, int)
        self.assertGreaterEqual(r.seed, 0)
        again = fjc.random_chain(32, 1.0, seed=r.seed)
        np.testing.assert_array_equal(r.points, again.points)

    def test_multichain_shapes(self):
        r = fjc.random_chain(7, 1.0, seed=1, chains=3)
        self.assertEqual(r.points.shape, (3, 8, 3))
        self.assertEqual(r.R.shape, (3, 3))
        self.assertEqual(r.R_mag.shape, (3,))
        # 各条链必须互相独立
        self.assertFalse(np.array_equal(r.points[0], r.points[1]))

    def test_R2_mean_consistent_with_R_mag(self):
        r = fjc.random_chain(100, 1.0, seed=5, chains=4)
        self.assertAlmostEqual(r.R2_mean, float(np.mean(r.R_mag ** 2)), places=12)
        # R_mag 必须就是末顶点到原点的距离
        np.testing.assert_allclose(
            r.R_mag, np.linalg.norm(r.points[:, -1, :], axis=1), atol=1e-12
        )

    def test_R2_theory_matches_compute(self):
        """新代码必须钉在旧内核上：理论值就是 compute() 的 ⟨h²⟩"""
        for n in (1, 10, 1000):
            r = fjc.random_chain(n, 1.0, seed=3)
            self.assertEqual(r.R2_theory, fjc.compute(n, 1.0).h2)
            self.assertEqual(r.h_rms, fjc.compute(n, 1.0).h_rms)

    def test_single_segment(self):
        """n=1 退化成一条线段：R 的长度精确为 l"""
        r = fjc.random_chain(1, 2.5, seed=8)
        np.testing.assert_allclose(r.points[0][0], np.zeros(3))
        self.assertAlmostEqual(float(r.R_mag[0]), 2.5, places=12)

    def test_scale_with_l(self):
        """整体尺度随 l 线性缩放（同种子下逐点成比例）"""
        a = fjc.random_chain(50, 1.0, seed=77).points
        b = fjc.random_chain(50, 3.0, seed=77).points
        np.testing.assert_allclose(b, a * 3.0, atol=1e-12)
        self.assertAlmostEqual(
            fjc.random_chain(50, 3.0, seed=77).R2_mean,
            fjc.random_chain(50, 1.0, seed=77).R2_mean * 9.0, places=9,
        )

    def test_dense_warning(self):
        """n 很大时提示画出来看不清；正常 n 不提示"""
        self.assertEqual(fjc.random_chain(1000, 1.0, seed=1).warnings, [])
        self.assertTrue(fjc.random_chain(12000, 1.0, seed=1).warnings)

    def test_chain_validation(self):
        """非法输入一律抛 FJCInputError"""
        bad = [
            dict(n=0), dict(n=2.5), dict(n=fjc.CHAIN_N_MAX + 1),
            dict(n=100, l=0), dict(n=100, l=-1),
            dict(n=100, chains=0), dict(n=100, chains=fjc.MAX_CHAINS + 1),
            dict(n=100, seed=-1),
            dict(n=100, seed=1.5),
            # 点预算：n 与 chains 各自合法，乘起来越界
            dict(n=fjc.CHAIN_POINTS_MAX // 3, chains=3),
        ]
        for kw in bad:
            with self.subTest(**kw):
                args = {"n": kw.pop("n"), "l": kw.pop("l", 1.0)}
                with self.assertRaises(fjc.FJCInputError):
                    fjc.random_chain(**args, **kw)

    def test_R2_rel_se(self):
        """⟨R²⟩ 的相对标准误 = √(2/3)/√k，用来判断偏差是否离谱。

        这条不是装饰：页面上「偏差 −20%」到底算不算收敛，全靠它来定性。
        所以它必须随 k 按 1/√k 收缩，且 k=1 时等于 √(2/3)。
        """
        one = fjc.random_chain(10, 1.0, seed=1, chains=1)
        self.assertAlmostEqual(one.R2_rel_se, math.sqrt(2.0 / 3.0), places=12)

        big = dict(max_chains=None, points_max=None)
        se5 = fjc.random_chain(10, 1.0, seed=1, chains=5, **big).R2_rel_se
        se100 = fjc.random_chain(10, 1.0, seed=1, chains=100, **big).R2_rel_se
        self.assertAlmostEqual(se5, math.sqrt(2.0 / 3.0) / math.sqrt(5), places=12)
        self.assertAlmostEqual(se100 / se5, math.sqrt(5.0 / 100.0), places=12)

    def test_ensemble_deviation_within_expected_scatter(self):
        """实测偏差应当落在 √(2/3)/√k 的量级内 —— 否则 rel_se 就是错的"""
        for k in (10, 50, 200):
            r = fjc.random_chain(40, 1.0, seed=31337 + k, chains=k,
                                 max_chains=None, points_max=None)
            dev = abs(r.R2_mean - r.R2_theory) / r.R2_theory
            self.assertLess(dev, 3 * r.R2_rel_se, msg=f"k={k} 偏差 {dev:.3f}")

    def test_to_dict_is_json_serialisable(self):
        """扁平顶点数组 + 限位小数，numpy 类型不能漏出去"""
        import json
        r = fjc.random_chain(30, 1.0, seed=11, chains=2)
        d = json.loads(json.dumps(r.to_dict()))
        self.assertEqual(d["n"], 30)
        self.assertEqual(d["seed"], 11)
        self.assertEqual(len(d["points"]), 2)
        self.assertEqual(len(d["points"][0]), 31 * 3)
        self.assertEqual(len(d["R"]), 2)
        self.assertEqual(len(d["R_mag"]), 2)
        self.assertAlmostEqual(d["R2_rel_se"], math.sqrt(2.0 / 3.0) / math.sqrt(2),
                               places=12)

    def test_seed_is_visible_in_payload(self):
        """回传的 seed 必须是实际用的那个，否则前端复现不了"""
        import json
        r = fjc.random_chain(10, 1.0, seed=None, chains=1)
        self.assertEqual(json.loads(json.dumps(r.to_dict()))["seed"], r.seed)


class TestSweep(unittest.TestCase):
    """链长扫描：⟨R²⟩ 对 n 的标度关系。这里验的是「⟨R²⟩ = n·l² 是标度律」。"""

    def test_slope_is_one(self):
        """log⟨R²⟩ 对 log n 的拟合斜率必须 ≈ 1 —— 这是核心结论的直接检验。

        斜率的标准误 ≈ rel_se/√Σ(ln n − ln n̄)²。M=2000 时 rel_se=1.83%，
        六个对数等距点的分母约 4.83，故斜率标准误约 0.38%。容差 0.02 ≈ 5σ；
        seed 写死以保证确定性，不能写成随机种子否则会偶发失败。
        """
        r = fjc.sweep([10, 30, 100, 300, 1000, 3000], 1.0, seed=2026, chains=2000)
        self.assertAlmostEqual(r.slope, 1.0, delta=0.02, msg=f"斜率 {r.slope}")
        self.assertGreater(r.fit_r2, 0.999)

    def test_Cn_stays_near_one_across_the_sweep(self):
        """每个 n 的 ⟨R²⟩/(n·l²) 都该落在它们的固有涨落范围内。

        这条抓的是「标度律只在某一段成立」这类错误：如果采样器在各 n 上
        表现不一致（比如 n 大时方向分布走偏），Cn 会随 n 系统性漂移。
        """
        r = fjc.sweep([10, 30, 100, 300, 1000, 3000], 1.0, seed=777, chains=2000)
        for n_i, cn in zip(r.ns, r.Cn_sim):
            self.assertAlmostEqual(cn, 1.0, delta=3 * r.rel_se, msg=f"n={n_i} 的 Cn={cn}")

    def test_h_rms_sim_is_sqrt_of_R2(self):
        r = fjc.sweep([100, 1000], 2.5, seed=5, chains=500)
        for r2, h in zip(r.R2_mean, r.h_rms_sim):
            self.assertAlmostEqual(h, math.sqrt(r2), places=12)

    def test_Cn_matches_its_definition(self):
        r = fjc.sweep([10, 100], 1.0, seed=5, chains=500)
        for r2, th, cn in zip(r.R2_mean, r.R2_theory, r.Cn_sim):
            self.assertAlmostEqual(cn, r2 / th, places=12)

    def test_theory_matches_compute(self):
        """理论量必须钉在解析内核上，不是第二套公式。"""
        for n in (10, 100, 1000):
            r = fjc.sweep([n], 1.0, seed=3, chains=500)
            self.assertEqual(r.R2_theory[0], fjc.compute(n, 1.0).h2)
            self.assertEqual(r.h_rms[0], fjc.compute(n, 1.0).h_rms)

    def test_rel_se_formula(self):
        """√(2/3)/√M，与 n 无关 —— 页面上判断「偏多少算正常」全靠它。"""
        for M in (100, 1000, 10000):
            r = fjc.sweep([100], 1.0, seed=1, chains=M)
            self.assertAlmostEqual(r.rel_se, math.sqrt(2.0 / 3.0) / math.sqrt(M),
                                   places=12)

    def test_endpoints_match_random_chain_on_the_same_stream(self):
        """同一个随机流上，两条路径必须**逐位相同**。

        random_chain 用 cumsum 取最后一个顶点，sweep 直接把步进矢量求和 ——
        这是两条独立实现。sweep 给每个 n 派生子种子，所以不能直接拿 seed 对齐，
        这里在 _sweep_endpoints 这一层对齐（chains×n ≤ 200 万，不会分块）。
        """
        n, M, seed = 100, 1000, 12345
        r = fjc._sweep_endpoints(np.random.default_rng(seed), n, M)
        c = fjc.random_chain(n, 1.0, seed=seed, chains=M,
                             max_chains=None, points_max=None)
        np.testing.assert_allclose(r, c.R, atol=1e-12)

    def test_sweep_statistics_agree_with_random_chain(self):
        """换一条独立路径复算同一个 n，⟨R²⟩ 必须落在彼此的涨落范围内。"""
        n, M = 100, 2000
        s = fjc.sweep([n], 1.0, seed=2024, chains=M)
        c = fjc.random_chain(n, 1.0, seed=99, chains=M,
                             max_chains=None, points_max=None)
        both = math.sqrt(2.0 / 3.0) / math.sqrt(M)
        self.assertAlmostEqual(s.Cn_sim[0], c.R2_mean / (n * 1.0), delta=5 * both)

    def test_reproducible(self):
        a = fjc.sweep([10, 100, 1000], 1.0, seed=42, chains=300)
        b = fjc.sweep([10, 100, 1000], 1.0, seed=42, chains=300)
        self.assertEqual(a.R2_mean, b.R2_mean)
        c = fjc.sweep([10, 100, 1000], 1.0, seed=43, chains=300)
        self.assertNotEqual(a.R2_mean, c.R2_mean)

    def test_chunked_path_is_deterministic(self):
        """n 大到触发分块时结果仍必须可复现（分块大小只由 n 决定）。"""
        a = fjc.sweep([20000], 1.0, seed=9, chains=300)
        b = fjc.sweep([20000], 1.0, seed=9, chains=300)
        self.assertEqual(a.R2_mean, b.R2_mean)
        self.assertAlmostEqual(a.Cn_sim[0], 1.0, delta=3 * a.rel_se)

    def test_seed_is_derived_per_position(self):
        """每个 n 用 SeedSequence 派生的独立子种子，而第 i 个子种子只取决于 i。

        这带来一条好性质：同一 seed 下，序列里第 i 位的 n 值相同就必得同一结果，
        与序列长度、其它位置放什么无关 —— 改一个扫描点不会扰动别的点。
        """
        a = fjc.sweep([10, 100, 1000], 1.0, seed=42, chains=300)
        b = fjc.sweep([10, 500], 1.0, seed=42, chains=300)
        self.assertEqual(a.R2_mean[0], b.R2_mean[0])        # 第 0 位都是 n=10

        c = fjc.sweep([500, 10], 1.0, seed=42, chains=300)  # n=10 挪到第 1 位
        self.assertNotEqual(a.R2_mean[0], c.R2_mean[1])     # 换了流，结果不同

    def test_single_point_has_no_slope(self):
        r = fjc.sweep([100], 1.0, seed=1, chains=500)
        self.assertIsNone(r.slope)
        self.assertIsNone(r.fit_r2)
        self.assertIsNone(r.slope_se)
        self.assertTrue(any("斜率" in w for w in r.warnings))

    def test_small_sample_warns(self):
        # 三个点：两点会额外触发「R² 恒为 1」那条，这里只想验样本量这一条
        self.assertFalse(fjc.sweep([100, 300, 1000], 1.0, seed=1, chains=2000).warnings)
        self.assertTrue(any("标准误" in w
                            for w in fjc.sweep([100, 300, 1000], 1.0, seed=1,
                                               chains=50).warnings))

    def test_slope_se_is_smaller_when_n_is_spread_out(self):
        """斜率的标准误是 rel_se/√Σ(ln n − ln n̄)² —— 点的个数和铺开程度都算在里面。

        这条把「拿 3·rel_se 当门槛」这个错钉死：对 10…3000 这种铺开的序列，斜率的
        标准误比 rel_se 小一个数量级，用 rel_se 判等于放水（5σ 的真偏差也放过去）；
        对 300 和 400 这种挤在一起的又反过来大五倍，那时它会冤枉好数据。
        """
        spread = fjc.sweep([10, 30, 100, 300, 1000, 3000], 1.0, seed=1, chains=2000)
        clustered = fjc.sweep([300, 400], 1.0, seed=1, chains=2000)

        for r in (spread, clustered):
            ln_n = [math.log(n) for n in r.ns]
            mean = sum(ln_n) / len(ln_n)
            sxx = sum((v - mean) ** 2 for v in ln_n)
            self.assertAlmostEqual(r.slope_se, r.rel_se / math.sqrt(sxx), places=12)

        self.assertLess(spread.slope_se, spread.rel_se / 4)
        self.assertGreater(clustered.slope_se, clustered.rel_se * 4)

    def test_clustered_n_does_not_trip_the_slope_warning(self):
        """n 挤在一起时斜率本来就估不准，别把这当成「不该发生」。

        实测 n=300/400、M=2000 时斜率 0.9149，看着离 1 很远，但它的标准误是 0.0898
        —— 偏了不到一个 σ。按 rel_se 判（3σ = 0.0548）会误报，让用户去追一个不存在的问题。
        """
        r = fjc.sweep([300, 400], 1.0, seed=2026, chains=2000)
        self.assertLess(abs(r.slope - 1.0), r.slope_se)
        # 「这不该发生」只出现在斜率那条警告里；「只有两个 n」那条会在括注里提到
        # 「三倍标准误」，所以不能拿那个词当筛子
        self.assertFalse(any("这不该发生" in w for w in r.warnings))

    def test_two_points_admit_r2_is_uninformative(self):
        """两点必然共线，R² 恒为 1 —— 页面上会把它显示成「拟合很好」，得先说明白。"""
        r = fjc.sweep([100, 1000], 1.0, seed=1, chains=2000)
        self.assertEqual(r.fit_r2, 1.0)
        self.assertTrue(any("决定系数" in w for w in r.warnings))

    def test_small_n_warns_about_the_scatter_formula_only(self):
        """小 n 要提示 √(2/3)/√M 只是量级参考，但不能说 ⟨R²⟩=n·l² 不成立。"""
        ws = fjc.sweep([3, 100], 1.0, seed=1, chains=2000).warnings
        self.assertTrue(any("量级参考" in w for w in ws))
        self.assertFalse(any("高斯近似不可靠" in w for w in ws))

    def test_accepts_string_and_single_number(self):
        for raw, want in (("10, 30, 100", [10, 30, 100]),
                          ("10 30 100", [10, 30, 100]),
                          ("10，30、100", [10, 30, 100]),
                          (100, [100]),
                          ([10], [10])):
            with self.subTest(raw=raw):
                r = fjc.sweep(raw, 1.0, seed=1, chains=200)
                self.assertEqual(r.ns, want)

    def test_validation(self):
        bad = [
            (dict(ns=[]), "空列表"),
            (dict(ns=None), "None"),
            (dict(ns=[0, 10]), "n=0"),
            (dict(ns=[10, 10]), "重复"),
            (dict(ns=[10, 2.5]), "n 非整数"),
            (dict(ns=[10], chains=0), "chains=0"),
            (dict(ns=list(range(1, fjc.SWEEP_N_MAX + 2))), "扫描点过多"),
            (dict(ns=[10], chains=fjc.SWEEP_CHAINS_MAX + 1), "链数超限"),
            (dict(ns=[10, 1000], chains=100_000), "超出计算量预算"),
            (dict(ns=[10], seed=-1), "种子为负"),
        ]
        for kw, tag in bad:
            with self.subTest(tag=tag):
                kw.setdefault("l", 1.0)
                kw.setdefault("seed", 1)
                kw.setdefault("chains", 10)
                with self.assertRaises(fjc.FJCInputError):
                    fjc.sweep(**kw)

    def test_budgets_are_independent_and_can_be_disabled(self):
        """三个上限是资源约束不是物理约束：各自独立，都能传 None 解除。

        测试里用「把上限调小」而不是「真的超限」来触发，否则为了验一条
        7 秒的路径要把整个测试套拖慢一个量级。
        """
        with self.assertRaises(fjc.FJCInputError):
            fjc.sweep([1, 2, 3], 1.0, seed=1, chains=10, max_points=2)
        with self.assertRaises(fjc.FJCInputError):
            fjc.sweep([10], 1.0, seed=1, chains=6, max_chains=5)
        with self.assertRaises(fjc.FJCInputError):
            fjc.sweep([10], 1.0, seed=1, chains=10, work_max=50)

        # 只解除被卡住的那一个即可 —— 说明三者互不牵连
        r = fjc.sweep([10], 1.0, seed=1, chains=6, max_chains=None)
        self.assertEqual(r.chains, 6)

        # 全部解除后确实能超过默认上限（20 万条链 × n=10 = 200 万，跑得动）
        r = fjc.sweep([10], 1.0, seed=1, chains=200_000,
                      max_points=None, max_chains=None, work_max=None)
        self.assertEqual(r.chains, 200_000)
        self.assertAlmostEqual(r.Cn_sim[0], 1.0, delta=3 * r.rel_se)

    def test_to_dict_is_json_serialisable(self):
        import json
        r = fjc.sweep([10, 100], 1.0, seed=11, chains=200)
        d = json.loads(json.dumps(r.to_dict()))
        self.assertEqual(d["n"], [10, 100])
        self.assertEqual(len(d["R2_mean"]), 2)
        self.assertEqual(len(d["Cn_sim"]), 2)
        self.assertEqual(d["seed"], 11)
        self.assertIsInstance(d["slope"], float)
        self.assertIsInstance(d["slope_se"], float)
        self.assertLess(d["slope_se"], d["rel_se"])


class TestLangevin(unittest.TestCase):
    """L(x) = coth(x) − 1/x。

    两条独立的参照：
      · mpmath 50 位小数（绝对真值，装了才用，见 skipIf）
      · 级数的更多项（项目里手算的 Bernoulli 系数，与实现里截断的 4 项不同源）
    """

    # coth(x) − 1/x 的展开系数，x^(2k+1) 的系数 c_k = 2^(2k)B_{2k}/(2k)!
    #   1/3, −1/45, 2/945, −1/4725, 2/93555, …
    # 实现里只用到前 4 项，这里写 6 项，所以是**另一份**展开，不是照抄实现。
    SERIES = (1.0 / 3.0, -1.0 / 45.0, 2.0 / 945.0,
              -1.0 / 4725.0, 2.0 / 93555.0, -1.0 / 14175.0)

    @classmethod
    def _series_ref(cls, x: float) -> float:
        total, p = 0.0, x
        for k, c in enumerate(cls.SERIES):
            p = x ** (2 * k + 1) if k == 0 else p * x * x
            total += c * p
        return total

    def test_zero_is_exactly_zero(self):
        """L(0)：直接式是 ∞−∞，级数式是 0·(1/3)=0。必须给出 0，不是 nan。"""
        self.assertEqual(fjc.langevin(0.0), 0.0)
        self.assertNotEqual(fjc.langevin(0.0), float("nan"))

    def test_matches_extended_series_where_it_converges(self):
        """|x| ≤ 0.05 时 6 项级数早已收敛，与实现应当重合。

        上限定在 0.05 是因为**截断误差本身**：第 7 项是 O(x¹³)，
        x=0.5 时还有 3e-8，再往上这条参照自己就不够准了 ——
        大 x 的正确性交给下面的 mpmath 对照，别拿半吊子参照去验。
        """
        for x in (1e-9, 1e-5, 1e-3, 0.01, 0.02, 0.05):
            ref = self._series_ref(x)
            self.assertAlmostEqual(fjc.langevin(x), ref, delta=1e-13,
                                   msg=f"x={x}")

    def test_two_branches_agree_at_the_split(self):
        """x=0.01 两侧换公式，**不能在接缝处跳一下**。

        做法是**交叉**：接缝左边用直接式验、右边用级数验 ——
        也就是说实现走哪一边，都要对得上另一边的公式。
        """
        # 左边（实现走级数）：用 1/tanh − 1/x 这个不同源的写法验
        for x in (0.005, 0.009, 0.009999, 0.009999999):
            direct = 1.0 / math.tanh(x) - 1.0 / x
            self.assertLess(abs(fjc.langevin(x) - direct) / abs(direct), 1e-11,
                            msg=f"接缝左 x={x}")

        # 右边（实现走直接式）：用多两项的级数验
        for x in (0.010000001, 0.010001, 0.011, 0.02, 0.05):
            ref = self._series_ref(x)
            self.assertLess(abs(fjc.langevin(x) - ref) / abs(ref), 1e-11,
                            msg=f"接缝右 x={x}")

    def test_reference_implementation_matches_mpmath(self):
        """mpmath 50 位 —— 唯一一处不依赖「自己验自己」的对照。"""
        try:
            from mpmath import coth, mp
        except ImportError:  # pragma: no cover
            self.skipTest("没有 mpmath，退回级数/分支两条对照")
        mp.dps = 50
        for x in (1e-9, 1e-5, 1e-3, 0.009999, 0.01, 0.010001, 0.05,
                  0.1, 0.5, 1.0, 3.0, 10.0, 100.0):
            ref = coth(mp.mpf(repr(x))) - 1 / mp.mpf(repr(x))
            got = mp.mpf(repr(fjc.langevin(x)))
            rel = abs(got - ref) / abs(ref)
            self.assertLess(float(rel), 5e-12, msg=f"x={x}, rel={rel}")

    def test_odd_function(self):
        for x in (0.001, 0.1, 1.0, 7.0):
            self.assertAlmostEqual(fjc.langevin(-x), -fjc.langevin(x), places=15)

    def test_monotone_increasing_and_bounded(self):
        """0 ≤ L(x) < 1，且单调 —— 反函数二分法的全部前提就这两条。"""
        prev = -1.0
        for i in range(0, 400):
            x = i * 0.25
            v = fjc.langevin(x)
            self.assertGreaterEqual(v, 0.0, msg=f"x={x}")
            self.assertLess(v, 1.0, msg=f"x={x}")
            self.assertGreaterEqual(v, prev, msg=f"x={x} 非单调")
            prev = v

    def test_gaussian_limit_slope_is_one_third(self):
        """小力下 L(x) → x/3，代回去正是高斯链的 3k_BT⟨x⟩/(nl²) = f。

        用差分取斜率：L 在 0 附近的导数必须是 1/3。这条钉的是
        「FJC 在小力下退化成高斯链」这个物理结论，不是实现细节。
        """
        eps = 1e-6
        slope = (fjc.langevin(eps) - fjc.langevin(-eps)) / (2 * eps)
        self.assertAlmostEqual(slope, 1.0 / 3.0, places=9)

        for x in (1e-8, 1e-6, 1e-4):
            self.assertAlmostEqual(fjc.langevin(x), x / 3.0,
                                   delta=abs(x / 3.0) * 1e-9, msg=f"x={x}")

    def test_known_value_at_one(self):
        """L(1) = coth(1) − 1，用 cosh/sinh 换一个写法算，和实现的 1/tanh 不同源。"""
        expect = math.cosh(1.0) / math.sinh(1.0) - 1.0
        self.assertAlmostEqual(fjc.langevin(1.0), expect, places=15)
        self.assertAlmostEqual(fjc.langevin(1.0), 0.3130352854993315, places=15)

    def test_asymptotics(self):
        """大 x 下 L(x) = 1 − 1/x + 2e^(−2x)/(1 − e^(−2x))。

        只取首项 1−1/x 在 x=10 处还差 4.1e-9（正是 2e^(−2x)）——
        拿 `places=14` 去卡 1−1/x 是**我把展开式记漏了一项**，不是实现不对。
        所以断言写成带头两阶修正的形式，余项给 5e^(−4x) 的界。
        """
        for x in (5.0, 6.0, 8.0):
            approx = 1.0 - 1.0 / x + 2.0 * math.exp(-2.0 * x)
            self.assertLess(abs(fjc.langevin(x) - approx),
                            5.0 * math.exp(-4.0 * x), msg=f"x={x}")

        # 再往上浮点自己就跟不动余项了（x=10 时 2e^{−20}=4.1e−9 还量得出，
        # x=20 时 1.7e−17 已经小于 1 附近的 ulp），只验主导项
        self.assertAlmostEqual(fjc.langevin(10.0) + 0.1 - 1.0,
                               2.0 * math.exp(-20.0), places=15)

    def test_rejects_non_finite(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(fjc.FJCInputError):
                fjc.langevin(bad)

    def test_not_a_numpy_scalar_sinkhole(self):
        """np.float64 也得能进（前端从曲线里取点喂回来会是这个类型）。"""
        v = fjc.langevin(np.float64(1.5))
        self.assertIsInstance(v, float)
        expect = math.cosh(1.5) / math.sinh(1.5) - 1.0 / 1.5
        self.assertAlmostEqual(v, expect, places=15)
        self.assertAlmostEqual(v, 0.4381247263158453, places=15)


class TestInverseLangevin(unittest.TestCase):
    """反解没有解析解，这里验的是二分法解得对不对。"""

    def test_zero(self):
        self.assertEqual(fjc.inverse_langevin(0.0), 0.0)

    def test_round_trip_lambda(self):
        """L(inv(λ)) 必须把 λ 还回来 —— 这是反解存在的意义。"""
        for lam in (1e-12, 1e-8, 1e-4, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9,
                    0.95, 0.99, 0.999, 1 - 1e-6, 1 - 1e-9, fjc.LAMBDA_MAX):
            x = fjc.inverse_langevin(lam)
            back = fjc.langevin(x)
            # 小 λ 处用绝对误差（相对误差会把 1e-12 放大），大 λ 处用相对
            tol = max(abs(lam) * 1e-9, 1e-15)
            self.assertLess(abs(back - lam), tol, msg=f"λ={lam}")

    def test_round_trip_x(self):
        """inv(L(x)) 必须把 x 还回来。"""
        for x in (1e-8, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0):
            lam = fjc.langevin(x)
            self.assertLess(abs(fjc.inverse_langevin(lam) - x), x * 1e-9,
                            msg=f"x={x}")

    def test_stays_inside_the_proved_bracket(self):
        """上界 hi = 1/(1−λ) 是**证出来的**，真解必须在里面。

        依据：L(x) > 1 − 1/x（因为 coth x − 1 = 2/(e^{2x}−1) > 0），
        所以 L(1/(1−λ)) > λ。这条断言就是那句话的可执行版本 ——
        改了二分上界而证没了，这里会先红。
        """
        for lam in (1e-6, 0.01, 0.1, 0.5, 0.9, 0.99, 1 - 1e-6):
            x = fjc.inverse_langevin(lam)
            hi = 1.0 / (1.0 - lam)
            self.assertLessEqual(x, hi, msg=f"λ={lam}")
            # 严格证明里是 L(hi) > λ，浮点下允许擦边
            self.assertGreaterEqual(fjc.langevin(hi), lam - 1e-15, msg=f"λ={lam}")

    def test_monotone_in_lam(self):
        prev = -1.0
        for i in range(1, 400):
            lam = i / 401.0
            v = fjc.inverse_langevin(lam)
            self.assertGreaterEqual(v, prev, msg=f"λ={lam} 非单调")
            prev = v

    def test_small_lam_is_three_lambda(self):
        """小 λ 下 L(x)≈x/3 ⟹ inv(λ) ≈ 3λ，即高斯链的 f = 3k_BT⟨x⟩/(nl²)。"""
        for lam in (1e-8, 1e-6, 1e-4, 1e-3):
            self.assertAlmostEqual(fjc.inverse_langevin(lam), 3 * lam,
                                   delta=3 * lam * 1e-6, msg=f"λ={lam}")

    def test_known_value_half_extension(self):
        """拉到一半伸长需要 x ≈ 1.7968（教科书里 Langevin 反函数的常见锚点）。"""
        x = fjc.inverse_langevin(0.5)
        self.assertAlmostEqual(x, 1.796755985, places=6)
        self.assertAlmostEqual(fjc.langevin(x), 0.5, places=14)

    def test_rejects_out_of_range(self):
        for bad in (-0.1, -1e-30, 1.0, 1.5, 100.0):
            with self.assertRaises(fjc.FJCInputError,
                                   msg=f"λ={bad} 应当被拒"):
                fjc.inverse_langevin(bad)
        with self.assertRaises(fjc.FJCInputError):
            fjc.inverse_langevin(float("nan"))
        with self.assertRaises(fjc.FJCInputError):
            fjc.inverse_langevin(float("inf"))

    def test_lambda_max_is_accepted(self):
        """LAMBDA_MAX 是留给可用区间的常量，自己得能解。"""
        x = fjc.inverse_langevin(fjc.LAMBDA_MAX)
        self.assertTrue(math.isfinite(x))
        self.assertGreater(x, 1e11)


class TestForceExtension(unittest.TestCase):
    """力–伸长曲线。核心性质：**它与 n、l 都无关**。"""

    def test_shape_and_monotone(self):
        r = fjc.force_extension(100, 1.0, 10.0)
        d = r.to_dict()
        self.assertEqual(len(d["curve"]["x"]), fjc.FORCE_CURVE_POINTS)
        self.assertEqual(len(d["curve"]["lambda"]), fjc.FORCE_CURVE_POINTS)
        self.assertEqual(d["curve"]["x"][0], 0.0)
        self.assertEqual(d["curve"]["lambda"][0], 0.0)
        self.assertEqual(d["curve"]["ext"][0], 0.0)
        self.assertEqual(d["curve"]["force_pn"][0], 0.0)
        lam = d["curve"]["lambda"]
        self.assertTrue(all(b >= a for a, b in zip(lam, lam[1:])), "λ 必须单调增")

    def test_curve_is_universal_in_n_and_l(self):
        """换 n、换 l，λ(x) 必须**逐点相同** —— L(x) 里根本没有 n 和 l。

        这是这条曲线值得单独画的全部理由（和 P(h) 归一化视图同源）。
        只要有一处把 n 或 l 混进 λ，这条立刻红。
        """
        base = fjc.force_extension(7, 1.0, 10.0).curve_lambda
        for n, l in ((1, 1.0), (1000, 1.0), (100, 0.25), (9999, 3.7)):
            other = fjc.force_extension(n, l, 10.0).curve_lambda
            self.assertTrue(np.array_equal(base, other),
                            f"n={n}, l={l} 的 λ 曲线变了")

    def test_absolute_extension_scales_with_n_and_l(self):
        """λ 不依赖 n、l，但绝对伸长 ⟨x⟩ = n·l·λ 必须依赖，且线性。"""
        r1 = fjc.force_extension(100, 1.0, 5.0)
        r2 = fjc.force_extension(100, 2.0, 5.0)   # l 翻倍
        r3 = fjc.force_extension(400, 1.0, 5.0)   # n 翻两番
        self.assertTrue(np.allclose(r2.curve_ext, r1.curve_ext * 2.0))
        self.assertTrue(np.allclose(r3.curve_ext, r1.curve_ext * 4.0))
        # n·l·λ 这条恒等式本身
        i = 17
        self.assertAlmostEqual(r1.curve_ext[i],
                               100 * 1.0 * r1.curve_lambda[i], places=12)

    def test_linear_reference_is_x_over_3_and_stops_at_full_extension(self):
        """对照的高斯链直线 λ_lin = x/3，截到 λ_lin ≤ 1（x ≤ 3）为止。

        截断的理由写在字段说明里：再往右直线会预测出超过全伸展的伸长。
        """
        r = fjc.force_extension(50, 1.0, 10.0)
        xs, ys = r.curve_linear_x, r.curve_linear_lambda
        self.assertGreater(len(xs), 10, "对照线不该被裁得看不见")
        # 末点是**网格上最后一个 ≤ 3 的点**，不必然等于 3（400 点的步长是 10/399）
        step = r.x_max / (fjc.FORCE_CURVE_POINTS - 1)
        self.assertLessEqual(xs[-1], 3.0)
        self.assertGreater(xs[-1], 3.0 - step, "3 应当落在网格点上或只差一步")
        self.assertAlmostEqual(ys[-1], xs[-1] / 3.0, places=15)
        self.assertAlmostEqual(ys[-1], 1.0, delta=step / 3.0)
        self.assertTrue(np.allclose(ys, xs / 3.0))
        self.assertTrue(np.all(ys <= 1.0 + 1e-15))

        # x_max < 3 时整段都在，末点就是 x_max
        r2 = fjc.force_extension(50, 1.0, 1.0)
        self.assertAlmostEqual(r2.curve_linear_x[-1], 1.0, places=12)
        self.assertAlmostEqual(r2.curve_linear_lambda[-1], 1.0 / 3.0, places=12)

    def test_x_linear_end_is_where_deviation_hits_two_percent(self):
        """x_linear_end 是「L 与直线差 2%」那个点，二分的两半要各说各话。"""
        r = fjc.force_extension(100, 1.0, 10.0)
        xe = r.x_linear_end
        self.assertGreater(xe, 0.0)
        self.assertLess(xe, 3.0)

        def dev(x):
            return abs(fjc.langevin(x) / x - 1.0 / 3.0) / (1.0 / 3.0)

        self.assertAlmostEqual(dev(xe), 0.02, delta=2e-4)
        self.assertLess(dev(xe * 0.999), 0.02)
        self.assertGreater(dev(xe * 1.001), 0.02)

    def test_default_range_emits_no_warning(self):
        """默认配置必须干净 —— 一打开就黄条等于没提示。"""
        r = fjc.force_extension(100, 1.0, 10.0)
        self.assertEqual(r.warnings, [])

    def test_warning_when_axis_too_narrow(self):
        r = fjc.force_extension(100, 1.0, 0.5)
        self.assertEqual(len(r.warnings), 1)
        self.assertIn("横轴", r.warnings[0])

    def test_warning_when_pushed_past_where_real_chains_hold(self):
        r = fjc.force_extension(100, 1.0, 30.0)
        self.assertEqual(len(r.warnings), 1)
        # 措辞要分清：FJC 内部精确，失真的是真实材料的前提
        self.assertIn("FJC 内部是精确的", r.warnings[0])
        self.assertGreater(r.lambda_at_x_max, 0.95)

    def test_force_pn_conversion(self):
        """单位链有唯一正确答案：x·k_BT/l。T=300K, l=1nm, x=1 → 4.1419 pN
        （即室温下的 k_BT/nm，是这类换算的标准量级自检）。"""
        self.assertAlmostEqual(fjc.force_in_pn(1.0, 300.0, 1.0),
                               4.141947, places=5)
        # 线性：x 加倍、T 加倍、l 减半，各按各自的方向走
        self.assertAlmostEqual(fjc.force_in_pn(2.0, 300.0, 1.0),
                               2 * fjc.force_in_pn(1.0, 300.0, 1.0), places=10)
        self.assertAlmostEqual(fjc.force_in_pn(1.0, 600.0, 1.0),
                               2 * fjc.force_in_pn(1.0, 300.0, 1.0), places=10)
        self.assertAlmostEqual(fjc.force_in_pn(1.0, 300.0, 0.5),
                               2 * fjc.force_in_pn(1.0, 300.0, 1.0), places=10)
        self.assertEqual(fjc.force_in_pn(0.0, 300.0, 1.0), 0.0)

    def test_force_pn_validates(self):
        for bad in (dict(x=-1.0), dict(x=float("nan")),
                    dict(x=1.0, temperature=0.0),
                    dict(x=1.0, temperature=2000.0),
                    dict(x=1.0, l_nm=0.0),
                    dict(x=1.0, l_nm=-3.0)):
            with self.assertRaises(fjc.FJCInputError, msg=str(bad)):
                fjc.force_in_pn(**bad)

    def test_x_from_force_pn_round_trips(self):
        """pN ↔ x 必须是互逆的 —— 否则「0.5 pN 能拉多长」和「拉到这个程度要多少 pN」
        会给出两个对不上的答案，而图上只画得出其中一条。"""
        for x, t, l in ((1.0, 300.0, 1.0), (0.0, 298.15, 1.0),
                        (17.3, 293.15, 2.5), (100.0, 400.0, 0.33)):
            with self.subTest(x=x, t=t, l=l):
                p = fjc.force_in_pn(x, t, l)
                self.assertAlmostEqual(fjc.x_from_force_pn(p, t, l), x, places=11)
                # 反过来也得走一遍，单向对上说明不了互逆
                self.assertAlmostEqual(
                    fjc.force_in_pn(fjc.x_from_force_pn(p, t, l), t, l), p,
                    places=9)

    def test_x_from_force_pn_known_value(self):
        """1 个 x 就是 k_BT/nm，所以反过来 k_BT/nm 这么大的力应当正好给出 x=1。"""
        self.assertAlmostEqual(fjc.x_from_force_pn(
            fjc.force_in_pn(1.0, 300.0, 1.0), 300.0, 1.0), 1.0, places=12)
        # 单位链那条自检换个方向问：4.141947 pN → x = 1
        self.assertAlmostEqual(fjc.x_from_force_pn(4.141947, 300.0, 1.0),
                               1.0, places=5)
        self.assertEqual(fjc.x_from_force_pn(0.0, 300.0, 1.0), 0.0)

    def test_x_from_force_pn_is_linear_in_l_and_inverse_in_t(self):
        """换算是线性的，所以 l 翻倍 x 翻倍、T 翻倍 x 减半 —— 这两条最容易搞反。"""
        base = fjc.x_from_force_pn(10.0, 300.0, 1.0)
        self.assertAlmostEqual(fjc.x_from_force_pn(10.0, 300.0, 2.0),
                               2 * base, places=11)
        self.assertAlmostEqual(fjc.x_from_force_pn(10.0, 600.0, 1.0),
                               0.5 * base, places=12)

    def test_x_from_force_pn_validates(self):
        for bad in (dict(force_pn=-1.0), dict(force_pn=float("nan")),
                    dict(force_pn="abc"),
                    dict(force_pn=1.0, temperature=0.0),
                    dict(force_pn=1.0, temperature=2000.0),
                    dict(force_pn=1.0, l_nm=0.0),
                    dict(force_pn=1.0, l_nm=-3.0),
                    dict(force_pn=1.0, temperature="abc")):
            with self.subTest(**bad):
                with self.assertRaises(fjc.FJCInputError):
                    fjc.x_from_force_pn(**bad)

    def test_force_in_pn_also_rejects_non_numeric(self):
        """原来 float("abc") 会漏出 TypeError，Flask 那边只接 FJCInputError，
        结果是 500 而不是一句中文。两个方向都得走同一条报错路。"""
        for bad in (dict(x="abc"), dict(x=None),
                    dict(x=1.0, temperature="abc"), dict(x=1.0, l_nm=None)):
            with self.subTest(**bad):
                with self.assertRaises(fjc.FJCInputError):
                    fjc.force_in_pn(**bad)

    def test_curve_force_pn_tracks_the_dimensionless_axis(self):
        """曲线里的 pN 列必须和 x 列一致，不能各算各的。"""
        r = fjc.force_extension(100, 1.0, 10.0, temperature=300.0)
        for i in (0, 5, 199, 399):
            self.assertAlmostEqual(
                r.curve_force_pn[i],
                fjc.force_in_pn(r.curve_x[i], 300.0, 1.0), places=9, msg=f"i={i}")

    def test_validation(self):
        cases = [
            dict(n=100, x_max=-1),
            dict(n=100, x_max=0),
            dict(n=100, x_max=1000),
            dict(n=100, x_max=float("nan")),
            dict(n=100, x_max="abc"),
            dict(n=100, points=1),
            dict(n=100, points=2001),
            dict(n=100, points=0),
            dict(n=100, temperature=-5.0),
            dict(n=100, l=-1.0),          # 借 validate() 的同一条路
            dict(n=0, x_max=10.0),
            dict(n=10 ** 9, x_max=10.0),
        ]
        for bad in cases:
            with self.assertRaises(fjc.FJCInputError, msg=str(bad)):
                fjc.force_extension(**bad)

    def test_to_dict_is_json_serialisable(self):
        import json
        d = json.loads(json.dumps(fjc.force_extension(100, 1.0, 10.0).to_dict()))
        self.assertEqual(d["n"], 100)
        self.assertEqual(len(d["curve"]["x"]), 400)
        self.assertIn("curve_linear", d)
        self.assertEqual(len(d["curve_linear"]["x"]), len(d["curve_linear"]["lambda"]))
        self.assertIsInstance(d["x_linear_end"], float)
        self.assertIsInstance(d["warnings"], list)


class TestSolveN(unittest.TestCase):
    """h → n 反解。"""

    KINDS = ("h_rms", "h_mean", "h_mp", "h_max")

    def test_known_values(self):
        self.assertEqual(fjc.solve_n(10.0, 1.0, "h_rms").n_exact, 100.0)
        self.assertEqual(fjc.solve_n(100.0, 1.0, "h_max").n_exact, 100.0)
        # ⟨h⟩ = l√(8n/(3π))，取 n=100 反推
        n = fjc.solve_n(fjc.compute(100, 1.0).h_mean, 1.0, "h_mean").n_exact
        self.assertAlmostEqual(n, 100.0, places=8)
        n = fjc.solve_n(fjc.compute(100, 1.0).h_mp, 1.0, "h_mp").n_exact
        self.assertAlmostEqual(n, 100.0, places=8)

    def test_round_trip_for_every_kind(self):
        """正向再反向必须回到原点 —— 两种 n、五个数量级。"""
        for kind in self.KINDS:
            for n in (1, 7, 100, 12345, 9_999_999):
                h = fjc._n_to_h(n, 1.0, kind)
                got = fjc._h_to_n_exact(h, 1.0, kind)
                self.assertLessEqual(abs(got - n), 1e-8 * max(1, n),
                                     msg=f"{kind}, n={n}")

    def test_forward_formulas_match_compute(self):
        """正向公式不能是第二套 —— 必须与 compute() 逐位相同。"""
        for kind in self.KINDS:
            for n, l in ((1, 1.0), (10, 1.0), (100, 1.0), (9999, 1.0),
                         (50, 2.5), (7, 0.38)):
                mine = fjc._n_to_h(n, l, kind)
                theirs = getattr(fjc.compute(n, l), kind)
                self.assertEqual(mine, theirs, msg=f"{kind}, n={n}, l={l}")

    def test_solve_backs_out_through_compute(self):
        """端到端：算出的整数 n，拿去调 compute()，得到的 h 要夹住目标。"""
        for kind in self.KINDS:
            h_target = 37.0
            r = fjc.solve_n(h_target, 1.0, kind)
            self.assertLessEqual(r.h_floor, h_target + 1e-9, msg=kind)
            self.assertGreaterEqual(r.h_ceil, h_target - 1e-9, msg=kind)
            # n_floor / n_ceil 是整数，且真的能喂给 compute()
            for nn in (r.n_floor, r.n_ceil):
                self.assertEqual(nn, int(nn))
                self.assertGreaterEqual(nn, 1)
                h_back = getattr(fjc.compute(nn, 1.0), kind)
                self.assertAlmostEqual(h_back,
                                       fjc._n_to_h(nn, 1.0, kind), places=9)

    def test_floor_and_ceil_bracket_the_target(self):
        # h ≤ 2000 是为了让**四种** kind 都不撞 N_MAX：
        # h_rms 需要 n=(h/l)²，2000² = 4e6；h_mean、h_mp 的系数都 >1，也还在内。
        for kind in self.KINDS:
            for h in (3.3, 17.0, 250.0, 2000.0):
                r = fjc.solve_n(h, 1.0, kind)
                self.assertLessEqual(r.n_floor, r.n_exact)
                self.assertGreaterEqual(r.n_ceil, r.n_exact)
                self.assertIn(r.n, (r.n_floor, r.n_ceil))
                self.assertEqual(r.n, r.n_ceil)  # 推荐值取「不低于目标」
                self.assertLessEqual(r.h_floor, h)
                self.assertGreaterEqual(r.h_ceil, h)

    def test_integer_input_is_recovered_exactly(self):
        """由整数 n 生成的 h 反解回去，必须拿回那个 n。

        断的是 `⌈⌉` 那个（推荐值），不是 floor == ceil —— 后者在浮点下
        不成立：h_mean 的往返会落到 63.99999999999999，floor 变成 63。
        那是**如实报告**（连续解确实略小于 64），不是 bug；要紧的是
        推荐值仍然给出 64，且两个整数解夹住它。
        """
        for kind in self.KINDS:
            n0 = 64
            h = fjc._n_to_h(n0, 1.0, kind)
            r = fjc.solve_n(h, 1.0, kind)
            self.assertLess(abs(r.n_exact - n0), 1e-6, msg=kind)
            self.assertEqual(r.n, n0, msg=f"{kind}: 推荐值应当回到 {n0}")
            self.assertIn(n0, (r.n_floor, r.n_ceil), msg=kind)
            # 取整的差只有浮点噪声量级，不该惊动用户
            self.assertFalse(any("差了" in w for w in r.warnings), r.warnings)

    def test_inverse_preserves_the_ordering_of_the_four_kinds(self):
        """同一 h 下，n 的次序必须和正向 h 的次序**反着来**。

        正向恒有 h* < ⟨h⟩ < h_rms < nl，所以每个 h 所需的 n 恰好倒序：
        nl 要的最少、h* 要的最多。四种反解写成同一个式子会立刻在这里红。
        """
        h = 50.0
        ns = [fjc.solve_n(h, 1.0, k).n_exact for k in self.KINDS]
        # KINDS 顺序 = h_rms, h_mean, h_mp, h_max —— 所以断言里点名，不写 sorted：
        # 这个次序是**推出来的**（下面三条），不是「碰巧升序」。
        n_rms, n_mean, n_mp, n_max = ns
        self.assertLess(n_max, n_rms)     # nl 最省
        self.assertLess(n_rms, n_mean)    # h_rms < ⟨h⟩ ⟹ 要的 n 更少
        self.assertLess(n_mean, n_mp)     # h* 最费

    def test_l_scales_the_answer(self):
        """l 翻倍，同一 h 需要的 n 全部降为 1/4（√ 型）或 1/2（h_max）。"""
        h = 40.0
        for kind in self.KINDS:
            a = fjc.solve_n(h, 1.0, kind).n_exact
            b = fjc.solve_n(h, 2.0, kind).n_exact
            factor = 0.5 if kind == "h_max" else 0.25
            self.assertAlmostEqual(b, a * factor, places=6, msg=kind)

    def test_rounding_warning_only_when_it_matters(self):
        """取整要不要紧，只看两个整数解给出的 h 差多少，不看 n 的绝对大小。"""
        # n_exact = 1.96 → ⌊⌋=1、⌈⌉=2，h 差 41%，必须提
        r = fjc.solve_n(1.4, 1.0, "h_rms")
        self.assertAlmostEqual(r.n_exact, 1.96, places=10)
        self.assertEqual(r.n_floor, 1)
        self.assertEqual(r.n_ceil, 2)
        self.assertTrue(any("差了" in w for w in r.warnings), r.warnings)

        # n_exact = 1e6，相邻整数只差 1e-6，不许提
        r = fjc.solve_n(1000.0, 1.0, "h_rms")
        self.assertEqual(r.n_floor, r.n_ceil)
        self.assertEqual(r.warnings, [])

    def test_gaussian_warning_only_for_the_two_gaussian_kinds(self):
        """h* 与 ⟨h⟩ 来自 P(h) 的高斯近似，小 n 要说；h_rms 与 nl 是精确的，不许说。"""
        h_mp_small = fjc._n_to_h(5.0, 1.0, "h_mp")   # n=5 < GAUSSIAN_N_MIN
        self.assertLess(5, fjc.GAUSSIAN_N_MIN)

        r = fjc.solve_n(h_mp_small, 1.0, "h_mp")
        self.assertTrue(any("高斯近似" in w for w in r.warnings), r.warnings)

        r = fjc.solve_n(h_mp_small, 1.0, "h_rms")
        self.assertFalse(any("高斯近似" in w for w in r.warnings), r.warnings)

        r = fjc.solve_n(50.0, 1.0, "h_max")   # n=50，够大
        self.assertFalse(any("高斯近似" in w for w in r.warnings), r.warnings)

    def test_rejects_bad_input(self):
        bad = [
            dict(h=-1.0),
            dict(h=0.0),
            dict(h=float("nan")),
            dict(h=float("inf")),
            dict(h="abc"),
            dict(h=10.0, kind="nope"),
            dict(h=10.0, kind="h_rms "),           # 尾随空格不认
            dict(h=10.0, kind=None),
            dict(h=10.0, l=-1.0),                  # 借 validate() 的同一条路
            dict(h=10.0, l=0.0),
            dict(h=1e12, l=1.0),                   # 反解出的 n 超上限
            dict(h=0.5, l=1.0),                    # h < l ⟹ n < 1
        ]
        for b in bad:
            with self.assertRaises(fjc.FJCInputError, msg=str(b)):
                fjc.solve_n(**b)

    def test_kind_list_is_the_four_physical_ones(self):
        self.assertEqual(set(fjc.H_KINDS),
                         {"h_rms", "h_mean", "h_mp", "h_max"})
        for kind in fjc.H_KINDS:
            self.assertIn(kind, fjc.H_KIND_LABELS)

    def test_to_dict_is_json_serialisable(self):
        import json
        d = json.loads(json.dumps(fjc.solve_n(10.0, 1.0, "h_rms").to_dict()))
        self.assertEqual(d["n"], 100)
        self.assertEqual(d["n_floor"], 100)
        self.assertEqual(d["n_ceil"], 100)
        self.assertEqual(d["n"], d["n"])           # 推荐值来自 ⌈⌉
        self.assertIsInstance(d["n_exact"], float)
        self.assertIsInstance(d["warnings"], list)


class TestKuhnPresets(unittest.TestCase):
    """Kuhn 长度预置表。**这张表刻意不装权威** —— 所以测的是形状与自洽，
    不是把某个来源的数字钉死（那会把「常见引值」假装成定论）。"""

    def test_b_is_exactly_twice_p(self):
        """b = 2p 是定义，没有余地 —— 表里哪一条破了都是抄错。"""
        for p in fjc.kuhn_presets():
            self.assertAlmostEqual(p["b_nm"], 2.0 * p["p_nm"], places=12,
                                   msg=p["name"])
            self.assertGreater(p["b_nm"], 0.0)
            self.assertGreater(p["p_nm"], 0.0)

    def test_every_row_has_a_confidence_label(self):
        """每条都得标明可信度 —— 不许有「没说可不可信」的行。"""
        rows = fjc.kuhn_presets()
        self.assertGreaterEqual(len(rows), 2)
        for p in rows:
            self.assertIn(p["conf"], fjc.KUHN_CONF_LABELS, msg=p["name"])
            self.assertTrue(p["conf_label"], msg=p["name"])
            self.assertTrue(p["note"], msg=p["name"])
            self.assertTrue(p["name"], msg=p["name"])

    def test_at_least_one_row_is_flagged_as_hard(self):
        """表里必须有硬的那条（DNA）—— 否则整张表就都是量级猜测，
        那不如不给。"""
        confs = {p["conf"] for p in fjc.kuhn_presets()}
        self.assertIn("hard", confs)
        self.assertIn("nominal", confs)  # 也得有明说是「散布大」的

    def test_dna_is_the_textbook_anchor(self):
        """DNA 的持久长度 ≈ 50 nm 是教科书值，这条错了说明表本身有问题。"""
        dna = next(p for p in fjc.kuhn_presets() if "DNA" in p["name"])
        self.assertAlmostEqual(dna["p_nm"], 50.0, places=9)
        self.assertEqual(dna["conf"], "hard")

    def test_pe_is_derived_from_the_kuhn_definition(self):
        """PE 那条是 b = a√C∞ 推出来的（a=0.254nm, C∞≈6.7），不是抄的。"""
        pe = next(p for p in fjc.kuhn_presets() if "聚乙烯" in p["name"])
        derived = 0.254 * math.sqrt(6.7)
        self.assertAlmostEqual(pe["b_nm"], derived, delta=0.02)
        self.assertEqual(pe["conf"], "derived")

    def test_returns_plain_data(self):
        """要能直接 json.dumps —— 这是给 /api 和页面下拉用的。"""
        import json
        rows = fjc.kuhn_presets()
        json.dumps(rows, ensure_ascii=False)
        self.assertIsInstance(rows, list)
        self.assertIsInstance(rows[0], dict)


if __name__ == "__main__":
    unittest.main(verbosity=2)
