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


class TestSaw(unittest.TestCase):
    """自回避行走：严格拒绝采样的正确性 + ⟨R²⟩ ∝ n^(2ν) 的实测。

    这里和别的类有一个根本区别：SAW **没有解析式**可以对照。所以「算得对不对」
    只能换一个独立来源来验 —— 用**现场 DFS 枚举**数出精确的 c_n，再看采样器的
    存活率是不是 c_n/6^n。这比断言某个硬编码常数强得多：把采样器改坏成
    Rosenbluth（只在空闲邻居里选）时存活率会变成 ~1，这条测试立刻炸。
    """

    # 现场 DFS 的 n 上限。n=8 要访问约 49 万个格点，不到一秒；n=9 就上千万了。
    DFS_N_MAX = 8

    def _exact_cn(self, n):
        """DFS 枚举 n 步自回避行走的条数。独立于内核里那张表。"""
        if not hasattr(self, "_cn_cache"):
            self._cn_cache = {}
        if n in self._cn_cache:
            return self._cn_cache[n]
        dirs = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
        total = 0

        def walk(x, y, z, visited, steps):
            nonlocal total
            if steps == n:
                total += 1
                return
            for dx, dy, dz in dirs:
                nxt = (x + dx, y + dy, z + dz)
                if nxt in visited:
                    continue
                visited.add(nxt)
                walk(nxt[0], nxt[1], nxt[2], visited, steps + 1)
                visited.discard(nxt)

        walk(0, 0, 0, {(0, 0, 0)}, 0)
        self._cn_cache[n] = total
        return total

    def test_exact_cn_table_matches_enumeration(self):
        """内核里那张精确表必须**逐项**等于现场枚举的结果。

        这张表是估批量和判断「采样器坏了没」的依据，它自己得先被钉死。
        """
        for n in range(1, self.DFS_N_MAX + 1):
            self.assertEqual(
                fjc._SAW_C_N[n], self._exact_cn(n),
                msg=f"n={n} 的表值 {fjc._SAW_C_N[n]} 与枚举 {self._exact_cn(n)} 不符",
            )

    def test_survival_equals_exact_cn_over_6_to_the_n(self):
        """采样器的**实测存活率**必须等于精确的 c_n/6^n。

        严格均匀的要害就在这里：每一步在全部 6 个方向里均匀选、撞上就丢，
        存活率才是「活下来的行走数 / 6^n」。若改成在空闲邻居里均匀选
        （Rosenbluth），存活率会趋近 1 而不再是 c_n/6^n —— 那是有偏采样。
        """
        trials = 150_000
        for n in (4, 8):
            exact = self._exact_cn(n) / 6.0 ** n
            _, t, produced = fjc._saw_endpoints(
                np.random.default_rng(4242 + n), n, 10 ** 9, trials
            )
            self.assertEqual(t, trials)
            got = produced / t
            se = math.sqrt(exact * (1 - exact) / t)
            self.assertLess(abs(got - exact), 4 * se,
                            msg=f"n={n} 存活率 {got:.5f} vs 精确 {exact:.5f}")

    def test_accept_rate_counts_survivors_not_returned_chains(self):
        """accept_rate 必须是「活下来多少 / 试投多少」，不是「返回多少 / 试投多少」。

        这两者只在最后一批被截断时不同，而恰恰是那里最容易写错：按返回条数算
        会报出 0.0014 这种数（真值 0.146），看着像采样器坏了。
        """
        n = 4
        exact = self._exact_cn(n) / 6.0 ** n
        r = fjc.saw_chain(n, seed=9, chains=300)
        self.assertEqual(r.chains, 300)
        self.assertGreater(r.trials, 300)          # 一定试投得比要的多
        se = math.sqrt(exact * (1 - exact) / r.trials)
        self.assertLess(abs(r.accept_rate - exact), 4 * se,
                        msg=f"accept_rate={r.accept_rate:.5f} vs 精确 {exact:.5f}")

    def test_batch_sizing_does_not_overshoot_wildly(self):
        """试投次数要贴着「需要多少投多少」，不能按内存上限整批投。

        按内存上限投的话 n=8 一批就是 16 万次试投（活下来几万条）而只要 500 条，
        白烧掉大量算力。这里用「试投次数 / (链数/存活率)」这个纯比值来卡，
        不用计时 —— 计时在不同机器上会飘。
        """
        n, want = 8, 500
        exact = self._exact_cn(n) / 6.0 ** n
        _, t, _ = fjc._saw_endpoints(np.random.default_rng(1), n, want, 10 ** 9)
        ideal = want / exact
        self.assertLess(t / ideal, 2.0, msg=f"试投 {t} 次，理想约 {ideal:.0f} 次")
        self.assertGreater(t / ideal, 0.3)

    def test_every_chain_is_self_avoiding(self):
        """每条链都不能重复访问同一个格点 —— 这是模型的定义。"""
        r = fjc.saw_chain(12, 1.0, seed=77, chains=60)
        for i, chain in enumerate(r.points):
            sites = {(round(x), round(y), round(z)) for x, y, z in chain}
            self.assertEqual(len(sites), chain.shape[0],
                             msg=f"第 {i} 条链有重复格点")

    def test_steps_are_unit_lattice_vectors_of_length_l(self):
        """每一步都必须是 ±l 的轴向步：步长恒为 l，且落在格点上。"""
        l = 2.5
        r = fjc.saw_chain(9, l, seed=31, chains=20)
        steps = np.diff(r.points, axis=1)
        # 每步只有一个分量非零，且绝对值等于 l
        self.assertTrue(np.all(np.sum(np.abs(steps) > 0, axis=2) == 1))
        np.testing.assert_allclose(np.linalg.norm(steps, axis=2), l, atol=1e-12)
        # 顶点是 l 的整数倍（格点）
        np.testing.assert_allclose(r.points / l, np.round(r.points / l), atol=1e-9)
        # 首点是原点
        np.testing.assert_allclose(r.points[:, 0, :], 0.0, atol=1e-12)

    def test_shapes_agree_between_saw_and_ideal(self):
        """SAW 与理想链必须同 n 同链数，否则并排画出来不是一个尺度的对照。"""
        r = fjc.saw_chain(7, 1.0, seed=5, chains=12)
        self.assertEqual(r.points.shape, (12, 8, 3))
        self.assertEqual(r.ideal_points.shape, (12, 8, 3))
        self.assertEqual(r.R.shape, (12, 3))
        self.assertEqual(r.R_mag.shape, (12,))

    def test_ideal_chain_on_the_lattice_recovers_nl2(self):
        """同格点的理想链（每步 6 选 1，不管走没走过）⟨R²⟩ = n·l² 精确成立。

        这是 SAW 卡片的对照基线：两者唯一的差别就是那条排除约束。理想链这一侧
        必须落在自己 √(2/3)/√M 的涨落里 —— 高斯链这里用得上，因为理想链**就是**
        高斯链。
        """
        n, M, l = 20, 3000, 1.0
        r = fjc.saw_chain(n, l, seed=101, chains=M)
        se = math.sqrt(2.0 / 3.0) / math.sqrt(M)
        self.assertAlmostEqual(r.ideal_R2_mean / (n * l * l), 1.0,
                               delta=4 * se, msg=f"理想链 ⟨R²⟩/{n}l²")

    def test_swelling_is_the_whole_point(self):
        """SAW 的 ⟨R²⟩ 必须明显**大于** n·l²，且随 n 单调变强。

        这就是排除体积效应：链越舒展，⟨R²⟩ 相对理想链越大。实测 n=5 时约 1.45，
        n=30 时约 2.06 —— 若装成第五个解析模型、⟨h²⟩ = n·l²·Cn，这一条根本不会出现。
        """
        small = fjc.saw_chain(5, 1.0, seed=11, chains=800)
        big = fjc.saw_chain(30, 1.0, seed=11, chains=800)
        self.assertGreater(small.swelling, 1.3)
        self.assertGreater(big.swelling, small.swelling)
        self.assertGreater(big.swelling, 1.7)
        for r in (small, big):
            self.assertAlmostEqual(r.swelling, r.R2_mean / (r.n * r.l ** 2), places=12)

    def test_measured_rel_se_beats_the_gaussian_formula(self):
        """SAW 的 rel_se 是实测的，而且**小于**高斯的 √(2/3)/√M。

        sweep() 是无顶点的，只能用高斯公式 √(2/3)/√M；SAW 手上就有每条链的 R²，
        所以用实测 std/√M 更准。实测 std(R²)/⟨R²⟩ ≈ 0.5~0.63 对高斯的 0.816，
        套高斯公式会把误差报大约 40% —— 这不是笔误，是刻意的分歧。
        """
        M = 2000
        r = fjc.saw_chain(12, 1.0, seed=13, chains=M)
        gaussian = math.sqrt(2.0 / 3.0) / math.sqrt(M)
        self.assertLess(r.R2_rel_se, gaussian)
        r2 = r.R_mag ** 2
        self.assertAlmostEqual(
            r.R2_rel_se, float(np.std(r2, ddof=1)) / r.R2_mean / math.sqrt(M),
            places=12,
        )

    def test_reproducible(self):
        a = fjc.saw_chain(8, 1.0, seed=2026, chains=40)
        b = fjc.saw_chain(8, 1.0, seed=2026, chains=40)
        np.testing.assert_array_equal(a.points, b.points)
        np.testing.assert_array_equal(a.ideal_points, b.ideal_points)
        self.assertEqual(a.trials, b.trials)
        # 换种子就该不一样（否则种子根本没接进去）
        c = fjc.saw_chain(8, 1.0, seed=2027, chains=40)
        self.assertFalse(np.array_equal(a.points, c.points))

    def test_sweep_slope_is_above_one(self):
        """拟合斜率必须显著 > 1 —— 这是「⟨R²⟩ 不再 ∝ n」的直接检验。

        1 是理想链的斜率。SAW 的斜率实测约 1.19，落在 2ν = 1.1752 附近而不是 1
        附近。seed 写死以保证确定性，不然会偶发失败。
        """
        r = fjc.saw_sweep([4, 8, 12, 16, 20], 1.0, seed=2026, chains=300)
        self.assertGreater(r.slope, 1.0 + 3 * r.slope_se,
                           msg=f"斜率 {r.slope} ± {r.slope_se}，没有显著大于 1")
        self.assertGreater(r.fit_r2, 0.99)

    def test_sweep_reference_is_2nu_and_reported(self):
        """参考斜率是 2ν = 1.175194，有效 ν = 斜率/2，且要如实高于 2ν。

        这一条把「不许承诺 1.176」写进测试：n ≤ 30 的有限尺寸修正把斜率**推高**，
        截面报的必须是实测的有效 ν，而不是拿 2ν 去凑。
        """
        r = fjc.saw_sweep([5, 8, 12, 16, 20, 25, 30], 1.0, seed=1, chains=400)
        self.assertAlmostEqual(r.slope_ref, 2 * fjc.SAW_NU, places=12)
        self.assertAlmostEqual(r.nu_eff, r.slope / 2, places=12)
        self.assertGreater(r.slope, r.slope_ref,
                           msg="这片 n 区间实测就该高于 2ν，否则是采样出了问题")
        # 但也不能高得离谱（高于 2ν 三倍标准误以上才算异常）
        self.assertLess(r.slope - r.slope_ref, 5 * r.slope_se)

    def test_sweep_reference_lines_pass_through_the_centroid(self):
        """两条参考线只有斜率是外部信息，都锚在数据质心上（不引振幅）。

        所以它们在 ln n = mean(ln n) 处必须同时穿过 mean(ln⟨R²⟩)。
        """
        r = fjc.saw_sweep([5, 10, 20, 30], 1.0, seed=8, chains=300)
        xbar = float(np.mean(np.log(r.ns)))
        ybar = float(np.mean(np.log(r.R2_mean)))
        self.assertAlmostEqual(r.intercept_ideal + 1.0 * xbar, ybar, places=10)
        self.assertAlmostEqual(r.intercept_ref + r.slope_ref * xbar, ybar, places=10)

    def test_sweep_slope_se_generalises_the_sweep_formula(self):
        """各点误差相同时，逐点传播必须退化成 σ/√Sxx（就是 sweep() 那条）。"""
        r = fjc.saw_sweep([4, 8, 16, 30], 1.0, seed=3, chains=200,
                          max_points=None)
        ln_n = np.log(np.asarray(r.ns, dtype=float))
        sxx = float(np.sum((ln_n - ln_n.mean()) ** 2))
        flat = float(np.mean(r.rel_se))
        naive = flat / math.sqrt(sxx)
        # 实测各点 rel_se 不完全相同，所以只能比量级与相近程度
        self.assertLess(abs(r.slope_se - naive) / naive, 0.5)
        # 但逐点传播必须严格按各自的 rel_se 来算
        num = float(np.sum(((ln_n - ln_n.mean()) ** 2) * np.asarray(r.rel_se) ** 2))
        self.assertAlmostEqual(r.slope_se, math.sqrt(num) / sxx, places=12)

    def test_sweep_reports_measured_survival_and_trials(self):
        """trials 与 accept_rate 必须是实测值，且随 n 显著衰减。"""
        r = fjc.saw_sweep([4, 12, 20], 1.0, seed=6, chains=200)
        self.assertEqual(len(r.trials), len(r.ns))
        self.assertEqual(len(r.chains_used), len(r.ns))
        self.assertTrue(all(c == 200 for c in r.chains_used))
        # 存活率随 n 单调下降，数量级差很多
        self.assertGreater(r.trials[0] < r.trials[1] < r.trials[2], 0)
        self.assertGreater(r.trials[2] / r.trials[0], 10)

    def test_sweep_warns_when_it_cannot_fill_the_sample(self):
        """试投上限兜住时要**说出来**是哪几个 n，不能静默少给。"""
        r = fjc.saw_sweep([4, 30], 1.0, seed=2, chains=500,
                          trials_per_n=20_000, trials_total=None)
        self.assertLess(r.chains_used[1], 500)
        self.assertTrue(any("n=30" in w for w in r.warnings))
        self.assertTrue(any("没采满" in w for w in r.warnings))

    def test_validation(self):
        """上限都要抛中文 FJCInputError，且理由要写清楚是算力而非物理。"""
        with self.assertRaises(fjc.FJCInputError):
            fjc.saw_chain(31)                       # 超过 SAW_N_MAX
        with self.assertRaises(fjc.FJCInputError):
            fjc.saw_chain(0)                        # n 至少为 1
        with self.assertRaises(fjc.FJCInputError):
            fjc.saw_chain(5, chains=0)               # 链数至少为 1
        with self.assertRaises(fjc.FJCInputError):
            fjc.saw_chain(5, chains=10, max_chains=5)
        with self.assertRaises(fjc.FJCInputError):
            fjc.saw_chain(5, -1.0)                   # l 必须为正
        with self.assertRaises(fjc.FJCInputError):
            fjc.saw_sweep("5, 31")                   # 扫描里也有同一个上限
        with self.assertRaises(fjc.FJCInputError):
            fjc.saw_sweep("5, 5")                    # n 重复
        with self.assertRaises(fjc.FJCInputError):
            fjc.saw_sweep("")                        # 空的 n 列表
        with self.assertRaises(fjc.FJCInputError) as cm:
            fjc.saw_sweep([5, 25, 30], chains=20_000)   # 试投总预算
        self.assertIn("超出预算", str(cm.exception))
        # 预算没超的同一组 n 就该放行（说明卡的是总量，不是 n 的个数）
        r = fjc.saw_sweep([5, 30], chains=200)
        self.assertEqual(r.chains_used, [200, 200])
        # 上限单独解除后确实能过（说明这些是表现/算力约束，不是物理约束）
        r = fjc.saw_chain(9, 1.0, seed=1, chains=60, max_n=None)
        self.assertEqual(r.chains, 60)

    def test_to_dict_is_json_serialisable(self):
        import json
        c = json.loads(json.dumps(fjc.saw_chain(6, 1.0, seed=4, chains=8).to_dict()))
        self.assertEqual(c["n"], 6)
        self.assertEqual(len(c["points"]), 8)
        self.assertEqual(len(c["points"][0]), 7 * 3)
        self.assertEqual(len(c["ideal_points"]), 8)
        self.assertEqual(c["n_max"], fjc.SAW_N_MAX)

        s = json.loads(json.dumps(
            fjc.saw_sweep([4, 8, 16], 1.0, seed=4, chains=150).to_dict()
        ))
        self.assertEqual(s["n"], [4, 8, 16])
        self.assertEqual(len(s["R2_ideal"]), 3)
        self.assertEqual(len(s["rel_se"]), 3)
        self.assertEqual(len(s["trials"]), 3)
        self.assertIsInstance(s["slope"], float)
        self.assertIsInstance(s["nu_eff"], float)


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


class TestChainModels(unittest.TestCase):
    """四个链模型：定义、极限、以及**采样器与模型理论是不是同一件事**。

    这一组盯的是「模型只定义一处」：compute / random_chain / sweep 都从
    model_h2 取数，采样器不许另算一套理论值 —— 所以下面每个模型都拿
    「采样出来的 ⟨R²⟩」和「模型的 ⟨h²⟩」直接对拍。数字锚点全是教科书值。
    """

    THETA_PE = math.degrees(math.acos(-1.0 / 3.0))   # 聚乙烯 sp³ 键角 109.47°

    def test_default_is_fjc_and_nothing_moved(self):
        self.assertEqual(fjc.ModelParams().key, fjc.MODEL_FJC)
        self.assertEqual(fjc.compute(100, 1.0).h2, 100.0)          # 老锚点
        self.assertEqual(fjc.compute(100, 1.0).Cn, 1.0)
        self.assertEqual(fjc.compute(100, 1.0).model, "fjc")
        self.assertEqual(fjc.compute(100, 1.0).kuhn_length, 1.0)

    def test_frc_anchors(self):
        """自由旋转链：θ=90° 精确退回 FJC；聚乙烯键角的长链极限 Cn = 2。"""
        self.assertAlmostEqual(fjc.freerotating_h2(100, 1.0, 90.0),
                               fjc.compute(100, 1.0).h2, places=9)
        mp = fjc.ModelParams.of("frc", theta_deg=self.THETA_PE)
        self.assertAlmostEqual(fjc.model_cn_infinite(mp), 2.0, places=9)
        # 有限 n 的精确离散值比渐近值小一点（差 O(1/n)），这正是页面对照图要对上的那个数
        cn100 = fjc.model_cn(mp, 100, 1.0)
        self.assertLess(cn100, 2.0)
        self.assertAlmostEqual(cn100, 1.9850, places=3)

    def test_hindered_anchors(self):
        """⟨cosφ⟩=0 时受累旋转链就是自由旋转链；聚乙烯那组长链极限 Cn = 6。"""
        frc = fjc.ModelParams.of("frc", theta_deg=self.THETA_PE)
        hind0 = fjc.ModelParams.of("hindered", theta_deg=self.THETA_PE, cos_phi=0.0)
        for n in (1, 10, 100, 500):
            self.assertAlmostEqual(fjc.model_h2(hind0, n, 1.0),
                                   fjc.model_h2(frc, n, 1.0), places=9)
        pe = fjc.ModelParams.of("hindered", theta_deg=self.THETA_PE, cos_phi=0.5)
        self.assertAlmostEqual(fjc.model_cn_infinite(pe), 6.0, places=6)

    def test_hindered_closed_form_matches_the_transfer_matrix(self):
        """闭式 vs 逐步递推的 2×2 转移矩阵（含 c+g=0 那个退化点）。"""
        def recur(n, c, g):
            if n == 1:
                return 1.0
            s = math.sqrt(max(0.0, 1.0 - c * c))
            a, b = 1.0, 0.0
            tot = 0.0
            for k in range(1, n):
                a, b = c * a + s * g * b, s * a - c * g * b
                tot += (n - k) * a
            return n + 2 * tot

        # θ=120°、⟨cosφ⟩=−0.5 正好落在 c+g=0 上（c=0.5、g=−0.5），是最容易写错的一格
        for theta, g in ((self.THETA_PE, 0.5), (90.0, 0.0), (120.0, -0.5),
                         (60.0, 0.9), (150.0, 0.99)):
            c = -math.cos(math.radians(theta))
            mp = fjc.ModelParams.of("hindered", theta_deg=theta, cos_phi=g)
            for n in (1, 2, 3, 7, 40, 200):
                self.assertAlmostEqual(fjc.model_h2(mp, n, 1.0), recur(n, c, g),
                                       places=9, msg=f"θ={theta} g={g} n={n}")

    def test_energy_figure_is_empty_for_the_two_constraint_only_models(self):
        """自由连接链与自由旋转链没有能量自由度：只能给一个 none 加一句中文理由。

        这一条钉的是「为什么这两张图画不出来」有**一个**落点：卡片收起、前端不写 if，
        全靠 energy_figure() 这条分支。所以它不能是死代码。
        """
        import json
        for key in ("fjc", "frc"):
            fig = fjc.energy_figure(fjc.ModelParams.of(key, theta_deg=self.THETA_PE))
            self.assertEqual(fig["kind"], "none")
            self.assertEqual(fig["model"], key)
            self.assertTrue(fig["reason"].strip(), f"{key} 必须给出中文理由")
            json.dumps(fig, ensure_ascii=False)

    def test_energy_barrier_is_the_sampler_probability(self):
        """图上的位垒与采样器用的是**同一个** p_trans：不同写法，同一个数。

        hindered_barrier() 从 ⟨cosφ⟩ 解 σ 再回代，_torsion_steps() 直接写 (2c+1)/3。
        两条路代数恒等，这里逐点对拍 —— 否则图上画的分布就不是采样器真正在抽的那个。
        """
        for c in (-0.5 + 1e-12, -0.4, 0.0, 0.25, 0.5, 0.68, 0.99):
            sigma, delta_e, p_trans = fjc.hindered_barrier(c)
            self.assertAlmostEqual(p_trans, (2.0 * c + 1.0) / 3.0, places=12, msg=f"c={c}")
            # ΔE 与 σ 互为反函数（位垒就是「gauche 的玻尔兹曼因子取对数」的定义）。
            # 比的是**相对**误差：c → COS_PHI_MIN 时 σ 能到 1e12，绝对小数位在那里没有意义
            self.assertAlmostEqual(math.exp(-delta_e) / sigma, 1.0, places=12, msg=f"c={c}")

    def test_energy_barrier_anchors(self):
        """两个教科书锚点：聚乙烯的那条 ΔE = ln 4，以及 c = 0 处与自由旋转链的接缝。"""
        pe = fjc.ModelParams.of("hindered", theta_deg=self.THETA_PE, cos_phi=0.5)
        fig = fjc.energy_figure(pe)
        self.assertEqual(fig["kind"], "torsion")
        # ⟨cosφ⟩ = 0.5 ⇒ σ = 1/4 ⇒ ΔE = ln 4 = 1.3863 kT —— 经典位阻因子 4
        self.assertAlmostEqual(fig["barrier"]["sigma"], 0.25, places=12)
        self.assertAlmostEqual(fig["barrier"]["delta_e_over_kt"], math.log(4.0), places=12)
        self.assertAlmostEqual(fig["barrier"]["p_trans"], 2.0 / 3.0, places=12)
        self.assertAlmostEqual(fig["cn"]["now"], 6.0, places=6)          # 聚乙烯的 C∞

        # 三态的权重加起来是 1，且 gauche± 对称、比 trans 低 ΔE
        lv = fig["levels"]
        self.assertEqual([x["phi_deg"] for x in lv], [0.0, 120.0, -120.0])
        self.assertAlmostEqual(sum(x["weight"] for x in lv), 1.0, places=12)
        self.assertAlmostEqual(lv[1]["weight"], lv[2]["weight"], places=12)
        self.assertAlmostEqual(lv[1]["u_over_kt"], fig["barrier"]["delta_e_over_kt"], places=12)
        self.assertEqual(lv[0]["u_over_kt"], 0.0)                        # trans 是能量零点

        # c = 0 ⇒ σ = 1 ⇒ ΔE = 0 ⇒ 三态等高 ⇒ 绕键自由旋转：这一点的 Cn 必须与
        # 自由旋转链**精确**重合（两条闭式在这一格给出同一个浮点数，不是差不多）
        zero = fjc.energy_figure(fjc.ModelParams.of("hindered", theta_deg=self.THETA_PE,
                                                    cos_phi=0.0))
        self.assertEqual(zero["barrier"]["delta_e_over_kt"], 0.0)
        # -log(1.0) 在 IEEE 下是 −0.0：数值上等于 0，但发到 JSON 里就是 "-0.0"，
        # 所以内核里那行收拾不是多余的，这里钉它
        self.assertEqual(math.copysign(1.0, zero["barrier"]["delta_e_over_kt"]), 1.0)
        self.assertFalse(zero["barrier"]["all_gauche"])
        self.assertEqual(zero["cn"]["now"], zero["cn"]["frc"])
        self.assertAlmostEqual(zero["levels"][0]["weight"], 1.0 / 3.0, places=12)

    def test_energy_cn_curve_is_the_closed_form(self):
        """第二张图的纵轴：Cn = Cn_FRC·(1 + 2e^(ΔE))/3，逐点现算而不是另抄一份。"""
        frc = fjc.ModelParams.of("frc", theta_deg=self.THETA_PE)
        cn_frc = fjc.model_kuhn_over_l(frc)
        fig = fjc.energy_figure(fjc.ModelParams.of("hindered", theta_deg=self.THETA_PE,
                                                   cos_phi=0.5))
        cn = fig["cn"]
        self.assertEqual(len(cn["delta_e"]), len(cn["values"]))
        self.assertEqual(len(cn["delta_e"]), fjc.ENERGY_CN_POINTS)
        self.assertAlmostEqual(cn["frc"], cn_frc, places=12)
        # 横轴从截断点起步（ΔE → −∞ 画不出来），右端封在 COS_PHI_MAX 那个位垒上
        self.assertEqual(cn["delta_e"][0], fjc.ENERGY_CN_DELTA_MIN)
        self.assertAlmostEqual(cn["delta_e"][-1], fig["delta_max"], places=12)
        for delta, value in zip(cn["delta_e"], cn["values"]):
            self.assertAlmostEqual(value, cn_frc * (1.0 + 2.0 * math.exp(delta)) / 3.0,
                                   places=6, msg=f"ΔE={delta}")
        # 单调：位垒越高（越偏爱 trans）链越硬
        self.assertEqual(cn["values"], sorted(cn["values"]))
        # 聚乙烯那个点是硬锚点，不是随手标的一格
        self.assertAlmostEqual(cn["pe_delta_e"], math.log(4.0), places=12)
        # 3 位小数而不是 6：这个点用的是目录里的默认键角 DEFAULT_THETA_DEG = 109.47
        # （截断值，不是 arccos(−1/3) 的 109.4712…），所以它落在 6 旁边而不是正好 6
        self.assertAlmostEqual(cn["pe_cn"], 6.0, places=3)

    def test_energy_all_gauche_endpoint_stays_finite_in_json(self):
        """⟨cosφ⟩ 压到下界：ΔE 真的是 −∞，但 JSON 里不许出现 Infinity。"""
        import json
        fig = fjc.energy_figure(fjc.ModelParams.of("hindered", theta_deg=self.THETA_PE,
                                                   cos_phi=fjc.COS_PHI_MIN))
        b = fig["barrier"]
        self.assertTrue(b["all_gauche"])
        self.assertIsNone(b["delta_e_over_kt"])     # 无穷用 null 表示
        self.assertIsNone(b["sigma"])
        self.assertEqual(b["p_trans"], 0.0)
        # 真值画不出来就取图画得出的最低那条线，但三个态仍然都在
        self.assertTrue(all(x["u_over_kt"] == fjc.ENERGY_CN_DELTA_MIN
                            for x in fig["levels"][1:]))
        # Cn 的下界：全部 gauche ⇒ (1+2e^ΔE)/3 → 1/3
        self.assertAlmostEqual(fig["cn"]["now"], fig["cn"]["frc"] / 3.0, places=9)
        text = json.dumps(fig, ensure_ascii=False)
        for bad in ("Infinity", "NaN", "-Infinity"):
            self.assertNotIn(bad, text)

    # 要逐点验的那几个 ⟨cosφ⟩。−0.5 是滑块下界（ΔE 截断成 −3），−0.25 是**负 ΔE**
    # 那一侧的代表 —— 井深差为负时 gauche 比 trans 低，是 a₃ 里那个绝对值的守卫。
    TORSION_CASES = (-0.5, -0.25, 0.25, 0.5, 0.68, 0.99)

    def test_energy_torsion_potential_wells_sit_on_the_three_states(self):
        """U(φ) 那条曲线：241 个点，井底**正好**落在采样器抽的那三个 φ 上。

        这一组钉的全是「图上的数就是内核里的数」——**不断言画法**：势垒多高是约定，
        换一个 TORSION_BARRIER_RATIO 下面这些断言一条都不该动。
        """
        grid = None
        for c in self.TORSION_CASES:
            fig = fjc.energy_figure(fjc.ModelParams.of(
                "hindered", theta_deg=self.THETA_PE, cos_phi=c))
            p, u = fig["phi_deg"], fig["potential"]
            if grid is None:
                grid = p
                # 1.5° 一格，而 0/±60/±120/±180 全是整格 —— 井底与极值点都落在格点上，
                # 图上那几个数不需要插值，测的也就是图上的数
                self.assertEqual(len(p), fjc.ENERGY_TORSION_POINTS)
                self.assertEqual((p[0], p[-1]), (-180.0, 180.0))
                self.assertEqual(len({round(p[i + 1] - p[i], 12)
                                      for i in range(len(p) - 1)}), 1)
                for anchor in (-180.0, -120.0, -60.0, 0.0, 60.0, 120.0, 180.0):
                    self.assertIn(anchor, p)
            else:
                self.assertEqual(p, grid)      # 网格与 ΔE 无关，六组共用同一条横轴
            self.assertEqual(len(u), len(p))

            # 三个井底：既要是**极小**（离散二阶差分 > 0），又要数值上等于 levels 里
            # 那个 u_over_kt —— 曲线上的点与读数行的数必须是同一个数。
            for lv in fig["levels"]:
                i = p.index(lv["phi_deg"])
                self.assertGreater(u[i - 1] + u[i + 1] - 2.0 * u[i], 0.0)
                self.assertAlmostEqual(u[i], lv["u_over_kt"], places=12)
            # trans 是能量零点。ΔE < 0 时它落在两个 gauche 井**上面**，但仍是极小 ——
            # 倒过来的是井深，不是井的位置
            self.assertEqual(u[p.index(0.0)], 0.0)
            # 井深差：U(±120°) − U(0°)，有限时与 barrier 里那个数是同一个数
            # （all_gauche 时 barrier 里是 None —— JSON 里不能出现 Infinity）
            d = u[p.index(120.0)] - u[p.index(0.0)]
            self.assertAlmostEqual(u[p.index(-120.0)] - u[p.index(0.0)], d, places=12)
            if fig["barrier"]["delta_e_over_kt"] is not None:
                self.assertAlmostEqual(d, fig["barrier"]["delta_e_over_kt"], places=12)
            # 纵轴范围就是数组自己的 max/min —— 由后端给，前端不自己挑
            self.assertEqual(fig["u_max"], max(u))
            self.assertEqual(fig["u_min"], min(u))
            # 纵轴那条换算说明：1 kT = R·T（T 用 DEFAULT_TEMPERATURE）= 2.479 kJ/mol
            self.assertAlmostEqual(fig["kt_in_kj_per_mol"],
                                   fjc.R_GAS * fjc.DEFAULT_TEMPERATURE / 1000.0, places=9)
            self.assertAlmostEqual(fig["kt_in_kj_per_mol"], 2.4789570296, places=9)

    def test_energy_torsion_barrier_is_the_stated_convention(self):
        """势垒高度是**约定**（TORSION_BARRIER_RATIO），井底不是 —— 这条测的是那个约定。

        gauche↔gauche 那处连接的是**两个等价的井**，「比更高的那个井高多少」在那里
        没有歧义，所以它有精确值 4|ΔE|；trans↔gauche 那处连接的两个井不等高（相差 ΔE），
        顶点又是 −α(sinφ + sin2φ) = 3a₃sin3φ 这个超越方程的根，没有闭式 ——
        所以那个数只从曲线上量，这里**不对它假造一个公式**。
        """
        for c in self.TORSION_CASES:
            fig = fjc.energy_figure(fjc.ModelParams.of(
                "hindered", theta_deg=self.THETA_PE, cos_phi=c))
            p, u, b = fig["phi_deg"], fig["potential"], fig["barrier"]
            # 传进来的 ΔE 是**截断过**的（⟨cosφ⟩ = −0.5 时真值是 −∞），
            # 约定与图上那个数都建在截断值上，测试也跟着它
            u_g = fig["levels"][1]["u_over_kt"]
            measured = max(v for q, v in zip(p, u) if q >= 120.0) - u_g
            self.assertEqual(measured, b["gg_over_kt"])          # 与图逐位一致
            self.assertAlmostEqual(measured, fjc.TORSION_BARRIER_RATIO * abs(u_g),
                                   places=12)
            # 绝对值那一项是必需的：ΔE < 0 时势垒高度不能跟着变负。下面两条钉住它 ——
            # 井底在下面（tg > 0），而另一条势垒**总**比 gauche↔gauche 那条低
            self.assertGreater(b["tg_over_kt"], 0.0)
            self.assertLess(b["tg_over_kt"], b["gg_over_kt"])

    def test_energy_torsion_potential_is_flat_at_the_frc_seam(self):
        """⟨cosφ⟩ = 0 ⇒ ΔE = 0 ⇒ 两个系数一起归零 ⇒ **整条曲线恒为 0**。

        这不是画不出来，是那一点真的没有位阻：图上这条平线与 model_h2() 里受阻旋转链
        精确退回自由旋转链说的是**同一件事**，图与闭式解在这一格必须同时归零。
        """
        fig = fjc.energy_figure(fjc.ModelParams.of("hindered", theta_deg=self.THETA_PE,
                                                   cos_phi=0.0))
        self.assertFalse(fig["barrier"]["all_gauche"])
        self.assertEqual(fig["barrier"]["delta_e_over_kt"], 0.0)
        # 逐点比数值 0（α·cos 那一串浮点上未必给出精确 0，见 hindered_potential 里
        # 那句「+ 0.0」的注释）
        self.assertEqual(max(abs(v) for v in fig["potential"]), 0.0)
        self.assertEqual((fig["u_min"], fig["u_max"]), (0.0, 0.0))
        self.assertEqual(fig["cn"]["now"], fig["cn"]["frc"])     # 闭式解那一侧的接缝

    def test_energy_torsion_potential_at_the_all_gauche_endpoint(self):
        """⟨cosφ⟩ 压到下界：ΔE 是 −∞，曲线用截断值画 —— 真值画不出来，但方向是对的。

        这一端最容易出的错是曲线整体翻号：井深差是负的，势垒高度不能跟着变负，
        否则 0°/±120° 会从极小点翻成极大点、**井底跑掉**。下面那两条二阶差分就是钉它。
        """
        import json
        fig = fjc.energy_figure(fjc.ModelParams.of("hindered", theta_deg=self.THETA_PE,
                                                   cos_phi=fjc.COS_PHI_MIN))
        b, p, u = fig["barrier"], fig["phi_deg"], fig["potential"]
        self.assertTrue(b["all_gauche"])
        self.assertTrue(all(math.isfinite(v) for v in u))
        i0, i120 = p.index(0.0), p.index(120.0)
        self.assertGreater(u[i0 - 1] + u[i0 + 1] - 2.0 * u[i0], 0.0)
        self.assertGreater(u[i120 - 1] + u[i120 + 1] - 2.0 * u[i120], 0.0)
        # gauche 井在 trans **下面** —— 这才是「gauche 更稳」的样子
        self.assertLess(u[i120], u[i0])
        self.assertAlmostEqual(u[i120], fig["levels"][1]["u_over_kt"], places=12)
        # 势垒建在截断值 −3 上，所以它是 4×3 = 12 而不是 ∞ —— 读数行要注明这一点，
        # 并排摆着那个「−∞」才不会被误读
        self.assertEqual(b["gg_over_kt"], fjc.TORSION_BARRIER_RATIO * 3.0)
        for bad in ("Infinity", "NaN", "-Infinity"):
            self.assertNotIn(bad, json.dumps(fig, ensure_ascii=False))

    def test_sampled_torsion_states_carry_the_barrier(self):
        """最关键的一条：ΔE 不只是个换算出来的数 —— 它真的预言了采样的态比例。

        从 _torsion_steps() 抽一大批，数出来的 trans 占比就是 p_trans、
        gauche : trans 的个数比就是 σ。这样「那个位垒是能量差」才落在真实采样上。
        """
        for c in (0.0, 0.25, 0.5, 0.9):
            sigma, delta_e, p_trans = fjc.hindered_barrier(c)
            phi = fjc._torsion_steps(np.random.default_rng(3), (400_000,), c)
            self.assertAlmostEqual(float(np.max(np.abs(phi))), 2.0 * math.pi / 3.0, places=12)
            n_trans = int(np.count_nonzero(phi == 0.0))
            n_gauche = phi.size - n_trans
            self.assertAlmostEqual(n_trans / phi.size, p_trans, delta=0.01, msg=f"c={c}")
            # gauche 有**两个**态（±120°），所以个数比是 2σ 而不是 σ ——
            # σ 是单个 gauche 态相对 trans 的因子。用对数比是为了在 c=0.9（σ≈0.036）那一格
            # 也拿相对误差衡量，而不是被绝对误差盖住
            self.assertAlmostEqual(math.log(n_gauche / n_trans), math.log(2.0 * sigma),
                                   delta=0.1, msg=f"c={c}：gauche(两个态)/trans 应当是 2σ")

    def test_energy_bending_is_the_boltzmann_distribution(self):
        """蠕虫状链那侧：采样器抽的 p(x) ∝ e^(κx) 就是 U/kT = κ(1−cosθ) 的玻尔兹曼分布。"""
        mp = fjc.ModelParams.of("wlc", p=10.0)
        fig = fjc.energy_figure(mp)
        self.assertEqual(fig["kind"], "bending")
        kappa, p_over_l, corr = fig["kappa"], fig["p_over_l"], fig["corr"]
        self.assertAlmostEqual(kappa, fjc._wlc_kappa(math.exp(-1.0 / 10.0)), places=12)
        # ⟨cosθ⟩ = L(κ) = e^(−l/p) 是构造出来的恒等式，页面上指纹卡的 corr[1] 就是它
        self.assertAlmostEqual(fjc._langevin(kappa), corr, places=12)
        self.assertAlmostEqual(corr, math.exp(-1.0 / p_over_l), places=12)

        # 势的值域是 0（相邻键对齐，cos θ = 1）… 2κ（完全反向，cos θ = −1），
        # 且就是 cos 的线性函数 —— xs 从 −1 排到 +1，所以首点是最高那级
        cos = np.asarray(fig["cos"])
        potential = np.asarray(fig["potential"])
        self.assertAlmostEqual(float(cos[0]), -1.0, places=12)
        self.assertAlmostEqual(fig["u_max"], 2.0 * kappa, places=12)
        self.assertAlmostEqual(float(potential[0]), fig["u_max"], places=9)
        self.assertAlmostEqual(float(potential[-1]), 0.0, places=12)
        np.testing.assert_allclose(potential, kappa * (1.0 - cos), rtol=1e-12)

        # 密度归一化、且在 cosθ = 1（相邻键完全对齐）那一端取最大值 κ。
        # 3 位小数是**梯形法**在这 400 点网格上的误差（O(h²κ²) ≈ 2e-4），不是密度的误差
        density = np.asarray(fig["density"])
        area = float(np.sum(np.diff(cos) * (density[1:] + density[:-1]) / 2.0))
        self.assertAlmostEqual(area, 1.0, places=3)
        self.assertAlmostEqual(float(density[-1]), kappa, places=6)
        self.assertEqual(int(np.argmax(density)), density.size - 1)

        # 真正把话钉死的一条：从采样器抽出来的步矢量，相邻两个的余弦均值 = corr
        steps = fjc._step_vectors_model(mp, np.random.default_rng(4), 2000, 60)
        adjacent = np.sum(steps[:, :-1] * steps[:, 1:], axis=-1)
        self.assertTrue(float(adjacent.min()) >= -1.0 and float(adjacent.max()) <= 1.0)
        self.assertAlmostEqual(float(adjacent.mean()), corr, delta=0.005)

    def test_energy_persistence_length_slope(self):
        """第三、四张图的纵轴：p/l 与渐近 κ ≈ p/l + 1/2 —— 大 κ 端才成立。"""
        fig = fjc.energy_figure(fjc.ModelParams.of("wlc", p=100.0))
        ks = np.asarray(fig["kappa_scan"])
        pl = np.asarray(fig["pl_scan"])
        self.assertEqual(len(ks), fjc.ENERGY_KAPPA_POINTS)
        self.assertEqual(len(pl), len(ks))
        self.assertTrue(np.all(np.diff(ks) > 0.0))
        # p/l = −1/ln L(κ)：就是 wlc_bending 的反函数，且与 κ 同向
        np.testing.assert_allclose(
            pl, [-1.0 / math.log(fjc._langevin(float(k))) for k in ks], rtol=1e-12)
        # 渐近线只发 κ ≥ 1 那一段（再往左它给不出正数，画在对数轴上会变成 NaN）
        asym = np.asarray(fig["pl_asymptote"])
        self.assertTrue(np.all(asym[:, 0] >= 1.0))
        np.testing.assert_allclose(asym[:, 1], asym[:, 0] - 0.5, rtol=1e-12)
        # 当前点的 κ 与 p/l 差得就是那 1/2（p/l = 100 时相对误差 8e-6）
        self.assertAlmostEqual(fig["kappa"] - fig["p_over_l"], 0.5, delta=0.005)
        self.assertAlmostEqual(fig["p_over_l"], 100.0, places=12)

    def test_wlc_limits(self):
        """蠕虫状链的两端：L≫p 回到 ⟨h²⟩ ≈ 2pL；L≪p 退化成刚杆 ⟨h²⟩ ≈ L²。"""
        p = 10.0
        mp = fjc.ModelParams.of("wlc", p=p)
        # 长链端：与 2pL 的相对差应当只剩 p/L 那一阶（不是随便一个「差不多」）
        for n in (1000, 100000):
            ratio = fjc.model_h2(mp, n, 1.0) / (2 * p * n)
            self.assertLess(abs(ratio - 1.0), 2.0 * p / n)
        # 短链端：L/p ≪ 1 时退化成刚杆 ⟨h²⟩ ≈ L²（差的是 (1/3)(L/p) 那一阶）。
        # 用 p=300 把 L/p 压到 1/300；p=10 时 L/p=0.1，偏差有 3%，故意不在这里断言。
        stiff = fjc.ModelParams.of("wlc", p=300.0)
        ratio = fjc.model_h2(stiff, 1, 1.0)
        self.assertLess(ratio, 1.0)
        self.assertAlmostEqual(ratio, 1.0, places=2)
        # 链越硬（p 越大）⟨h²⟩ 越大，且永远不超过刚杆的 L²
        n, l = 200, 1.0
        vals = [fjc.model_h2(fjc.ModelParams.of("wlc", p=pp), n, l) for pp in (1.0, 10.0, 100.0)]
        self.assertEqual(vals, sorted(vals))
        self.assertLess(vals[-1], (n * l) ** 2)

    def test_sampler_steps_and_correlations(self):
        """采样器：每步长度精确为 l；方向关联恰好是模型要求的那个数。"""
        n, chains = 120, 400
        cases = (
            (fjc.ModelParams.of("frc", theta_deg=self.THETA_PE), -math.cos(math.radians(self.THETA_PE))),
            (fjc.ModelParams.of("hindered", theta_deg=self.THETA_PE, cos_phi=0.5),
             -math.cos(math.radians(self.THETA_PE))),
            (fjc.ModelParams.of("wlc", p=10.0), math.exp(-0.1)),
        )
        for mp, want_c1 in cases:
            r = fjc.random_chain(n, 1.0, seed=11, chains=chains, max_n=None,
                                 max_chains=None, points_max=None, params=mp)
            u = np.diff(r.points, axis=1)
            norms = np.linalg.norm(u, axis=-1)
            self.assertAlmostEqual(float(norms.mean()), 1.0, places=12, msg=mp.key)
            c1 = float(np.mean(np.sum(u[:, :-1] * u[:, 1:], axis=-1)))
            self.assertAlmostEqual(c1, want_c1, delta=0.02, msg=f"{mp.key} 的 ⟨u₀·u₁⟩")
        # 受累旋转链的二阶关联还要看内旋转：⟨u₀·u₂⟩ = c² + s²⟨cosφ⟩
        c = -math.cos(math.radians(self.THETA_PE))
        r = fjc.random_chain(n, 1.0, seed=11, chains=chains, max_n=None,
                             max_chains=None, points_max=None,
                             params=fjc.ModelParams.of("hindered", theta_deg=self.THETA_PE,
                                                       cos_phi=0.5))
        u = np.diff(r.points, axis=1)
        c2 = float(np.mean(np.sum(u[:, :-2] * u[:, 2:], axis=-1)))
        self.assertAlmostEqual(c2, c * c + (1 - c * c) * 0.5, delta=0.02)

    def test_sampled_R2_matches_every_model(self):
        """四个模型各采一批链，⟨R²⟩ 必须落在模型 ⟨h²⟩ 的 3σ 内（seed 写死）。"""
        cases = (fjc.ModelParams.of("fjc"),
                 fjc.ModelParams.of("frc", theta_deg=self.THETA_PE),
                 fjc.ModelParams.of("hindered", theta_deg=self.THETA_PE, cos_phi=0.5),
                 fjc.ModelParams.of("wlc", p=10.0))
        for mp in cases:
            for n in (20, 100):
                r = fjc.random_chain(n, 1.0, seed=7, chains=4000, max_n=None,
                                     max_chains=None, points_max=None, params=mp)
                dev = (r.R2_mean - r.R2_theory) / r.R2_theory
                self.assertLess(abs(dev), 3 * r.R2_rel_se,
                                msg=f"{mp.key} n={n}：偏差 {dev:.4f} 超出 3σ={3 * r.R2_rel_se:.4f}")

    def test_sampler_and_sweep_share_the_stream_for_every_model(self):
        """同一条随机流上，_sweep_endpoints 与 random_chain 的末端矢量逐位相同。"""
        for mp in (fjc.ModelParams.of("fjc"), fjc.ModelParams.of("frc"),
                   fjc.ModelParams.of("hindered"), fjc.ModelParams.of("wlc", p=5.0)):
            a = fjc.random_chain(30, 1.0, seed=5, chains=4, params=mp)
            b = fjc._sweep_endpoints(np.random.default_rng(5), 30, 4, mp)
            self.assertTrue(np.array_equal(a.R, b), msg=mp.key)
            # 3D 视图要在链上标出「每过一个 Kuhn 长度」，所以这条也得跟着模型走
            self.assertAlmostEqual(a.kuhn_length, fjc.model_kuhn_length(mp, 1.0), places=12,
                                   msg=mp.key)

    def test_sweep_reference_slope_is_model_aware(self):
        """斜率参考值：前三个模型恒为 1，WLC 由它自己的曲线定（不是 1）。"""
        plain = fjc.sweep([10, 100, 1000], 1.0, seed=1, chains=500,
                          params=fjc.ModelParams.of("frc"))
        self.assertEqual(plain.slope_ref, 1.0)
        self.assertFalse(any("这不该发生" in w for w in plain.warnings))

        wlc = fjc.sweep([10, 100, 1000], 1.0, seed=1, chains=500,
                        params=fjc.ModelParams.of("wlc", p=10.0))
        self.assertGreater(wlc.slope_ref, 1.05)      # 短链端是 L²，斜率明显大于 1
        self.assertLess(wlc.slope_ref, 1.4)
        self.assertFalse(any("这不该发生" in w for w in wlc.warnings))
        self.assertEqual(wlc.model, "wlc")

    def test_model_params_validation(self):
        """非法参数要报中文错，而不是崩掉或悄悄算错。"""
        for bad in (dict(model="nope"), dict(theta_deg=0.0), dict(theta_deg=180.0),
                    dict(cos_phi=1.0), dict(cos_phi=-0.9), dict(p=0.0), dict(p=1e6)):
            with self.assertRaises(fjc.FJCInputError):
                fjc.ModelParams.of(**bad)
        self.assertEqual(fjc.ModelParams.of("FRC  ").key, "frc")      # 大小写与空格都认
        self.assertEqual(fjc.ModelParams.of("hindered", cos_phi="0.25").cos_phi, 0.25)

    def test_model_catalog_is_json_safe(self):
        import json
        rows = fjc.model_catalog()
        # 目录 = 四个解析模型 + 自回避行走。SAW **不在 MODEL_KEYS 里**（它进不去
        # 公式层，理由见 fjc_core「自回避行走」那一节），但它是用户能选的模型，
        # 所以必须出现在同一份目录里 —— 前端只认这一份。
        self.assertEqual([r["key"] for r in rows], list(fjc.MODEL_KEYS) + ["saw"])
        json.dumps(rows, ensure_ascii=False)
        for r in rows:
            self.assertTrue(r["label"] and r["formula"] and r["note"])
            for spec in r["params"].values():
                self.assertIn("default", spec)
                self.assertIn("used", spec)
        # 每个模型至少用一个参数：FJC 一个都不用，其余各用它自己那个
        self.assertFalse(any(s["used"] for s in rows[0]["params"].values()))
        self.assertTrue(rows[1]["params"]["theta_deg"]["used"])
        self.assertTrue(rows[3]["params"]["p"]["used"])

        # analytic 这一位就是「哪些算得了」的唯一真源：四个解析模型有闭式 ⟨h²⟩，
        # 自回避行走没有 —— 前端靠它收起不适用的卡片，这里把它钉住。
        saw = rows[-1]
        self.assertTrue(all(r["analytic"] is True for r in rows[:-1]))
        self.assertIs(saw["analytic"], False)
        self.assertFalse(any(s["used"] for s in saw["params"].values()))
        # ν 得进公式，且只能是 SAW_NU（别手抄一个数字进来）
        self.assertIn(f"{fjc.SAW_NU:.7f}", saw["formula"])

    def test_fingerprint_curves(self):
        """模型指纹：关联的解析锚点、Cn 的收敛、四个模型的对比行。"""
        c = -math.cos(math.radians(self.THETA_PE))
        base = fjc.model_fingerprint(fjc.ModelParams.of("fjc"), points=8)
        self.assertEqual(base["corr"][0], 1.0)
        self.assertTrue(all(v == 0.0 for v in base["corr"][1:]), "FJC 隔一段就该不相关")
        self.assertEqual([r["key"] for r in base["compare"]], list(fjc.MODEL_KEYS))
        self.assertEqual(len(base["cn"]), len(base["ns"]))

        frc = fjc.model_fingerprint(fjc.ModelParams.of("frc", theta_deg=self.THETA_PE), points=8)
        for k in (1, 2, 5):
            self.assertAlmostEqual(frc["corr"][k], c ** k, places=12)

        wlc = fjc.model_fingerprint(fjc.ModelParams.of("wlc", p=10.0), points=8)
        self.assertAlmostEqual(wlc["corr"][3], math.exp(-0.3), places=12)

        # 受阻旋转链的关联是两根之和：⟨cosφ⟩=0 时必须与自由旋转链逐点相同
        h0 = fjc.model_fingerprint(
            fjc.ModelParams.of("hindered", theta_deg=self.THETA_PE, cos_phi=0.0), points=8)
        for k in range(6):
            self.assertAlmostEqual(h0["corr"][k], c ** k, places=9)

        # Cn 随 n 单调趋向 C∞（有限 n 一律更小），且与 l 无关（只由模型参数定）
        for key in fjc.MODEL_KEYS:
            cn = fjc.model_fingerprint(fjc.ModelParams.of(key), points=10)["cn"]
            self.assertLessEqual(max(cn), cn[-1] + 1e-9, f"{key} 的 Cn 应当在末尾最大")
        mp = fjc.ModelParams.of("wlc", p=10.0)
        self.assertAlmostEqual(fjc.model_cn(mp, 100, 3.0), fjc.model_cn(mp, 100, 1.0), places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
