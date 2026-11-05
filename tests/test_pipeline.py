"""Unit tests (stdlib unittest; run with ``python -m unittest discover tests``)."""
import unittest

import numpy as np

from src import config as C
from src.defenses import DEFENSES, buflo, pad_to_mtu, timing_jitter
from src.features import FEATURE_GROUPS, FEATURE_NAMES, extract_features
from src.traffic_generator import generate_flow


class TestFeatures(unittest.TestCase):
    def test_hand_computed_values(self):
        # 4 packets, 10 ms apart, alternating direction, sizes known
        t = np.array([0.0, 0.01, 0.02, 0.03])
        s = np.array([100, 1500, 100, 1500])
        d = np.array([0, 1, 0, 1], dtype=np.int8)
        f = extract_features(t, s, d)
        self.assertEqual(f["n_pkts"], 4)
        self.assertEqual(f["n_bytes"], 3200)
        self.assertEqual(f["fwd_bytes"], 200)
        self.assertEqual(f["bwd_bytes"], 3000)
        self.assertAlmostEqual(f["duration"], 0.03)
        self.assertAlmostEqual(f["size_mean"], 800)
        self.assertAlmostEqual(f["frac_small"], 0.5)
        self.assertAlmostEqual(f["frac_large"], 0.5)
        self.assertAlmostEqual(f["iat_regularity"], 1.0)  # perfectly periodic
        self.assertAlmostEqual(f["dir_change_rate"], 1.0)
        self.assertAlmostEqual(f["bwd_fwd_byte_ratio"], 3001 / 201)

    def test_feature_vector_shape_and_finiteness(self):
        self.assertEqual(len(FEATURE_NAMES), len(set(FEATURE_NAMES)))
        self.assertEqual(sum(map(len, FEATURE_GROUPS.values())), len(FEATURE_NAMES))
        f = extract_features(np.array([0.0]), np.array([60]), np.array([0], dtype=np.int8))  # degenerate flow
        self.assertEqual(list(f), FEATURE_NAMES)
        self.assertTrue(np.all(np.isfinite(list(f.values()))))


class TestGenerator(unittest.TestCase):
    def test_every_class_produces_valid_traces(self):
        rng = np.random.default_rng(0)
        for label in C.CLASSES:
            for _ in range(5):
                t, s, d, sub, _ = generate_flow(label, rng, C.BROADBAND)
                self.assertGreaterEqual(t.size, 4)
                self.assertTrue(np.all(np.diff(t) >= 0), label)
                self.assertEqual(t[0], 0.0)
                self.assertLess(t[-1], C.WINDOW_S)
                self.assertTrue(np.all((s >= C.MIN_PKT) & (s <= C.MTU)), label)
                self.assertTrue(set(np.unique(d)) <= {0, 1})
                self.assertIsInstance(sub, str)

    def test_determinism(self):
        a = generate_flow("video", np.random.default_rng(7), C.BROADBAND)
        b = generate_flow("video", np.random.default_rng(7), C.BROADBAND)
        for x, y in zip(a[:3], b[:3]):
            np.testing.assert_array_equal(x, y)


class TestDefenses(unittest.TestCase):
    def setUp(self):
        self.t, self.s, self.d, _, _ = generate_flow("web", np.random.default_rng(3), C.BROADBAND)

    def test_padding(self):
        _, s2, _, delay = pad_to_mtu(self.t, self.s, self.d)
        self.assertTrue(np.all(s2 == C.MTU))
        self.assertEqual(delay, 0.0)

    def test_jitter_preserves_packets(self):
        t2, s2, d2, delay = timing_jitter(self.t, self.s, self.d, np.random.default_rng(0))
        self.assertEqual(t2.size, self.t.size)
        self.assertEqual(s2.sum(), self.s.sum())
        self.assertTrue(0 < delay <= C.JITTER_MAX_S)

    def test_buflo_is_constant_rate(self):
        t2, s2, d2, delay = buflo(self.t, self.s, self.d)
        for dirn in (0, 1):
            gaps = np.diff(t2[d2 == dirn])
            np.testing.assert_allclose(gaps, C.BUFLO_RHO_S, atol=1e-9)
        self.assertTrue(np.all(s2 == C.MTU))
        self.assertGreaterEqual(delay, 0.0)
        self.assertGreaterEqual(t2[-1], C.BUFLO_MIN_S - 2 * C.BUFLO_RHO_S)

    def test_all_defenses_return_four_values(self):
        for fn in DEFENSES.values():
            self.assertEqual(len(fn(self.t, self.s, self.d, np.random.default_rng(0))), 4)


class TestLearnability(unittest.TestCase):
    def test_small_dataset_beats_chance(self):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        rng = np.random.default_rng(1)
        X, y = [], []
        for label in C.CLASSES:
            for _ in range(25):
                t, s, d, _, _ = generate_flow(label, rng, C.BROADBAND)
                X.append(list(extract_features(t, s, d).values()))
                y.append(label)
        acc = cross_val_score(RandomForestClassifier(100, random_state=0), np.array(X), y, cv=3).mean()
        self.assertGreater(acc, 0.7)


if __name__ == "__main__":
    unittest.main()
