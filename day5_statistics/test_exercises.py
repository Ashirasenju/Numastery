"""
Tests for Day 5 — Statistics & Random Number Generation
Run: pytest day5_statistics/test_exercises.py -v
"""

import numpy as np
import pytest
from day5_statistics.exercises import (
    summary_stats, weighted_average, z_score, correlation_matrix,
    histogram_mode, digitize_labels,
    reproducible_sample, bootstrap_mean_ci, random_walk,
    estimate_pi, option_price_mc,
)


class TestDescriptiveStats:

    def test_summary_stats_keys(self):
        arr = np.arange(10, dtype=float)
        d = summary_stats(arr)
        for k in ('mean', 'std', 'min', 'max', 'median', 'q25', 'q75'):
            assert k in d

    def test_summary_stats_mean(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d = summary_stats(arr)
        assert d['mean'] == pytest.approx(3.0)

    def test_summary_stats_std_ddof1(self):
        arr = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        d = summary_stats(arr)
        assert d['std'] == pytest.approx(np.std(arr, ddof=1))

    def test_summary_stats_median(self):
        arr = np.array([1.0, 3.0, 5.0])
        d = summary_stats(arr)
        assert d['median'] == pytest.approx(3.0)

    def test_summary_stats_quartiles(self):
        arr = np.arange(1, 101, dtype=float)
        d = summary_stats(arr)
        assert d['q25'] == pytest.approx(np.percentile(arr, 25))
        assert d['q75'] == pytest.approx(np.percentile(arr, 75))

    def test_weighted_average_equal_weights(self):
        values = np.array([1.0, 2.0, 3.0])
        weights = np.array([1.0, 1.0, 1.0])
        assert weighted_average(values, weights) == pytest.approx(2.0)

    def test_weighted_average_skewed(self):
        values = np.array([0.0, 10.0])
        weights = np.array([1.0, 0.0])
        assert weighted_average(values, weights) == pytest.approx(0.0)

    def test_z_score_mean_zero(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = z_score(arr)
        assert pytest.approx(out.mean(), abs=1e-10) == 0.0

    def test_z_score_std_one(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = z_score(arr)
        assert pytest.approx(out.std(ddof=1)) == 1.0

    def test_correlation_matrix_shape(self):
        X = np.random.rand(50, 4)
        out = correlation_matrix(X)
        assert out.shape == (4, 4)

    def test_correlation_matrix_diagonal_ones(self):
        X = np.random.rand(50, 4)
        out = correlation_matrix(X)
        assert np.allclose(np.diag(out), 1.0)

    def test_correlation_matrix_symmetric(self):
        X = np.random.rand(30, 3)
        out = correlation_matrix(X)
        assert np.allclose(out, out.T)

    def test_correlation_matrix_range(self):
        X = np.random.rand(30, 3)
        out = correlation_matrix(X)
        assert np.all(out >= -1.0 - 1e-10)
        assert np.all(out <= 1.0 + 1e-10)


class TestHistogramsBinning:

    def test_histogram_mode_returns_float(self):
        arr = np.concatenate([np.ones(100), np.zeros(10)])
        result = histogram_mode(arr, bins=5)
        assert isinstance(result, float)

    def test_histogram_mode_peak(self):
        rng = np.random.default_rng(0)
        arr = np.concatenate([rng.normal(10, 0.1, 1000), rng.normal(0, 0.1, 10)])
        result = histogram_mode(arr, bins=50)
        assert abs(result - 10.0) < 1.0

    def test_digitize_labels_basic(self):
        arr = np.array([0.5, 1.5, 2.5, 3.5])
        bins = np.array([1.0, 2.0, 3.0])
        out = digitize_labels(arr, bins)
        assert np.array_equal(out, [1, 2, 3, 4])

    def test_digitize_labels_dtype(self):
        arr = np.array([0.5, 2.5])
        bins = np.array([1.0, 2.0])
        out = digitize_labels(arr, bins)
        assert out.dtype in (np.int32, np.int64, np.intp)


class TestRNG:

    def test_reproducible_sample_length(self):
        out = reproducible_sample(100)
        assert len(out) == 100

    def test_reproducible_sample_deterministic(self):
        a = reproducible_sample(50, seed=7)
        b = reproducible_sample(50, seed=7)
        assert np.array_equal(a, b)

    def test_reproducible_sample_different_seeds(self):
        a = reproducible_sample(50, seed=1)
        b = reproducible_sample(50, seed=2)
        assert not np.array_equal(a, b)

    def test_bootstrap_ci_contains_mean(self):
        rng = np.random.default_rng(99)
        data = rng.normal(5.0, 1.0, 200)
        lo, hi = bootstrap_mean_ci(data, n_bootstrap=3000, ci=0.95, seed=0)
        assert lo < data.mean() < hi

    def test_bootstrap_ci_order(self):
        data = np.arange(50, dtype=float)
        lo, hi = bootstrap_mean_ci(data)
        assert lo < hi

    def test_random_walk_starts_at_zero(self):
        out = random_walk(100)
        assert out[0] == 0

    def test_random_walk_length(self):
        out = random_walk(50)
        assert len(out) == 51

    def test_random_walk_steps_pm1(self):
        out = random_walk(100, seed=5)
        diffs = np.diff(out)
        assert np.all(np.abs(diffs) == 1)


class TestMonteCarlo:

    def test_estimate_pi_close(self):
        result = estimate_pi(n_samples=2_000_000, seed=0)
        assert abs(result - np.pi) < 0.01

    def test_estimate_pi_range(self):
        result = estimate_pi(n_samples=100_000, seed=1)
        assert 2.9 < result < 3.3

    def test_option_price_mc_positive(self):
        price = option_price_mc(100, 100, 0.05, 0.2, 1.0)
        assert price > 0

    def test_option_price_mc_deep_itm(self):
        """Deep in-the-money option should have price close to intrinsic."""
        # S0=200, K=100 deep ITM: price ≈ (200-100)*exp(-r*T) roughly
        price = option_price_mc(200, 100, 0.05, 0.01, 1.0, n_paths=500_000)
        assert price > 90

    def test_option_price_mc_deep_otm(self):
        """Deep out-of-the-money option should have price close to 0."""
        price = option_price_mc(50, 200, 0.05, 0.2, 1.0, n_paths=500_000)
        assert price < 1.0

    def test_option_price_mc_deterministic(self):
        p1 = option_price_mc(100, 100, 0.05, 0.2, 1.0, seed=7)
        p2 = option_price_mc(100, 100, 0.05, 0.2, 1.0, seed=7)
        assert p1 == pytest.approx(p2)
