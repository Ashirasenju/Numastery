"""
DAY 5 — Statistics & Random Number Generation
==============================================
Topics: descriptive stats, histograms, percentiles, covariance,
        np.random (legacy & Generator API), distributions, seeds,
        Monte Carlo methods.

Run: pytest day5_statistics/test_exercises.py -v
"""

import numpy as np


# ── 5.1  Descriptive Statistics ──────────────────────────────────────────────

def summary_stats(arr: np.ndarray) -> dict:
    """
    Return a dict with keys: mean, std, min, max, median, q25, q75.
    All values are Python floats.
    Use np.mean, np.std (ddof=1), np.min, np.max, np.median, np.percentile.
    """
    raise NotImplementedError


def weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Compute the weighted average:  sum(values * weights) / sum(weights).
    Use np.average.
    """
    raise NotImplementedError


def z_score(arr: np.ndarray) -> np.ndarray:
    """
    Standardise `arr` to zero mean and unit variance (z-score normalisation).
    Use ddof=1 for the standard deviation.
    """
    raise NotImplementedError


def correlation_matrix(X: np.ndarray) -> np.ndarray:
    """
    Return the Pearson correlation matrix for a (n_samples, n_features)
    data matrix X.  Use np.corrcoef (rows are variables).
    """
    raise NotImplementedError


# ── 5.2  Histograms & Binning ────────────────────────────────────────────────

def histogram_mode(arr: np.ndarray, bins: int = 10) -> float:
    """
    Return the midpoint of the most-populated bin in a histogram.
    Use np.histogram.
    """
    raise NotImplementedError


def digitize_labels(arr: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """
    Assign each element of `arr` a bin label (1-indexed) according to `bins`.
    Use np.digitize.
    """
    raise NotImplementedError


# ── 5.3  Random Number Generation ────────────────────────────────────────────

def reproducible_sample(n: int, seed: int = 42) -> np.ndarray:
    """
    Return `n` samples from a standard normal distribution using the
    modern Generator API (np.random.default_rng) with the given seed.
    """
    raise NotImplementedError


def bootstrap_mean_ci(data: np.ndarray, n_bootstrap: int = 2000,
                      ci: float = 0.95, seed: int = 0) -> tuple:
    """
    Estimate a confidence interval for the mean using bootstrap resampling.
    Steps:
      1. Draw `n_bootstrap` samples of size len(data) with replacement.
      2. Compute the mean of each resample.
      3. Return (lower, upper) percentiles for the CI.
    Use np.random.default_rng(seed).
    """
    raise NotImplementedError


def random_walk(n_steps: int, seed: int = 0) -> np.ndarray:
    """
    Simulate a 1-D random walk of `n_steps` steps (±1 each step).
    Return the cumulative position array of length n_steps + 1,
    starting at 0.
    Use np.random.default_rng(seed).choice or integers.
    """
    raise NotImplementedError


# ── 5.4  Monte Carlo ─────────────────────────────────────────────────────────

def estimate_pi(n_samples: int = 1_000_000, seed: int = 42) -> float:
    """
    Estimate π using Monte Carlo: draw random points in [-1,1]² and
    count how many fall inside the unit circle.
    π ≈ 4 * (inside count) / n_samples
    Use np.random.default_rng(seed).
    """
    raise NotImplementedError


def option_price_mc(S0: float, K: float, r: float, sigma: float,
                    T: float, n_paths: int = 100_000,
                    seed: int = 42) -> float:
    """
    Price a European call option using Monte Carlo (Black-Scholes model).
    S_T = S0 * exp((r - 0.5*sigma²)*T + sigma*sqrt(T)*Z),  Z ~ N(0,1)
    Price = exp(-r*T) * mean(max(S_T - K, 0))
    Use np.random.default_rng(seed).standard_normal.
    """
    raise NotImplementedError
