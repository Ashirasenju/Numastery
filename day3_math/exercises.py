"""
DAY 3 — Universal Functions & Broadcasting
===========================================
Topics: ufuncs, reduce/accumulate/outer, broadcasting rules,
        vectorised operations, custom ufuncs with np.frompyfunc.

Run: pytest day3_math/test_exercises.py -v
"""

import numpy as np


# ── 3.1  Universal Functions (ufuncs) ────────────────────────────────────────

def running_sum(arr: np.ndarray) -> np.ndarray:
    """
    Return the cumulative (running) sum of a 1-D array.
    Use np.add.accumulate — not np.cumsum (same result, different call).
    """
    raise NotImplementedError


def outer_product_sum(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the outer product of vectors `a` and `b`, then return the
    sum of all elements.  Use np.multiply.outer.
    """
    raise NotImplementedError


def log_sum_exp(arr: np.ndarray) -> float:
    """
    Numerically stable log(sum(exp(arr))).
    Formula: max(arr) + log(sum(exp(arr - max(arr)))).
    Implement using ufuncs/vectorised ops only.
    """
    raise NotImplementedError


def custom_relu_ufunc(arr: np.ndarray) -> np.ndarray:
    """
    Apply ReLU (max(x, 0)) element-wise.
    Build a ufunc via np.frompyfunc, then call it.
    """
    raise NotImplementedError


# ── 3.2  Broadcasting ────────────────────────────────────────────────────────

def row_normalize(matrix: np.ndarray) -> np.ndarray:
    """
    Normalise each row of a 2-D matrix so it sums to 1.
    Must use broadcasting (no loop).
    Return float64 array.
    """
    raise NotImplementedError


def pairwise_l2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Given A of shape (m, d) and B of shape (n, d), return the (m, n) matrix
    of Euclidean distances between every pair of rows.
    Use broadcasting — NOT a nested loop.
    Hint: ||a - b||² = ||a||² + ||b||² - 2 a·bᵀ
    """
    raise NotImplementedError


def add_bias(X: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """
    X has shape (N, D), bias has shape (D,).
    Return X + bias using broadcasting.
    """
    raise NotImplementedError


def outer_subtract(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Given 1-D arrays `a` (length m) and `b` (length n), return an (m, n)
    matrix where result[i, j] = a[i] - b[j].
    Use broadcasting — no loops, no np.subtract.outer.
    """
    raise NotImplementedError


# ── 3.3  Vectorised logic ────────────────────────────────────────────────────

def softmax(arr: np.ndarray) -> np.ndarray:
    """
    Compute the softmax of a 1-D array in a numerically stable way.
    softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))
    """
    raise NotImplementedError


def moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Return the simple moving average of `arr` using a window of size `window`.
    Output length = len(arr) - window + 1.
    Use np.cumsum — not np.convolve (though both work).
    """
    raise NotImplementedError


def polynomial_eval(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Evaluate a polynomial at every point in `x`.
    `coeffs` = [a0, a1, a2, …] so poly(x) = a0 + a1*x + a2*x² + …
    Use np.polyval (note: np.polyval expects highest-degree first, so flip).
    """
    raise NotImplementedError
