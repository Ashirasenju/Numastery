"""
DAY 2 — Indexing, Slicing & Masking
=====================================
Topics: basic slicing, advanced (fancy) indexing, boolean masks,
        np.where, np.nonzero, index tricks.

Run: pytest day2_indexing/test_exercises.py -v
"""

import numpy as np


# ── 2.1  Basic Slicing ───────────────────────────────────────────────────────

def extract_submatrix(arr: np.ndarray, row_start: int, row_end: int,
                      col_start: int, col_end: int) -> np.ndarray:
    """
    Return the sub-array arr[row_start:row_end, col_start:col_end].
    The returned object must be a VIEW, not a copy.
    """
    raise NotImplementedError


def every_other_row(arr: np.ndarray) -> np.ndarray:
    """Return every second row (0, 2, 4, …) of a 2-D array as a view."""
    raise NotImplementedError


def reverse_columns(arr: np.ndarray) -> np.ndarray:
    """Return `arr` with its columns in reversed order (view, not copy)."""
    raise NotImplementedError


def diagonal_sum(arr: np.ndarray) -> float:
    """
    Return the sum of the main diagonal elements of a square 2-D array.
    Use np.trace or np.diag — no explicit loops.
    """
    raise NotImplementedError


# ── 2.2  Fancy Indexing ──────────────────────────────────────────────────────

def select_rows(arr: np.ndarray, indices: list) -> np.ndarray:
    """
    Return the rows of `arr` at `indices`, in that order.
    Result must be a COPY (fancy indexing always copies).
    """
    raise NotImplementedError


def set_diagonal(arr: np.ndarray, value: float) -> np.ndarray:
    """
    Return a copy of `arr` with its main diagonal set to `value`.
    Do not use a loop; use np.fill_diagonal or advanced indexing.
    """
    raise NotImplementedError


def scatter_add(base: np.ndarray, indices: np.ndarray,
                values: np.ndarray) -> np.ndarray:
    """
    Return a copy of `base` where values[i] is ADDED to base[indices[i]]
    for each i.  Handle repeated indices correctly (use np.add.at).
    """
    raise NotImplementedError


# ── 2.3  Boolean Masking ─────────────────────────────────────────────────────

def clamp(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """
    Return a copy of `arr` where values below `lo` are set to `lo` and
    values above `hi` are set to `hi`.  Use np.clip.
    """
    raise NotImplementedError


def replace_outliers(arr: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """
    Return a copy of `arr` (float64) where any element more than `sigma`
    standard deviations from the mean is replaced by the mean.
    """
    raise NotImplementedError


def mask_nan_inf(arr: np.ndarray) -> np.ndarray:
    """
    Return a copy of `arr` where NaN and Inf values are replaced by 0.
    Use np.isfinite.
    """
    raise NotImplementedError


# ── 2.4  np.where & np.nonzero ───────────────────────────────────────────────

def sign_array(arr: np.ndarray) -> np.ndarray:
    """
    Return an array of the same shape where:
      positive values → +1, negative values → -1, zero → 0.
    Use np.where (or np.sign).
    """
    raise NotImplementedError


def first_nonzero_index(arr: np.ndarray) -> int:
    """
    Return the index of the first nonzero element in a 1-D array.
    Return -1 if all elements are zero.
    Use np.nonzero.
    """
    raise NotImplementedError


def top_k_indices(arr: np.ndarray, k: int) -> np.ndarray:
    """
    Return the indices (into the flattened array) of the k largest values,
    sorted from largest to smallest.
    Hint: np.argpartition, then np.argsort on the partition result.
    """
    raise NotImplementedError
