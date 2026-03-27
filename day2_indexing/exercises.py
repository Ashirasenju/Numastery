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
    view = arr.view()
    return view[row_start:row_end,col_start:col_end]

def every_other_row(arr: np.ndarray) -> np.ndarray:
    """Return every second row (0, 2, 4, …) of a 2-D array as a view."""
    return arr[::2]

def reverse_columns(arr: np.ndarray) -> np.ndarray:
    """Return `arr` with its columns in reversed order (view, not copy)."""
    return np.fliplr(arr)


def diagonal_sum(arr: np.ndarray) -> float:
    """
    Return the sum of the main diagonal elements of a square 2-D array.
    Use np.trace or np.diag — no explicit loops.
    """
    return float(np.diag(arr).sum())


# ── 2.2  Fancy Indexing ──────────────────────────────────────────────────────

def select_rows(arr: np.ndarray, indices: list) -> np.ndarray:
    """
    Return the rows of `arr` at `indices`, in that order.
    Result must be a COPY (fancy indexing always copies).
    """
    return arr[indices]


def set_diagonal(arr: np.ndarray, value: float) -> np.ndarray:
    """
    Return a copy of `arr` with its main diagonal set to `value`.
    Do not use a loop; use np.fill_diagonal or advanced indexing.
    """
    da_copy = arr.copy()
    np.fill_diagonal(da_copy,value)
    return da_copy


def scatter_add(base: np.ndarray, indices: np.ndarray,
                values: np.ndarray) -> np.ndarray:
    """
    Return a copy of `base` where values[i] is ADDED to base[indices[i]]
    for each i.  Handle repeated indices correctly (use np.add.at).
    """
    copy = base.copy()
    np.add.at(copy,indices,values)   
    return copy


# ── 2.3  Boolean Masking ─────────────────────────────────────────────────────

def clamp(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """
    Return a copy of `arr` where values below `lo` are set to `lo` and
    values above `hi` are set to `hi`.  Use np.clip.
    """
    arr_copy = arr.copy()
    copy = np.clip(arr_copy,lo,hi)
    return copy

def replace_outliers(arr: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """
    Return a copy of `arr` (float64) where any element more than `sigma`
    standard deviations from the mean is replaced by the mean.
    """
    output = arr.copy()
    mean = arr.mean()

    return output








def mask_nan_inf(arr: np.ndarray) -> np.ndarray:
    """
    Return a copy of `arr` where NaN and Inf values are replaced by 0.
    Use np.isfinite.
    """
    mask = ~np.isfinite(arr)
    dacopy = arr.copy()
    dacopy[mask] = 0
    return dacopy



# ── 2.4  np.where & np.nonzero ───────────────────────────────────────────────

def sign_array(arr: np.ndarray) -> np.ndarray:
    """
    Return an array of the same shape where:
      positive values → +1, negative values → -1, zero → 0.
    Use np.where (or np.sign).
    """
    arr = np.where(arr >0, 1, np.where(arr < 0, -1,0))
    return arr


def first_nonzero_index(arr: np.ndarray) -> int:
    """
    Return the index of the first nonzero element in a 1-D array.
    Return -1 if all elements are zero.
    Use np.nonzero.
    """
    tuple = np.nonzero(arr)
    return tuple[0][0] if len(tuple[0]) > 0 else -1


def top_k_indices(arr: np.ndarray, k: int) -> np.ndarray:
    """
    Return the indices (into the flattened array) of the k largest values,
    sorted from largest to smallest.
    Hint: np.argpartition, then np.argsort on the partition result.
    """
    partitioned_indices = np.argpartition(arr,-k)[-k:]
    sorted_indices = partitioned_indices[np.argsort(arr[partitioned_indices])[::-1]]
    return sorted_indices
