"""
DAY 1 — Array Basics & Data Types
==================================
Topics: array creation, shape/ndim/size, dtypes, reshaping, stacking, copying.
"""

import numpy as np


# ── 1.1  Creation ────────────────────────────────────────────────────────────

def zeros_like_int(shape: tuple) -> np.ndarray:
    """Return an integer zero-array of the given shape (dtype=np.int32)."""
    arr = np.zeros(shape, dtype=np.int32)
    return arr


def create_range(start: float, stop: float, n: int) -> np.ndarray:
    """Return `n` evenly-spaced float64 values in [start, stop] (inclusive)."""
    arr = np.linspace(start, stop, n, dtype=np.float64)
    return arr


def identity_block(n: int, k: int) -> np.ndarray:
    """
    Return an (n, n) float64 matrix that is an identity matrix shifted by k
    diagonals.  k=0 → main diagonal, k=1 → one above, k=-1 → one below.
    Hint: np.eye has a `k` parameter.
    """
    arr = np.eye(n, n, k=k, dtype=np.float64)
    return arr


def build_checkerboard(n: int) -> np.ndarray:
    """
    Return an (n, n) uint8 array of 0s and 1s arranged like a checkerboard.
    Element [i, j] is 1 if (i + j) is even, else 0.
    Must work without any Python loop.
    """

    return np.uint8(np.indices((n, n), dtype=np.uint8).sum(axis=0) % 2)


# ── 1.2  Shape & dtype ───────────────────────────────────────────────────────

def describe(arr: np.ndarray) -> dict:
    """
    Return a dict with keys:
      'shape'   → tuple
      'ndim'    → int
      'size'    → int  (total number of elements)
      'dtype'   → str  (e.g. 'float64')
      'itemsize'→ int  (bytes per element)
      'nbytes'  → int  (total bytes)
    """
    return {'shape': arr.shape, 'ndim': arr.ndim, 'dtype': arr.dtype, 'itemsize': arr.itemsize, 'nbytes': arr.nbytes, 'size': arr.size}


def safe_cast(arr: np.ndarray, target_dtype: np.dtype) -> np.ndarray:
    """
    Cast `arr` to `target_dtype` using numpy's 'same_kind' casting rule.
    If the cast is not allowed, return the original array unchanged
    (catch the TypeError numpy raises).
    """
    backup = arr.copy()
    try:
        new = arr.astype(target_dtype, casting='safe')
        return new
    except TypeError:
        return backup


# ── 1.3  Reshaping & Stacking ─────────────────────────────────────────────────

def flatten_and_sort(arr: np.ndarray) -> np.ndarray:
    """Return a 1-D copy of `arr` sorted in ascending order."""
    return np.sort(arr.flatten())


def stack_as_matrix(arrays: list) -> np.ndarray:
    """
    Given a list of 1-D arrays of equal length, return a 2-D array where
    each input array is one ROW.
    """
    return np.vstack(arrays)


def tile_border(inner: np.ndarray, pad: int) -> np.ndarray:
    """
    Pad `inner` (2-D) with `pad` zeros on every side and return the result.
    Shape goes from (H, W) → (H + 2*pad, W + 2*pad).
    Hint: np.pad.
    """
    h, w = inner.shape
    return np.pad(inner, pad)

# ── 1.4  Copies vs Views ─────────────────────────────────────────────────────


def make_view(arr: np.ndarray) -> np.ndarray:
    """Return a view (not a copy) of `arr` reshaped to be 1-D."""
    return arr.view().reshape(-1)


def is_view_of(candidate: np.ndarray, base: np.ndarray) -> bool:
    """Return True if `candidate` shares memory with `base`."""
    return np.shares_memory(candidate, base)


def forced_copy(arr: np.ndarray) -> np.ndarray:
    """Return a contiguous C-order copy of `arr` with the same dtype."""
    return arr.copy()
