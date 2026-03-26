"""
DAY 6 — Advanced Internals
============================
Topics: structured arrays, memory layout & strides, np.einsum,
        vectorised string ops (np.char), masked arrays (np.ma),
        np.lib.stride_tricks, performance profiling.

Run: pytest day6_advanced/test_exercises.py -v
"""

import numpy as np
import numpy.lib.stride_tricks as stride_tricks


# ── 6.1  Structured Arrays ───────────────────────────────────────────────────

STUDENT_DTYPE = np.dtype([
    ('name', 'U20'),
    ('age', np.int32),
    ('gpa', np.float64),
])


def create_student_array(records: list) -> np.ndarray:
    """
    Given a list of (name, age, gpa) tuples, return a structured array
    with dtype STUDENT_DTYPE.
    """
    raise NotImplementedError


def top_students(students: np.ndarray, n: int) -> np.ndarray:
    """
    Return the `n` students with the highest GPA (structured array).
    Use fancy indexing with np.argsort on the 'gpa' field.
    """
    raise NotImplementedError


def gpa_above(students: np.ndarray, threshold: float) -> np.ndarray:
    """
    Return names (as a 1-D string array) of students with gpa > threshold.
    """
    raise NotImplementedError


# ── 6.2  Strides & Memory Layout ─────────────────────────────────────────────

def get_strides_info(arr: np.ndarray) -> dict:
    """
    Return a dict with keys:
      'strides'   → tuple of strides (bytes)
      'itemsize'  → int
      'c_contiguous' → bool
      'f_contiguous' → bool
    """
    raise NotImplementedError


def sliding_window_view(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Return a (len(arr) - window + 1, window) view using strides.
    Use np.lib.stride_tricks.sliding_window_view.
    Each row is a consecutive window of the input array.
    """
    raise NotImplementedError


def as_fortran(arr: np.ndarray) -> np.ndarray:
    """
    Return a Fortran-contiguous (column-major) copy of `arr`.
    Use np.asfortranarray.
    """
    raise NotImplementedError


# ── 6.3  np.einsum ───────────────────────────────────────────────────────────

def einsum_trace(A: np.ndarray) -> float:
    """Compute the trace of a square matrix using np.einsum."""
    raise NotImplementedError


def einsum_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Matrix multiply A @ B using np.einsum."""
    raise NotImplementedError


def einsum_batch_dot(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Given A of shape (batch, m, k) and B of shape (batch, k, n),
    return the batched matrix product of shape (batch, m, n)
    using np.einsum.
    """
    raise NotImplementedError


def einsum_outer(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Outer product of 1-D vectors a and b using np.einsum."""
    raise NotImplementedError


# ── 6.4  Masked Arrays ───────────────────────────────────────────────────────

def masked_mean(arr: np.ndarray, fill_value: float = np.nan) -> float:
    """
    Return the mean of `arr` ignoring NaN values.
    Convert to a masked array using np.ma.masked_invalid, then call .mean().
    """
    raise NotImplementedError


def interpolate_missing(arr: np.ndarray) -> np.ndarray:
    """
    Given a 1-D float array with NaN values, return a copy where each NaN
    is replaced by the linear interpolation between the nearest valid
    neighbours.  Use np.interp with the indices of valid values.
    """
    raise NotImplementedError


# ── 6.5  Vectorised String Operations ────────────────────────────────────────

def normalise_strings(arr: np.ndarray) -> np.ndarray:
    """
    Given a string array, return a new array where each string is:
      - stripped of leading/trailing whitespace
      - lowercased
    Use np.char.strip and np.char.lower.
    """
    raise NotImplementedError


def count_vowels(arr: np.ndarray) -> np.ndarray:
    """
    Given a string array, return an int array counting the vowels (aeiou)
    in each string.  Use np.char.count in a vectorised way.
    """
    raise NotImplementedError
