"""
Tests for Day 6 — Advanced Internals
Run: pytest day6_advanced/test_exercises.py -v
"""

import numpy as np
import pytest
from day6_advanced.exercises import (
    STUDENT_DTYPE,
    create_student_array, top_students, gpa_above,
    get_strides_info, sliding_window_view, as_fortran,
    einsum_trace, einsum_matmul, einsum_batch_dot, einsum_outer,
    masked_mean, interpolate_missing,
    normalise_strings, count_vowels,
)


# ── 6.1  Structured Arrays ────────────────────────────────────────────────────

class TestStructuredArrays:

    RECORDS = [("Alice", 21, 3.9), ("Bob", 22, 3.2), ("Carol", 20, 3.7)]

    def test_create_dtype(self):
        out = create_student_array(self.RECORDS)
        assert out.dtype == STUDENT_DTYPE

    def test_create_length(self):
        out = create_student_array(self.RECORDS)
        assert len(out) == 3

    def test_create_field_access(self):
        out = create_student_array(self.RECORDS)
        assert out['name'][0] == "Alice"
        assert out['age'][1] == 22
        assert out['gpa'][2] == pytest.approx(3.7)

    def test_top_students_order(self):
        students = create_student_array(self.RECORDS)
        top = top_students(students, 2)
        assert top['name'][0] == "Alice"
        assert top['name'][1] == "Carol"

    def test_top_students_n1(self):
        students = create_student_array(self.RECORDS)
        top = top_students(students, 1)
        assert top['gpa'][0] == pytest.approx(3.9)

    def test_gpa_above_returns_names(self):
        students = create_student_array(self.RECORDS)
        names = gpa_above(students, 3.5)
        assert set(names) == {"Alice", "Carol"}

    def test_gpa_above_none(self):
        students = create_student_array(self.RECORDS)
        names = gpa_above(students, 4.0)
        assert len(names) == 0


# ── 6.2  Strides ──────────────────────────────────────────────────────────────

class TestStrides:

    def test_strides_info_keys(self):
        arr = np.zeros((3, 4))
        info = get_strides_info(arr)
        for k in ('strides', 'itemsize', 'c_contiguous', 'f_contiguous'):
            assert k in info

    def test_strides_info_c_contiguous(self):
        arr = np.zeros((3, 4), order='C')
        info = get_strides_info(arr)
        assert info['c_contiguous'] is True
        assert info['f_contiguous'] is False

    def test_strides_info_f_contiguous(self):
        arr = np.zeros((3, 4), order='F')
        info = get_strides_info(arr)
        assert info['f_contiguous'] is True

    def test_strides_info_itemsize(self):
        arr = np.zeros(5, dtype=np.float64)
        info = get_strides_info(arr)
        assert info['itemsize'] == 8

    def test_sliding_window_shape(self):
        arr = np.arange(10, dtype=float)
        out = sliding_window_view(arr, 3)
        assert out.shape == (8, 3)

    def test_sliding_window_values(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = sliding_window_view(arr, 3)
        assert np.array_equal(out[0], [1, 2, 3])
        assert np.array_equal(out[2], [3, 4, 5])

    def test_sliding_window_is_view(self):
        arr = np.arange(20, dtype=float)
        out = sliding_window_view(arr, 4)
        assert np.shares_memory(out, arr)

    def test_as_fortran_flag(self):
        arr = np.zeros((3, 4), order='C')
        out = as_fortran(arr)
        assert out.flags['F_CONTIGUOUS']

    def test_as_fortran_values(self):
        arr = np.arange(6).reshape(2, 3)
        out = as_fortran(arr)
        assert np.array_equal(out, arr)


# ── 6.3  einsum ───────────────────────────────────────────────────────────────

class TestEinsum:

    def test_trace_identity(self):
        assert einsum_trace(np.eye(5)) == pytest.approx(5.0)

    def test_trace_known(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert einsum_trace(A) == pytest.approx(5.0)

    def test_matmul_shape(self):
        A = np.random.rand(3, 4)
        B = np.random.rand(4, 5)
        out = einsum_matmul(A, B)
        assert out.shape == (3, 5)

    def test_matmul_values(self):
        A = np.random.rand(4, 3)
        B = np.random.rand(3, 6)
        out = einsum_matmul(A, B)
        assert np.allclose(out, A @ B)

    def test_batch_dot_shape(self):
        A = np.random.rand(8, 3, 4)
        B = np.random.rand(8, 4, 5)
        out = einsum_batch_dot(A, B)
        assert out.shape == (8, 3, 5)

    def test_batch_dot_values(self):
        A = np.random.rand(4, 2, 3)
        B = np.random.rand(4, 3, 2)
        out = einsum_batch_dot(A, B)
        expected = np.stack([A[i] @ B[i] for i in range(4)])
        assert np.allclose(out, expected)

    def test_outer_shape(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0])
        out = einsum_outer(a, b)
        assert out.shape == (3, 2)

    def test_outer_values(self):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        out = einsum_outer(a, b)
        assert np.allclose(out, np.outer(a, b))


# ── 6.4  Masked Arrays ────────────────────────────────────────────────────────

class TestMasked:

    def test_masked_mean_with_nan(self):
        arr = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        result = masked_mean(arr)
        assert result == pytest.approx(3.0)

    def test_masked_mean_no_nan(self):
        arr = np.array([2.0, 4.0, 6.0])
        result = masked_mean(arr)
        assert result == pytest.approx(4.0)

    def test_masked_mean_all_nan(self):
        arr = np.array([np.nan, np.nan])
        result = masked_mean(arr)
        assert np.isnan(result) or result == pytest.approx(0.0, abs=1e9)

    def test_interpolate_missing_no_nan(self):
        arr = np.array([1.0, 2.0, 3.0])
        out = interpolate_missing(arr)
        assert np.allclose(out, arr)

    def test_interpolate_missing_middle(self):
        arr = np.array([1.0, np.nan, 3.0])
        out = interpolate_missing(arr)
        assert out[1] == pytest.approx(2.0)

    def test_interpolate_missing_multiple(self):
        arr = np.array([0.0, np.nan, np.nan, 6.0])
        out = interpolate_missing(arr)
        assert out[1] == pytest.approx(2.0)
        assert out[2] == pytest.approx(4.0)

    def test_interpolate_is_copy(self):
        arr = np.array([1.0, np.nan, 3.0])
        out = interpolate_missing(arr)
        assert not np.shares_memory(out, arr)


# ── 6.5  Vectorised Strings ───────────────────────────────────────────────────

class TestStrings:

    def test_normalise_strip(self):
        arr = np.array(["  Hello  ", "World "])
        out = normalise_strings(arr)
        assert out[0] == "hello"
        assert out[1] == "world"

    def test_normalise_lower(self):
        arr = np.array(["ABC", "DeF"])
        out = normalise_strings(arr)
        assert np.all(out == np.array(["abc", "def"]))

    def test_count_vowels_basic(self):
        arr = np.array(["hello", "world"])
        out = count_vowels(arr)
        assert out[0] == 2  # e, o
        assert out[1] == 1  # o

    def test_count_vowels_empty(self):
        arr = np.array(["bcdf", ""])
        out = count_vowels(arr)
        assert out[0] == 0
        assert out[1] == 0

    def test_count_vowels_dtype(self):
        arr = np.array(["aeiou"])
        out = count_vowels(arr)
        assert out[0] == 5
        assert np.issubdtype(out.dtype, np.integer)
