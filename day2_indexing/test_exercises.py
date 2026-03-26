"""
Tests for Day 2 — Indexing, Slicing & Masking
Run: pytest day2_indexing/test_exercises.py -v
"""

import numpy as np
import pytest
from day2_indexing.exercises import (
    extract_submatrix, every_other_row, reverse_columns, diagonal_sum,
    select_rows, set_diagonal, scatter_add,
    clamp, replace_outliers, mask_nan_inf,
    sign_array, first_nonzero_index, top_k_indices,
)


class TestBasicSlicing:

    def test_extract_submatrix_values(self):
        arr = np.arange(25).reshape(5, 5)
        out = extract_submatrix(arr, 1, 3, 2, 5)
        expected = arr[1:3, 2:5]
        assert np.array_equal(out, expected)

    def test_extract_submatrix_is_view(self):
        arr = np.arange(25).reshape(5, 5)
        out = extract_submatrix(arr, 0, 2, 0, 2)
        assert np.shares_memory(out, arr)

    def test_every_other_row_shape(self):
        arr = np.arange(30).reshape(6, 5)
        out = every_other_row(arr)
        assert out.shape == (3, 5)

    def test_every_other_row_values(self):
        arr = np.arange(20).reshape(4, 5)
        out = every_other_row(arr)
        assert np.array_equal(out, arr[::2])

    def test_every_other_row_is_view(self):
        arr = np.arange(20).reshape(4, 5)
        out = every_other_row(arr)
        assert np.shares_memory(out, arr)

    def test_reverse_columns_values(self):
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        out = reverse_columns(arr)
        assert np.array_equal(out, np.array([[3, 2, 1], [6, 5, 4]]))

    def test_reverse_columns_is_view(self):
        arr = np.arange(12).reshape(3, 4)
        out = reverse_columns(arr)
        assert np.shares_memory(out, arr)

    def test_diagonal_sum(self):
        arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        assert diagonal_sum(arr) == pytest.approx(15.0)

    def test_diagonal_sum_identity(self):
        arr = np.eye(5)
        assert diagonal_sum(arr) == pytest.approx(5.0)


class TestFancyIndexing:

    def test_select_rows_order(self):
        arr = np.arange(20).reshape(4, 5)
        out = select_rows(arr, [3, 0, 2])
        assert np.array_equal(out[0], arr[3])
        assert np.array_equal(out[1], arr[0])

    def test_select_rows_is_copy(self):
        arr = np.arange(20).reshape(4, 5)
        out = select_rows(arr, [0, 1])
        assert not np.shares_memory(out, arr)

    def test_set_diagonal_values(self):
        arr = np.zeros((3, 3))
        out = set_diagonal(arr, 7.0)
        assert np.array_equal(np.diag(out), [7.0, 7.0, 7.0])

    def test_set_diagonal_is_copy(self):
        arr = np.zeros((3, 3))
        out = set_diagonal(arr, 5.0)
        assert not np.shares_memory(out, arr) or arr[0, 0] == 0

    def test_scatter_add_no_repeat(self):
        base = np.zeros(5)
        out = scatter_add(base, np.array([1, 3]), np.array([10.0, 20.0]))
        assert out[1] == pytest.approx(10.0)
        assert out[3] == pytest.approx(20.0)

    def test_scatter_add_repeated_indices(self):
        base = np.zeros(4)
        out = scatter_add(base, np.array([2, 2, 2]), np.array([1.0, 2.0, 3.0]))
        assert out[2] == pytest.approx(6.0)

    def test_scatter_add_is_copy(self):
        base = np.zeros(5)
        out = scatter_add(base, np.array([0]), np.array([1.0]))
        assert not np.shares_memory(out, base)


class TestBooleanMasking:

    def test_clamp_below(self):
        arr = np.array([-5.0, 0.0, 3.0, 10.0])
        out = clamp(arr, 0.0, 5.0)
        assert out[0] == pytest.approx(0.0)

    def test_clamp_above(self):
        arr = np.array([-5.0, 0.0, 3.0, 10.0])
        out = clamp(arr, 0.0, 5.0)
        assert out[3] == pytest.approx(5.0)

    def test_clamp_middle_unchanged(self):
        arr = np.array([2.0])
        out = clamp(arr, 0.0, 5.0)
        assert out[0] == pytest.approx(2.0)

    def test_replace_outliers_no_outlier(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = replace_outliers(arr, sigma=3.0)
        assert np.allclose(out, arr)

    def test_replace_outliers_replaces(self):
        rng = np.random.default_rng(0)
        arr = rng.normal(0, 1, 1000)
        arr[500] = 100.0  # extreme outlier
        out = replace_outliers(arr, sigma=2.0)
        assert out[500] == pytest.approx(arr.mean(), abs=0.5)

    def test_replace_outliers_is_copy(self):
        arr = np.array([1.0, 2.0, 100.0])
        out = replace_outliers(arr, sigma=1.0)
        assert not np.shares_memory(out, arr)

    def test_mask_nan_inf_nan(self):
        arr = np.array([1.0, np.nan, 3.0])
        out = mask_nan_inf(arr)
        assert out[1] == pytest.approx(0.0)

    def test_mask_nan_inf_inf(self):
        arr = np.array([np.inf, 2.0, -np.inf])
        out = mask_nan_inf(arr)
        assert out[0] == pytest.approx(0.0)
        assert out[2] == pytest.approx(0.0)

    def test_mask_nan_inf_keeps_valid(self):
        arr = np.array([1.0, np.nan, 3.0])
        out = mask_nan_inf(arr)
        assert out[0] == pytest.approx(1.0)
        assert out[2] == pytest.approx(3.0)


class TestWhereNonzero:

    def test_sign_array_positive(self):
        arr = np.array([3.0, -2.0, 0.0])
        out = sign_array(arr)
        assert out[0] == 1

    def test_sign_array_negative(self):
        arr = np.array([3.0, -2.0, 0.0])
        out = sign_array(arr)
        assert out[1] == -1

    def test_sign_array_zero(self):
        arr = np.array([3.0, -2.0, 0.0])
        out = sign_array(arr)
        assert out[2] == 0

    def test_first_nonzero_index_normal(self):
        arr = np.array([0, 0, 5, 0, 3])
        assert first_nonzero_index(arr) == 2

    def test_first_nonzero_index_all_zero(self):
        arr = np.zeros(5)
        assert first_nonzero_index(arr) == -1

    def test_first_nonzero_index_first_element(self):
        arr = np.array([7, 0, 0])
        assert first_nonzero_index(arr) == 0

    def test_top_k_indices_values(self):
        arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
        out = top_k_indices(arr, 3)
        assert set(out) == {5, 7, 4}  # 9, 6, 5

    def test_top_k_indices_sorted(self):
        arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
        out = top_k_indices(arr, 3)
        vals = arr[out]
        assert list(vals) == sorted(vals, reverse=True)

    def test_top_k_indices_k1(self):
        arr = np.array([10, 3, 7, 1])
        out = top_k_indices(arr, 1)
        assert out[0] == 0
