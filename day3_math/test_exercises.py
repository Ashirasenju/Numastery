"""
Tests for Day 3 — Universal Functions & Broadcasting
Run: pytest day3_math/test_exercises.py -v
"""

import numpy as np
import pytest
from day3_math.exercises import (
    running_sum, outer_product_sum, log_sum_exp, custom_relu_ufunc,
    row_normalize, pairwise_l2, add_bias, outer_subtract,
    softmax, moving_average, polynomial_eval,
)


class TestUfuncs:

    def test_running_sum_values(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        out = running_sum(arr)
        assert np.allclose(out, [1, 3, 6, 10])

    def test_running_sum_length(self):
        arr = np.arange(10, dtype=float)
        assert len(running_sum(arr)) == len(arr)

    def test_outer_product_sum(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0])
        # outer sum = sum of a_i * b_j = sum(a)*sum(b)
        assert outer_product_sum(a, b) == pytest.approx(6.0 * 9.0)

    def test_log_sum_exp_basic(self):
        arr = np.array([0.0, 0.0, 0.0])
        # log(3*exp(0)) = log(3)
        assert log_sum_exp(arr) == pytest.approx(np.log(3))

    def test_log_sum_exp_stable(self):
        arr = np.array([1000.0, 1001.0, 1002.0])
        # Should not overflow
        result = log_sum_exp(arr)
        assert np.isfinite(result)
        # Exact: log(e^1000 + e^1001 + e^1002) = 1000 + log(1 + e + e^2)
        expected = 1000 + np.log(1 + np.e + np.e**2)
        assert result == pytest.approx(expected, rel=1e-5)

    def test_relu_positive(self):
        arr = np.array([1.0, 2.0, 3.0])
        out = custom_relu_ufunc(arr)
        assert np.allclose(out, arr)

    def test_relu_negative(self):
        arr = np.array([-1.0, -2.0])
        out = custom_relu_ufunc(arr)
        assert np.allclose(out, [0.0, 0.0])

    def test_relu_mixed(self):
        arr = np.array([-3.0, 0.0, 4.0])
        out = custom_relu_ufunc(arr)
        assert np.allclose(out, [0.0, 0.0, 4.0])


class TestBroadcasting:

    def test_row_normalize_sums(self):
        matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out = row_normalize(matrix)
        assert np.allclose(out.sum(axis=1), [1.0, 1.0])

    def test_row_normalize_values(self):
        matrix = np.array([[1.0, 1.0]])
        out = row_normalize(matrix)
        assert np.allclose(out, [[0.5, 0.5]])

    def test_row_normalize_dtype(self):
        matrix = np.ones((3, 4))
        out = row_normalize(matrix)
        assert out.dtype == np.float64

    def test_pairwise_l2_shape(self):
        A = np.random.rand(5, 3)
        B = np.random.rand(7, 3)
        out = pairwise_l2(A, B)
        assert out.shape == (5, 7)

    def test_pairwise_l2_zero_self(self):
        A = np.random.rand(4, 3)
        out = pairwise_l2(A, A)
        assert np.allclose(np.diag(out), 0.0)

    def test_pairwise_l2_known(self):
        A = np.array([[0.0, 0.0]])
        B = np.array([[3.0, 4.0]])
        out = pairwise_l2(A, B)
        assert out[0, 0] == pytest.approx(5.0)

    def test_add_bias_shape(self):
        X = np.ones((10, 5))
        bias = np.arange(5, dtype=float)
        out = add_bias(X, bias)
        assert out.shape == (10, 5)

    def test_add_bias_values(self):
        X = np.zeros((3, 2))
        bias = np.array([1.0, 2.0])
        out = add_bias(X, bias)
        assert np.allclose(out, [[1, 2], [1, 2], [1, 2]])

    def test_outer_subtract_shape(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([10.0, 20.0])
        out = outer_subtract(a, b)
        assert out.shape == (3, 2)

    def test_outer_subtract_values(self):
        a = np.array([5.0, 6.0])
        b = np.array([1.0, 2.0, 3.0])
        out = outer_subtract(a, b)
        assert np.allclose(out, [[4, 3, 2], [5, 4, 3]])


class TestVectorisedLogic:

    def test_softmax_sums_to_one(self):
        arr = np.array([1.0, 2.0, 3.0])
        out = softmax(arr)
        assert pytest.approx(out.sum()) == 1.0

    def test_softmax_positive(self):
        arr = np.array([1.0, -1.0, 0.0])
        out = softmax(arr)
        assert np.all(out > 0)

    def test_softmax_max_dominates(self):
        arr = np.array([100.0, 0.0, 0.0])
        out = softmax(arr)
        assert out[0] > 0.999

    def test_softmax_stable(self):
        arr = np.array([1000.0, 1001.0])
        out = softmax(arr)
        assert np.all(np.isfinite(out))

    def test_moving_average_length(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = moving_average(arr, 3)
        assert len(out) == 3

    def test_moving_average_values(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = moving_average(arr, 3)
        assert np.allclose(out, [2.0, 3.0, 4.0])

    def test_moving_average_window1(self):
        arr = np.array([7.0, 8.0, 9.0])
        out = moving_average(arr, 1)
        assert np.allclose(out, arr)

    def test_polynomial_eval_constant(self):
        # p(x) = 3
        out = polynomial_eval(np.array([3.0]), np.array([0.0, 1.0, 2.0]))
        assert np.allclose(out, [3.0, 3.0, 3.0])

    def test_polynomial_eval_linear(self):
        # p(x) = 2 + 3x
        out = polynomial_eval(np.array([2.0, 3.0]), np.array([0.0, 1.0, 2.0]))
        assert np.allclose(out, [2.0, 5.0, 8.0])

    def test_polynomial_eval_quadratic(self):
        # p(x) = 1 + 0*x + x^2
        out = polynomial_eval(np.array([1.0, 0.0, 1.0]), np.array([2.0, 3.0]))
        assert np.allclose(out, [5.0, 10.0])
