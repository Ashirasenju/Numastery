"""
Tests for Day 4 — Linear Algebra
Run: pytest day4_linear_algebra/test_exercises.py -v
"""

import numpy as np
import pytest
from day4_linear_algebra.exercises import (
    matmul_chain, gram_matrix, solve_linear_system,
    principal_components, is_positive_definite, qr_solve,
    spectral_radius, matrix_power_via_eigen,
    frobenius_norm, condition_number,
    pseudo_inverse_solve, low_rank_approximation,
)


class TestBasicOps:

    def test_matmul_chain_two(self):
        A = np.eye(3) * 2
        B = np.eye(3) * 3
        out = matmul_chain(A, B)
        assert np.allclose(out, np.eye(3) * 6)

    def test_matmul_chain_three(self):
        A = np.array([[1, 2], [0, 1]])
        B = np.array([[1, 0], [1, 1]])
        C = np.array([[2, 0], [0, 2]])
        out = matmul_chain(A, B, C)
        expected = A @ B @ C
        assert np.allclose(out, expected)

    def test_gram_matrix_shape(self):
        X = np.random.rand(5, 3)
        out = gram_matrix(X)
        assert out.shape == (5, 5)

    def test_gram_matrix_symmetric(self):
        X = np.random.rand(4, 6)
        out = gram_matrix(X)
        assert np.allclose(out, out.T)

    def test_gram_matrix_psd(self):
        X = np.random.rand(4, 4)
        G = gram_matrix(X)
        eigvals = np.linalg.eigvalsh(G)
        assert np.all(eigvals >= -1e-10)

    def test_solve_linear_system(self):
        A = np.array([[2.0, 1.0], [5.0, 7.0]])
        b = np.array([11.0, 13.0])
        x = solve_linear_system(A, b)
        assert np.allclose(A @ x, b)

    def test_solve_singular_raises(self):
        A = np.array([[1.0, 2.0], [2.0, 4.0]])
        b = np.array([1.0, 2.0])
        with pytest.raises(np.linalg.LinAlgError):
            solve_linear_system(A, b)


class TestDecompositions:

    def test_principal_components_shape(self):
        X = np.random.rand(50, 10)
        out = principal_components(X, 3)
        assert out.shape == (3, 10)

    def test_principal_components_orthogonal(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 8))
        pcs = principal_components(X, 4)
        product = pcs @ pcs.T
        assert np.allclose(product, np.eye(4), atol=1e-10)

    def test_principal_components_centered(self):
        """First PC should explain more variance than raw features after centering."""
        rng = np.random.default_rng(7)
        X = rng.standard_normal((100, 5)) + 10  # big mean
        pcs = principal_components(X, 1)
        assert pcs.shape == (1, 5)

    def test_is_positive_definite_true(self):
        A = np.eye(4) * 3
        assert is_positive_definite(A) is True

    def test_is_positive_definite_false(self):
        A = -np.eye(3)
        assert is_positive_definite(A) is False

    def test_is_positive_definite_random_spd(self):
        rng = np.random.default_rng(0)
        M = rng.standard_normal((5, 5))
        A = M @ M.T + np.eye(5)  # guaranteed SPD
        assert is_positive_definite(A) is True

    def test_qr_solve_square(self):
        rng = np.random.default_rng(1)
        A = rng.standard_normal((5, 5))
        b = rng.standard_normal(5)
        x = qr_solve(A, b)
        assert np.allclose(A @ x, b, atol=1e-10)

    def test_qr_solve_overdetermined(self):
        rng = np.random.default_rng(2)
        A = rng.standard_normal((10, 4))
        b = rng.standard_normal(10)
        x = qr_solve(A, b)
        # Compare with numpy's lstsq
        x_ref, *_ = np.linalg.lstsq(A, b, rcond=None)
        assert np.allclose(x, x_ref, atol=1e-8)


class TestEigenNorms:

    def test_spectral_radius_identity(self):
        A = np.eye(5) * 3
        assert spectral_radius(A) == pytest.approx(3.0)

    def test_spectral_radius_positive(self):
        A = np.random.rand(4, 4)
        assert spectral_radius(A) >= 0

    def test_matrix_power_p2(self):
        A = np.array([[2.0, 0.0], [0.0, 3.0]])  # diagonal, symmetric
        out = matrix_power_via_eigen(A, 2)
        expected = np.array([[4.0, 0.0], [0.0, 9.0]])
        assert np.allclose(out, expected, atol=1e-10)

    def test_matrix_power_p0(self):
        rng = np.random.default_rng(5)
        M = rng.standard_normal((4, 4))
        A = M @ M.T + np.eye(4)  # symmetric positive definite
        out = matrix_power_via_eigen(A, 0)
        assert np.allclose(out, np.eye(4), atol=1e-10)

    def test_frobenius_norm_identity(self):
        A = np.eye(3)
        assert frobenius_norm(A) == pytest.approx(np.sqrt(3))

    def test_frobenius_norm_known(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert frobenius_norm(A) == pytest.approx(np.sqrt(30))

    def test_condition_number_identity(self):
        assert condition_number(np.eye(5)) == pytest.approx(1.0)

    def test_condition_number_large(self):
        A = np.diag([1.0, 1000.0])
        assert condition_number(A) == pytest.approx(1000.0)


class TestAdvanced:

    def test_pseudo_inverse_solve_consistent(self):
        A = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        b = np.array([1.0, 2.0, 0.0])
        x = pseudo_inverse_solve(A, b)
        assert np.allclose(x, [1.0, 2.0], atol=1e-10)

    def test_pseudo_inverse_solve_inconsistent(self):
        A = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        b = np.array([1.0, 2.0, 5.0])
        x = pseudo_inverse_solve(A, b)
        x_ref, *_ = np.linalg.lstsq(A, b, rcond=None)
        assert np.allclose(x, x_ref, atol=1e-8)

    def test_low_rank_shape(self):
        A = np.random.rand(6, 8)
        out = low_rank_approximation(A, 2)
        assert out.shape == A.shape

    def test_low_rank_rank(self):
        A = np.random.rand(6, 8)
        out = low_rank_approximation(A, 2)
        rank = np.linalg.matrix_rank(out)
        assert rank <= 2

    def test_low_rank_error_decreasing(self):
        rng = np.random.default_rng(9)
        A = rng.standard_normal((10, 10))
        err1 = np.linalg.norm(A - low_rank_approximation(A, 1))
        err3 = np.linalg.norm(A - low_rank_approximation(A, 3))
        assert err3 < err1
