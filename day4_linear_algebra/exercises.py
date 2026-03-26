"""
DAY 4 — Linear Algebra
========================
Topics: np.linalg — dot products, matrix multiply, inverse, solve,
        eigenvalues, SVD, QR, Cholesky, norms, determinants.

Run: pytest day4_linear_algebra/test_exercises.py -v
"""

import numpy as np


# ── 4.1  Basic Operations ────────────────────────────────────────────────────

def matmul_chain(*matrices) -> np.ndarray:
    """
    Multiply any number of 2-D matrices together (left to right).
    Use np.linalg.multi_dot for efficiency when ≥ 3 matrices are given,
    fall back to @ for 2.
    """
    raise NotImplementedError


def gram_matrix(X: np.ndarray) -> np.ndarray:
    """
    Return the Gram matrix G = X @ X.T for a data matrix X of shape (n, d).
    """
    raise NotImplementedError


def solve_linear_system(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve Ax = b for x using np.linalg.solve.
    Raise np.linalg.LinAlgError if A is singular (propagate the exception).
    """
    raise NotImplementedError


# ── 4.2  Decompositions ──────────────────────────────────────────────────────

def principal_components(X: np.ndarray, k: int) -> np.ndarray:
    """
    Given a data matrix X of shape (n, d), return the top-k principal
    components (right singular vectors) as rows of a (k, d) matrix.
    Steps:
      1. Center X (subtract column means).
      2. Compute the thin SVD.
      3. Return the first k rows of Vt.
    """
    raise NotImplementedError


def is_positive_definite(A: np.ndarray) -> bool:
    """
    Return True if A is symmetric positive-definite.
    Use the Cholesky decomposition — catch LinAlgError.
    """
    raise NotImplementedError


def qr_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve the least-squares problem min ||Ax - b||² using QR decomposition.
    Steps:
      1. Compute the reduced QR: Q, R = np.linalg.qr(A, mode='reduced')
      2. Solve R @ x = Q.T @ b using np.linalg.solve.
    A has shape (m, n) with m ≥ n (overdetermined).
    """
    raise NotImplementedError


# ── 4.3  Eigenvalues & Norms ─────────────────────────────────────────────────

def spectral_radius(A: np.ndarray) -> float:
    """
    Return the spectral radius of A: max(|eigenvalues|).
    Use np.linalg.eigvals.
    """
    raise NotImplementedError


def matrix_power_via_eigen(A: np.ndarray, p: int) -> np.ndarray:
    """
    Compute A^p using eigendecomposition for a symmetric matrix A:
      A = V @ diag(λ) @ V.T  →  A^p = V @ diag(λ^p) @ V.T
    Use np.linalg.eigh (for symmetric matrices).
    Return real part only.
    """
    raise NotImplementedError


def frobenius_norm(A: np.ndarray) -> float:
    """Return the Frobenius norm of matrix A using np.linalg.norm."""
    raise NotImplementedError


def condition_number(A: np.ndarray) -> float:
    """
    Return the condition number of A (ratio of largest to smallest singular
    value).  Use np.linalg.cond.
    """
    raise NotImplementedError


# ── 4.4  Advanced ────────────────────────────────────────────────────────────

def pseudo_inverse_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute the least-squares solution to Ax ≈ b using the Moore-Penrose
    pseudo-inverse (np.linalg.pinv).
    """
    raise NotImplementedError


def low_rank_approximation(A: np.ndarray, rank: int) -> np.ndarray:
    """
    Return the best rank-`rank` approximation of A using truncated SVD.
    A ≈ U[:, :rank] @ diag(s[:rank]) @ Vt[:rank, :]
    """
    raise NotImplementedError
