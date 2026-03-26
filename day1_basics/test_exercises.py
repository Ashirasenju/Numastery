"""
Tests for Day 1 — Array Basics & Data Types
Run: pytest day1_basics/test_exercises.py -v
"""

import numpy as np
import pytest
from day1_basics.exercises import (
    zeros_like_int, create_range, identity_block, build_checkerboard,
    describe, safe_cast,
    flatten_and_sort, stack_as_matrix, tile_border,
    make_view, is_view_of, forced_copy,
)


# ── 1.1 Creation ──────────────────────────────────────────────────────────────

class TestCreation:

    def test_zeros_like_int_shape(self):
        out = zeros_like_int((3, 4))
        assert out.shape == (3, 4)

    def test_zeros_like_int_dtype(self):
        out = zeros_like_int((2, 2))
        assert out.dtype == np.int32

    def test_zeros_like_int_values(self):
        out = zeros_like_int((5,))
        assert np.all(out == 0)

    def test_create_range_length(self):
        out = create_range(0, 1, 11)
        assert len(out) == 11

    def test_create_range_endpoints(self):
        out = create_range(2.0, 5.0, 7)
        assert pytest.approx(out[0]) == 2.0
        assert pytest.approx(out[-1]) == 5.0

    def test_create_range_dtype(self):
        out = create_range(0, 10, 5)
        assert out.dtype == np.float64

    def test_identity_block_main(self):
        out = identity_block(4, 0)
        assert np.array_equal(out, np.eye(4))

    def test_identity_block_above(self):
        out = identity_block(4, 1)
        expected = np.eye(4, k=1)
        assert np.array_equal(out, expected)

    def test_identity_block_below(self):
        out = identity_block(3, -1)
        expected = np.eye(3, k=-1)
        assert np.array_equal(out, expected)

    def test_checkerboard_shape(self):
        out = build_checkerboard(6)
        assert out.shape == (6, 6)

    def test_checkerboard_dtype(self):
        out = build_checkerboard(4)
        assert out.dtype == np.uint8

    def test_checkerboard_pattern(self):
        out = build_checkerboard(4)
        for i in range(4):
            for j in range(4):
                expected = 1 if (i + j) % 2 == 0 else 0
                assert out[i, j] == expected, f"Wrong at [{i},{j}]"

    def test_checkerboard_no_loop(self):
        """Smoke test: must be fast for large n (no Python loop)."""
        import time
        t = time.perf_counter()
        build_checkerboard(1000)
        elapsed = time.perf_counter() - t
        assert elapsed < 0.5, "Too slow — you may be using a Python loop"


# ── 1.2 Shape & dtype ─────────────────────────────────────────────────────────

class TestDescribe:

    def test_keys(self):
        arr = np.zeros((2, 3, 4))
        d = describe(arr)
        for key in ('shape', 'ndim', 'size', 'dtype', 'itemsize', 'nbytes'):
            assert key in d, f"Missing key: {key}"

    def test_values_3d(self):
        arr = np.ones((2, 3, 4), dtype=np.float32)
        d = describe(arr)
        assert d['shape'] == (2, 3, 4)
        assert d['ndim'] == 3
        assert d['size'] == 24
        assert d['dtype'] == 'float32'
        assert d['itemsize'] == 4
        assert d['nbytes'] == 96

    def test_values_1d(self):
        arr = np.arange(10, dtype=np.int64)
        d = describe(arr)
        assert d['shape'] == (10,)
        assert d['ndim'] == 1
        assert d['size'] == 10


class TestSafeCast:

    def test_allowed_cast(self):
        arr = np.array([1, 2, 3], dtype=np.int32)
        out = safe_cast(arr, np.float64)
        assert out.dtype == np.float64

    def test_disallowed_cast_returns_original(self):
        arr = np.array([1.5, 2.5], dtype=np.float64)
        out = safe_cast(arr, np.int32)  # float → int is not 'same_kind'
        assert out is arr or np.array_equal(out, arr)
        assert out.dtype == np.float64


# ── 1.3 Reshaping & Stacking ─────────────────────────────────────────────────

class TestReshaping:

    def test_flatten_and_sort_values(self):
        arr = np.array([[3, 1], [4, 1], [5, 9]])
        out = flatten_and_sort(arr)
        assert np.array_equal(out, np.array([1, 1, 3, 4, 5, 9]))

    def test_flatten_and_sort_ndim(self):
        arr = np.arange(12).reshape(3, 4)
        out = flatten_and_sort(arr)
        assert out.ndim == 1

    def test_flatten_is_copy(self):
        arr = np.array([[2, 1], [4, 3]])
        out = flatten_and_sort(arr)
        out[0] = 999
        assert arr[0, 0] != 999  # original unchanged

    def test_stack_as_matrix_shape(self):
        arrays = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        out = stack_as_matrix(arrays)
        assert out.shape == (2, 3)

    def test_stack_as_matrix_values(self):
        arrays = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
        out = stack_as_matrix(arrays)
        assert np.array_equal(out[0], [1, 2])
        assert np.array_equal(out[2], [5, 6])

    def test_tile_border_shape(self):
        inner = np.ones((3, 4))
        out = tile_border(inner, 2)
        assert out.shape == (7, 8)

    def test_tile_border_zeros(self):
        inner = np.ones((2, 2))
        out = tile_border(inner, 1)
        assert out[0, 0] == 0
        assert out[-1, -1] == 0
        assert out[1, 1] == 1

    def test_tile_border_pad0(self):
        inner = np.array([[7, 8], [9, 10]])
        out = tile_border(inner, 0)
        assert np.array_equal(out, inner)


# ── 1.4 Copies vs Views ───────────────────────────────────────────────────────

class TestCopiesViews:

    def test_make_view_is_1d(self):
        arr = np.arange(12).reshape(3, 4)
        out = make_view(arr)
        assert out.ndim == 1

    def test_make_view_shares_memory(self):
        arr = np.arange(12).reshape(3, 4)
        out = make_view(arr)
        assert np.shares_memory(out, arr)

    def test_make_view_mutation_propagates(self):
        arr = np.arange(12).reshape(3, 4)
        out = make_view(arr)
        out[0] = 99
        assert arr[0, 0] == 99

    def test_is_view_of_true(self):
        base = np.arange(10)
        view = base[2:7]
        assert is_view_of(view, base) is True

    def test_is_view_of_false(self):
        base = np.arange(10)
        copy = base.copy()
        assert is_view_of(copy, base) is False

    def test_forced_copy_no_shared_memory(self):
        arr = np.arange(12).reshape(3, 4)
        out = forced_copy(arr)
        assert not np.shares_memory(out, arr)

    def test_forced_copy_is_contiguous(self):
        arr = np.arange(12).reshape(3, 4)[::2]
        out = forced_copy(arr)
        assert out.flags['C_CONTIGUOUS']

    def test_forced_copy_same_values(self):
        arr = np.array([[1, 2], [3, 4]])
        out = forced_copy(arr)
        assert np.array_equal(out, arr)
