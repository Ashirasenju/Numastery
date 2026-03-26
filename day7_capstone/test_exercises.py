"""
Tests for Day 7 — Capstone: Neural Network from Scratch
Run: pytest day7_capstone/test_exercises.py -v
"""

import numpy as np
import pytest
from day7_capstone.exercises import (
    relu, relu_backward, sigmoid, binary_cross_entropy,
    TwoLayerNet, train, accuracy, roc_auc, make_moons_numpy,
)


# ── Activations & Loss ────────────────────────────────────────────────────────

class TestActivations:

    def test_relu_positive(self):
        Z = np.array([[1.0, 2.0], [-1.0, 0.0]])
        out = relu(Z)
        assert np.allclose(out, [[1.0, 2.0], [0.0, 0.0]])

    def test_relu_zero(self):
        Z = np.zeros((3, 3))
        assert np.all(relu(Z) == 0)

    def test_relu_shape_preserved(self):
        Z = np.random.randn(4, 5)
        assert relu(Z).shape == Z.shape

    def test_relu_backward_shape(self):
        Z = np.random.randn(3, 4)
        dA = np.ones_like(Z)
        dZ = relu_backward(dA, Z)
        assert dZ.shape == Z.shape

    def test_relu_backward_values(self):
        Z = np.array([[1.0, -1.0], [0.0, 2.0]])
        dA = np.ones_like(Z)
        dZ = relu_backward(dA, Z)
        assert np.allclose(dZ, [[1.0, 0.0], [0.0, 1.0]])

    def test_sigmoid_range(self):
        Z = np.random.randn(10, 10) * 100
        out = sigmoid(Z)
        assert np.all(out > 0) and np.all(out < 1)

    def test_sigmoid_zero(self):
        assert sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)

    def test_sigmoid_large_positive(self):
        assert sigmoid(np.array([100.0]))[0] == pytest.approx(1.0, abs=1e-6)

    def test_sigmoid_large_negative(self):
        assert sigmoid(np.array([-100.0]))[0] == pytest.approx(0.0, abs=1e-6)

    def test_bce_perfect(self):
        y_hat = np.array([1.0, 0.0])
        y = np.array([1.0, 0.0])
        loss = binary_cross_entropy(y_hat, y)
        assert loss == pytest.approx(0.0, abs=0.05)

    def test_bce_worst(self):
        y_hat = np.array([0.0, 1.0])
        y = np.array([1.0, 0.0])
        loss = binary_cross_entropy(y_hat, y)
        assert loss > 10.0  # very high loss

    def test_bce_symmetric(self):
        y_hat = np.array([0.5])
        y = np.array([1.0])
        loss1 = binary_cross_entropy(y_hat, y)
        y2 = np.array([0.0])
        loss2 = binary_cross_entropy(y_hat, y2)
        assert loss1 == pytest.approx(loss2, rel=1e-5)


# ── Network Shapes ────────────────────────────────────────────────────────────

class TestNetworkShapes:

    def setup_method(self):
        self.net = TwoLayerNet(input_size=3, hidden_size=5, seed=42)
        self.X = np.random.randn(3, 10)

    def test_w1_shape(self):
        assert self.net.W1.shape == (5, 3)

    def test_b1_shape(self):
        assert self.net.b1.shape == (5, 1)

    def test_w2_shape(self):
        assert self.net.W2.shape == (1, 5)

    def test_b2_shape(self):
        assert self.net.b2.shape == (1, 1)

    def test_forward_output_shape(self):
        A2 = self.net.forward(self.X)
        assert A2.shape == (1, 10)

    def test_forward_output_range(self):
        A2 = self.net.forward(self.X)
        assert np.all(A2 > 0) and np.all(A2 < 1)

    def test_forward_populates_cache(self):
        self.net.forward(self.X)
        for key in ('Z1', 'A1', 'Z2', 'A2', 'X'):
            assert hasattr(self.net, 'cache') and key in self.net.cache

    def test_backward_grad_keys(self):
        self.net.forward(self.X)
        y = np.ones((1, 10))
        grads = self.net.backward(y)
        for key in ('dW1', 'db1', 'dW2', 'db2'):
            assert key in grads

    def test_backward_grad_shapes(self):
        self.net.forward(self.X)
        y = np.zeros((1, 10))
        grads = self.net.backward(y)
        assert grads['dW1'].shape == (5, 3)
        assert grads['db1'].shape == (5, 1)
        assert grads['dW2'].shape == (1, 5)
        assert grads['db2'].shape == (1, 1)

    def test_predict_shape(self):
        out = self.net.predict(self.X)
        assert out.shape == (1, 10)

    def test_predict_binary(self):
        out = self.net.predict(self.X)
        assert set(np.unique(out)).issubset({0, 1})


# ── Gradient Check ────────────────────────────────────────────────────────────

class TestGradientCheck:
    """Numerical gradient check for W2 (a quick sanity check)."""

    def _loss(self, net, X, y):
        A2 = net.forward(X)
        return binary_cross_entropy(A2.ravel(), y.ravel())

    def test_numerical_gradient_W2(self):
        net = TwoLayerNet(input_size=2, hidden_size=4, seed=1)
        X = np.random.randn(2, 8)
        y = (np.random.rand(1, 8) > 0.5).astype(float)

        net.forward(X)
        grads = net.backward(y)
        dW2_analytic = grads['dW2']

        eps = 1e-5
        dW2_numeric = np.zeros_like(net.W2)
        for i in range(net.W2.shape[0]):
            for j in range(net.W2.shape[1]):
                net.W2[i, j] += eps
                loss_plus = self._loss(net, X, y)
                net.W2[i, j] -= 2 * eps
                loss_minus = self._loss(net, X, y)
                net.W2[i, j] += eps
                dW2_numeric[i, j] = (loss_plus - loss_minus) / (2 * eps)

        rel_err = np.linalg.norm(dW2_analytic - dW2_numeric) / (
            np.linalg.norm(dW2_analytic) + np.linalg.norm(dW2_numeric) + 1e-12
        )
        assert rel_err < 1e-4, f"Gradient check failed: rel_err={rel_err:.2e}"

    def test_numerical_gradient_b1(self):
        net = TwoLayerNet(input_size=2, hidden_size=3, seed=2)
        X = np.random.randn(2, 6)
        y = (np.random.rand(1, 6) > 0.5).astype(float)

        net.forward(X)
        grads = net.backward(y)
        db1_analytic = grads['db1']

        eps = 1e-5
        db1_numeric = np.zeros_like(net.b1)
        for i in range(net.b1.shape[0]):
            net.b1[i, 0] += eps
            loss_plus = self._loss(net, X, y)
            net.b1[i, 0] -= 2 * eps
            loss_minus = self._loss(net, X, y)
            net.b1[i, 0] += eps
            db1_numeric[i, 0] = (loss_plus - loss_minus) / (2 * eps)

        rel_err = np.linalg.norm(db1_analytic - db1_numeric) / (
            np.linalg.norm(db1_analytic) + np.linalg.norm(db1_numeric) + 1e-12
        )
        assert rel_err < 1e-4, f"Gradient check failed: rel_err={rel_err:.2e}"


# ── Training Loop ─────────────────────────────────────────────────────────────

class TestTraining:

    def test_train_returns_loss_list(self):
        net = TwoLayerNet(2, 8, seed=0)
        X, y = make_moons_numpy(100)
        history = train(net, X, y, epochs=5, lr=0.01)
        assert isinstance(history, list)
        assert len(history) == 5

    def test_train_loss_decreases(self):
        net = TwoLayerNet(2, 16, seed=0)
        X, y = make_moons_numpy(300, seed=1)
        history = train(net, X, y, epochs=200, lr=0.1, batch_size=64)
        assert history[-1] < history[0], "Loss should decrease during training"

    def test_train_convergence(self):
        """Network should achieve > 85% accuracy on 2-moons after training."""
        net = TwoLayerNet(2, 32, seed=7)
        X, y = make_moons_numpy(600, noise=0.05, seed=7)
        split = 400
        X_train, X_test = X[:, :split], X[:, split:]
        y_train, y_test = y[:, :split], y[:, split:]

        train(net, X_train, y_train, epochs=500, lr=0.1, batch_size=64)
        preds = net.predict(X_test)
        acc = accuracy(preds, y_test)
        assert acc > 0.85, f"Expected accuracy > 85%, got {acc:.2%}"


# ── Metrics ───────────────────────────────────────────────────────────────────

class TestMetrics:

    def test_accuracy_perfect(self):
        y_pred = np.array([[1, 0, 1, 0]])
        y_true = np.array([[1, 0, 1, 0]])
        assert accuracy(y_pred, y_true) == pytest.approx(1.0)

    def test_accuracy_zero(self):
        y_pred = np.array([[1, 1, 1, 1]])
        y_true = np.array([[0, 0, 0, 0]])
        assert accuracy(y_pred, y_true) == pytest.approx(0.0)

    def test_accuracy_half(self):
        y_pred = np.array([[1, 0, 1, 0]])
        y_true = np.array([[0, 0, 0, 0]])
        assert accuracy(y_pred, y_true) == pytest.approx(0.5)

    def test_roc_auc_perfect(self):
        y_scores = np.array([0.9, 0.8, 0.2, 0.1])
        y_true = np.array([1, 1, 0, 0])
        auc = roc_auc(y_scores, y_true)
        assert auc == pytest.approx(1.0)

    def test_roc_auc_random(self):
        rng = np.random.default_rng(0)
        y_scores = rng.random(100)
        y_true = rng.integers(0, 2, 100)
        auc = roc_auc(y_scores, y_true)
        assert 0.3 < auc < 0.7  # should be near 0.5 for random

    def test_roc_auc_range(self):
        rng = np.random.default_rng(1)
        y_scores = rng.random(200)
        y_true = rng.integers(0, 2, 200)
        auc = roc_auc(y_scores, y_true)
        assert 0.0 <= auc <= 1.0


# ── Data Generation ───────────────────────────────────────────────────────────

class TestDataGen:

    def test_moons_shape(self):
        X, y = make_moons_numpy(100)
        assert X.shape == (2, 100)
        assert y.shape == (1, 100)

    def test_moons_binary(self):
        X, y = make_moons_numpy(100)
        assert set(np.unique(y)).issubset({0.0, 1.0})

    def test_moons_balanced(self):
        X, y = make_moons_numpy(200)
        assert np.sum(y == 0) == 100
        assert np.sum(y == 1) == 100

    def test_moons_reproducible(self):
        X1, y1 = make_moons_numpy(100, seed=5)
        X2, y2 = make_moons_numpy(100, seed=5)
        assert np.allclose(X1, X2)
