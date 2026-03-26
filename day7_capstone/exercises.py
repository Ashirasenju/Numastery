"""
DAY 7 — Capstone: Neural Network from Scratch
===============================================
Build a fully-vectorised 2-layer neural network for binary classification
using ONLY NumPy.  No loops over samples allowed in forward/backward pass.

Architecture
------------
  Input (d features)
    → Dense(hidden_size)  + ReLU
    → Dense(1)            + Sigmoid
    → Binary cross-entropy loss

You will implement:
  • Weight initialisation (He init)
  • Forward pass
  • Loss computation
  • Backward pass (backprop)
  • Parameter update (SGD)
  • Mini-batch training loop
  • Accuracy & AUC-like metric

Run: pytest day7_capstone/test_exercises.py -v

This is the hardest day.  Take your time, draw the computation graph,
and verify shapes at every step.
"""

import numpy as np


# ── Helper: Activations & Loss ────────────────────────────────────────────────

def relu(Z: np.ndarray) -> np.ndarray:
    """ReLU activation: max(0, Z), element-wise."""
    raise NotImplementedError


def relu_backward(dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """
    Gradient of ReLU.
    dL/dZ = dL/dA * (Z > 0)
    """
    raise NotImplementedError


def sigmoid(Z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid: 1 / (1 + exp(-Z))."""
    raise NotImplementedError


def binary_cross_entropy(y_hat: np.ndarray, y: np.ndarray,
                         eps: float = 1e-12) -> float:
    """
    Compute mean binary cross-entropy loss.
    L = -mean(y * log(y_hat + eps) + (1 - y) * log(1 - y_hat + eps))
    """
    raise NotImplementedError


# ── Neural Network Class ──────────────────────────────────────────────────────

class TwoLayerNet:
    """
    Two-layer (one hidden) neural network for binary classification.

    Parameters
    ----------
    input_size  : number of features
    hidden_size : number of hidden units
    seed        : for reproducibility
    """

    def __init__(self, input_size: int, hidden_size: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        # He initialisation: W ~ N(0, sqrt(2/fan_in))
        # TODO: initialise W1 (hidden_size, input_size), b1 (hidden_size, 1)
        #                   W2 (1, hidden_size),         b2 (1, 1)
        raise NotImplementedError

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Parameters
        ----------
        X : shape (d, N)  — d features, N samples

        Returns
        -------
        A2 : shape (1, N) — predicted probabilities

        Also cache intermediate values needed for backprop:
          self.cache = {'Z1', 'A1', 'Z2', 'A2', 'X'}
        """
        raise NotImplementedError

    def backward(self, y: np.ndarray) -> dict:
        """
        Backward pass (backpropagation).

        Parameters
        ----------
        y : shape (1, N) — ground-truth labels {0, 1}

        Returns
        -------
        grads : dict with keys 'dW1', 'db1', 'dW2', 'db2'

        Derivations (N = batch size):
          dZ2 = A2 - y                          shape (1, N)
          dW2 = dZ2 @ A1.T / N
          db2 = mean(dZ2, axis=1, keepdims=True)
          dA1 = W2.T @ dZ2
          dZ1 = relu_backward(dA1, Z1)
          dW1 = dZ1 @ X.T / N
          db1 = mean(dZ1, axis=1, keepdims=True)
        """
        raise NotImplementedError

    def update(self, grads: dict, lr: float) -> None:
        """SGD update: param -= lr * grad."""
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return predicted probabilities for X (shape (d, N))."""
        raise NotImplementedError

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions (0 or 1) for X."""
        raise NotImplementedError


# ── Training Loop ─────────────────────────────────────────────────────────────

def train(net: TwoLayerNet, X_train: np.ndarray, y_train: np.ndarray,
          epochs: int = 100, lr: float = 0.01,
          batch_size: int = 64, seed: int = 0) -> list:
    """
    Mini-batch SGD training loop.

    Parameters
    ----------
    net      : TwoLayerNet instance
    X_train  : shape (d, N_train)
    y_train  : shape (1, N_train)
    epochs   : number of full passes over the data
    lr       : learning rate
    batch_size : samples per mini-batch
    seed     : for shuffling

    Returns
    -------
    loss_history : list of mean loss per epoch
    """
    raise NotImplementedError


# ── Metrics ───────────────────────────────────────────────────────────────────

def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Return classification accuracy (fraction of correct predictions)."""
    raise NotImplementedError


def roc_auc(y_scores: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute the ROC-AUC score without sklearn.

    Algorithm (trapezoid rule):
      Sort by descending score.
      Walk through thresholds computing (FPR, TPR) at each point.
      Integrate using np.trapz.
    """
    raise NotImplementedError


# ── Data Generation ───────────────────────────────────────────────────────────

def make_moons_numpy(n_samples: int = 500, noise: float = 0.1,
                     seed: int = 42) -> tuple:
    """
    Generate a 2-D 'two moons' dataset without sklearn.

    Returns X of shape (2, n_samples) and y of shape (1, n_samples).
    Each moon has n_samples // 2 points.

    Moon 0: x = cos(θ),        y = sin(θ),        θ ∈ [0, π]
    Moon 1: x = 1 - cos(θ),    y = 0.5 - sin(θ),  θ ∈ [0, π]
    """
    rng = np.random.default_rng(seed)
    n = n_samples // 2
    theta = np.linspace(0, np.pi, n)

    moon0 = np.stack([np.cos(theta), np.sin(theta)])
    moon1 = np.stack([1 - np.cos(theta), 0.5 - np.sin(theta)])

    X = np.concatenate([moon0, moon1], axis=1)
    X += rng.normal(0, noise, X.shape)

    y = np.concatenate([np.zeros(n), np.ones(n)])[np.newaxis, :]
    idx = rng.permutation(n_samples)
    return X[:, idx], y[:, idx]
