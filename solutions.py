"""
SOLUTIONS — NumPy Mastery
==========================
⚠️  Try to solve the exercises yourself before reading this file!
    The real learning happens when you struggle with the problems.

To check your work, run the tests first:
  pytest dayN_*/test_exercises.py -v

Only open this file after a genuine attempt or when totally stuck.
"""

import numpy as np
import numpy.lib.stride_tricks as stride_tricks


# ═══════════════════════════════════════════════════════════════════════════════
# DAY 1 — Array Basics
# ═══════════════════════════════════════════════════════════════════════════════

def zeros_like_int(shape): return np.zeros(shape, dtype=np.int32)
def create_range(start, stop, n): return np.linspace(start, stop, n)
def identity_block(n, k): return np.eye(n, k=k)
def build_checkerboard(n):
    idx = np.arange(n)
    i, j = np.meshgrid(idx, idx, indexing='ij')
    return ((i + j) % 2 == 0).astype(np.uint8)

def describe(arr):
    return {
        'shape': arr.shape, 'ndim': arr.ndim, 'size': arr.size,
        'dtype': str(arr.dtype), 'itemsize': arr.itemsize, 'nbytes': arr.nbytes,
    }

def safe_cast(arr, target_dtype):
    try:
        return arr.astype(target_dtype, casting='same_kind')
    except TypeError:
        return arr

def flatten_and_sort(arr): return np.sort(arr.flatten())
def stack_as_matrix(arrays): return np.vstack(arrays)
def tile_border(inner, pad): return np.pad(inner, pad)
def make_view(arr): return arr.reshape(-1)
def is_view_of(candidate, base): return np.shares_memory(candidate, base)
def forced_copy(arr): return np.ascontiguousarray(arr.copy())


# ═══════════════════════════════════════════════════════════════════════════════
# DAY 2 — Indexing & Masking
# ═══════════════════════════════════════════════════════════════════════════════

def extract_submatrix(arr, r0, r1, c0, c1): return arr[r0:r1, c0:c1]
def every_other_row(arr): return arr[::2]
def reverse_columns(arr): return arr[:, ::-1]
def diagonal_sum(arr): return float(np.trace(arr))

def select_rows(arr, indices): return arr[indices]
def set_diagonal(arr, value):
    out = arr.copy()
    np.fill_diagonal(out, value)
    return out

def scatter_add(base, indices, values):
    out = base.copy()
    np.add.at(out, indices, values)
    return out

def clamp(arr, lo, hi): return np.clip(arr, lo, hi)
def replace_outliers(arr, sigma=2.0):
    out = arr.astype(np.float64).copy()
    mu, std = out.mean(), out.std(ddof=1)
    mask = np.abs(out - mu) > sigma * std
    out[mask] = mu
    return out

def mask_nan_inf(arr):
    out = arr.copy()
    out[~np.isfinite(out)] = 0
    return out

def sign_array(arr): return np.sign(arr)
def first_nonzero_index(arr):
    idx = np.nonzero(arr)[0]
    return int(idx[0]) if len(idx) > 0 else -1

def top_k_indices(arr, k):
    flat = arr.ravel()
    part = np.argpartition(flat, -k)[-k:]
    return part[np.argsort(flat[part])[::-1]]


# ═══════════════════════════════════════════════════════════════════════════════
# DAY 3 — Math & Broadcasting
# ═══════════════════════════════════════════════════════════════════════════════

def running_sum(arr): return np.add.accumulate(arr)
def outer_product_sum(a, b): return float(np.multiply.outer(a, b).sum())
def log_sum_exp(arr):
    m = arr.max()
    return float(m + np.log(np.sum(np.exp(arr - m))))

def custom_relu_ufunc(arr):
    relu = np.frompyfunc(lambda x: max(x, 0), 1, 1)
    return relu(arr).astype(arr.dtype)

def row_normalize(matrix):
    m = matrix.astype(np.float64)
    return m / m.sum(axis=1, keepdims=True)

def pairwise_l2(A, B):
    A2 = np.sum(A**2, axis=1, keepdims=True)  # (m,1)
    B2 = np.sum(B**2, axis=1)                  # (n,)
    AB = A @ B.T                                # (m,n)
    dist2 = A2 + B2 - 2 * AB
    return np.sqrt(np.clip(dist2, 0, None))

def add_bias(X, bias): return X + bias
def outer_subtract(a, b): return a[:, np.newaxis] - b[np.newaxis, :]

def softmax(arr):
    e = np.exp(arr - arr.max())
    return e / e.sum()

def moving_average(arr, window):
    cs = np.cumsum(arr)
    cs[window:] = cs[window:] - cs[:-window]
    return cs[window - 1:] / window

def polynomial_eval(coeffs, x):
    return np.polyval(coeffs[::-1], x)


# ═══════════════════════════════════════════════════════════════════════════════
# DAY 4 — Linear Algebra
# ═══════════════════════════════════════════════════════════════════════════════

def matmul_chain(*matrices):
    if len(matrices) == 2:
        return matrices[0] @ matrices[1]
    return np.linalg.multi_dot(matrices)

def gram_matrix(X): return X @ X.T
def solve_linear_system(A, b): return np.linalg.solve(A, b)

def principal_components(X, k):
    Xc = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:k]

def is_positive_definite(A):
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False

def qr_solve(A, b):
    Q, R = np.linalg.qr(A, mode='reduced')
    return np.linalg.solve(R, Q.T @ b)

def spectral_radius(A):
    return float(np.max(np.abs(np.linalg.eigvals(A))))

def matrix_power_via_eigen(A, p):
    vals, vecs = np.linalg.eigh(A)
    return np.real(vecs @ np.diag(vals**p) @ vecs.T)

def frobenius_norm(A): return float(np.linalg.norm(A, 'fro'))
def condition_number(A): return float(np.linalg.cond(A))
def pseudo_inverse_solve(A, b): return np.linalg.pinv(A) @ b

def low_rank_approximation(A, rank):
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return U[:, :rank] @ np.diag(s[:rank]) @ Vt[:rank, :]


# ═══════════════════════════════════════════════════════════════════════════════
# DAY 5 — Statistics & Random
# ═══════════════════════════════════════════════════════════════════════════════

def summary_stats(arr):
    return {
        'mean': float(np.mean(arr)), 'std': float(np.std(arr, ddof=1)),
        'min': float(np.min(arr)), 'max': float(np.max(arr)),
        'median': float(np.median(arr)),
        'q25': float(np.percentile(arr, 25)),
        'q75': float(np.percentile(arr, 75)),
    }

def weighted_average(values, weights): return float(np.average(values, weights=weights))
def z_score(arr):
    return (arr - arr.mean()) / arr.std(ddof=1)

def correlation_matrix(X): return np.corrcoef(X.T)

def histogram_mode(arr, bins=10):
    counts, edges = np.histogram(arr, bins=bins)
    peak = np.argmax(counts)
    return float((edges[peak] + edges[peak + 1]) / 2)

def digitize_labels(arr, bins): return np.digitize(arr, bins)

def reproducible_sample(n, seed=42):
    return np.random.default_rng(seed).standard_normal(n)

def bootstrap_mean_ci(data, n_bootstrap=2000, ci=0.95, seed=0):
    rng = np.random.default_rng(seed)
    n = len(data)
    means = np.array([
        rng.choice(data, size=n, replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    alpha = (1 - ci) / 2
    return float(np.percentile(means, 100 * alpha)), float(np.percentile(means, 100 * (1 - alpha)))

def random_walk(n_steps, seed=0):
    rng = np.random.default_rng(seed)
    steps = rng.choice([-1, 1], size=n_steps)
    return np.concatenate([[0], np.cumsum(steps)])

def estimate_pi(n_samples=1_000_000, seed=42):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-1, 1, (2, n_samples))
    inside = (pts[0]**2 + pts[1]**2) <= 1
    return 4.0 * inside.sum() / n_samples

def option_price_mc(S0, K, r, sigma, T, n_paths=100_000, seed=42):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)
    S_T = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoff = np.maximum(S_T - K, 0)
    return float(np.exp(-r * T) * payoff.mean())


# ═══════════════════════════════════════════════════════════════════════════════
# DAY 6 — Advanced Internals
# ═══════════════════════════════════════════════════════════════════════════════

STUDENT_DTYPE = np.dtype([('name', 'U20'), ('age', np.int32), ('gpa', np.float64)])

def create_student_array(records):
    return np.array(records, dtype=STUDENT_DTYPE)

def top_students(students, n):
    idx = np.argsort(students['gpa'])[::-1][:n]
    return students[idx]

def gpa_above(students, threshold):
    return students['name'][students['gpa'] > threshold]

def get_strides_info(arr):
    return {
        'strides': arr.strides, 'itemsize': arr.itemsize,
        'c_contiguous': bool(arr.flags['C_CONTIGUOUS']),
        'f_contiguous': bool(arr.flags['F_CONTIGUOUS']),
    }

def sliding_window_view(arr, window):
    return np.lib.stride_tricks.sliding_window_view(arr, window)

def as_fortran(arr): return np.asfortranarray(arr)

def einsum_trace(A): return float(np.einsum('ii->', A))
def einsum_matmul(A, B): return np.einsum('ik,kj->ij', A, B)
def einsum_batch_dot(A, B): return np.einsum('bij,bjk->bik', A, B)
def einsum_outer(a, b): return np.einsum('i,j->ij', a, b)

def masked_mean(arr, fill_value=np.nan):
    return float(np.ma.masked_invalid(arr).mean())

def interpolate_missing(arr):
    out = arr.copy()
    nans = np.isnan(out)
    x_valid = np.where(~nans)[0]
    y_valid = out[~nans]
    out[nans] = np.interp(np.where(nans)[0], x_valid, y_valid)
    return out

def normalise_strings(arr):
    return np.char.lower(np.char.strip(arr))

def count_vowels(arr):
    vowels = 'aeiou'
    counts = np.zeros(len(arr), dtype=int)
    for v in vowels:
        counts += np.char.count(arr, v)
    return counts


# ═══════════════════════════════════════════════════════════════════════════════
# DAY 7 — Capstone Neural Network
# ═══════════════════════════════════════════════════════════════════════════════

def relu(Z): return np.maximum(0, Z)
def relu_backward(dA, Z): return dA * (Z > 0)
def sigmoid(Z): return 1 / (1 + np.exp(-np.clip(Z, -500, 500)))

def binary_cross_entropy(y_hat, y, eps=1e-12):
    return float(-np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps)))


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((hidden_size, input_size)) * np.sqrt(2 / input_size)
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = rng.standard_normal((1, hidden_size)) * np.sqrt(2 / hidden_size)
        self.b2 = np.zeros((1, 1))
        self.cache = {}

    def forward(self, X):
        Z1 = self.W1 @ X + self.b1
        A1 = relu(Z1)
        Z2 = self.W2 @ A1 + self.b2
        A2 = sigmoid(Z2)
        self.cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2, 'X': X}
        return A2

    def backward(self, y):
        N = y.shape[1]
        A2, A1, Z1, X = self.cache['A2'], self.cache['A1'], self.cache['Z1'], self.cache['X']
        dZ2 = A2 - y
        dW2 = dZ2 @ A1.T / N
        db2 = np.mean(dZ2, axis=1, keepdims=True)
        dA1 = self.W2.T @ dZ2
        dZ1 = relu_backward(dA1, Z1)
        dW1 = dZ1 @ X.T / N
        db1 = np.mean(dZ1, axis=1, keepdims=True)
        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}

    def update(self, grads, lr):
        self.W1 -= lr * grads['dW1']
        self.b1 -= lr * grads['db1']
        self.W2 -= lr * grads['dW2']
        self.b2 -= lr * grads['db2']

    def predict_proba(self, X): return self.forward(X)
    def predict(self, X, threshold=0.5): return (self.predict_proba(X) >= threshold).astype(int)


def train(net, X_train, y_train, epochs=100, lr=0.01, batch_size=64, seed=0):
    rng = np.random.default_rng(seed)
    N = X_train.shape[1]
    history = []
    for _ in range(epochs):
        idx = rng.permutation(N)
        X_s, y_s = X_train[:, idx], y_train[:, idx]
        epoch_losses = []
        for start in range(0, N, batch_size):
            Xb = X_s[:, start:start + batch_size]
            yb = y_s[:, start:start + batch_size]
            A2 = net.forward(Xb)
            loss = binary_cross_entropy(A2.ravel(), yb.ravel())
            grads = net.backward(yb)
            net.update(grads, lr)
            epoch_losses.append(loss)
        history.append(float(np.mean(epoch_losses)))
    return history


def accuracy(y_pred, y_true):
    return float(np.mean(y_pred == y_true))


def roc_auc(y_scores, y_true):
    order = np.argsort(y_scores)[::-1]
    y_sorted = y_true[order]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    tpr = tps / tps[-1]
    fpr = fps / fps[-1]
    return float(np.trapz(tpr, fpr))


def make_moons_numpy(n_samples=500, noise=0.1, seed=42):
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
