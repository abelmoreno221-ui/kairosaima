"""Calibración y scoring. Sin ML pesado: regresión logística sobre logits (2-3 parámetros)."""
import numpy as np

EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def log_loss(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(p, y, bins=10):
    """Expected Calibration Error."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def reliability(p, y, bins=10):
    p, y = np.asarray(p, float), np.asarray(y, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return [(b / bins, (b + 1) / bins, int((idx == b).sum()), float(p[idx == b].mean()), float(y[idx == b].mean()))
            for b in range(bins) if (idx == b).any()]


def report(p, y):
    return {"n": len(y), "brier": brier(p, y), "logloss": log_loss(p, y), "ece": ece(p, y)}


class Blend:
    """p = sigmoid(w0 + w1*logit(p_pm) + w2*logit(p_book)). Ajuste IRLS con L2.
    w1 vs w2 te dice quién 'tiene razón': si w2 >> w1, la casa (sharp) manda."""

    def __init__(self, l2=30.0):
        self.l2, self.w = l2, None

    @staticmethod
    def _X(p_pm, p_book):
        return np.column_stack([np.ones(len(p_pm)), logit(p_pm), logit(p_book)])

    def fit(self, p_pm, p_book, y, iters=50):
        X, y = self._X(p_pm, p_book), np.asarray(y, float)
        w = np.array([0.0, 0.5, 0.5])
        R = self.l2 * np.diag([0, 1, 1])
        for _ in range(iters):
            p = sigmoid(X @ w)
            g = X.T @ (p - y) + R @ (w - [0, 0.5, 0.5])
            H = X.T @ (X * (p * (1 - p))[:, None]) + R + 1e-9 * np.eye(3)
            step = np.linalg.solve(H, g)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        self.w = w
        return self

    def predict(self, p_pm, p_book):
        return sigmoid(self._X(p_pm, p_book) @ self.w)
