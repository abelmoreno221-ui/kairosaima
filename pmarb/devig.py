"""Quitar el margen (vig) de cuotas decimales -> probabilidades justas."""
import numpy as np


def _bisect(f, lo, hi, it=100):
    for _ in range(it):
        mid = (lo + hi) / 2
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def multiplicative(odds):
    q = 1 / np.asarray(odds, float)
    return q / q.sum()


def power(odds):
    q = 1 / np.asarray(odds, float)
    k = _bisect(lambda k: (q ** k).sum() - 1, 1.0, 10.0)
    return q ** k


def shin(odds):
    """Método de Shin: corrige el sesgo favorito-longshot (recomendado para 1X2)."""
    q = 1 / np.asarray(odds, float)
    Q = q.sum()
    if Q <= 1:
        return q / Q

    def probs(z):
        return (np.sqrt(z ** 2 + 4 * (1 - z) * q ** 2 / Q) - z) / (2 * (1 - z))

    z = _bisect(lambda z: probs(z).sum() - 1, 0.0, 0.4)
    p = probs(z)
    return p / p.sum()


# "log" = método logarítmico de Buchdahl (p_i = q_i^k), equivalente al power.
METHODS = {"shin": shin, "prop": multiplicative, "log": power}
