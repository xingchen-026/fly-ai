"""Deterministic math helpers: integer hashing, value noise, fBm.

All random world content is a pure function of (x, y, seed), so the world can be
infinite and never needs to be stored, only cached in chunks.
"""
from __future__ import annotations

import numpy as np

_INT64 = np.int64
_MASK31 = _INT64(0x7FFFFFFFFFFFFFFF)


def _i64(a):
    return np.asarray(a, dtype=np.int64)


def hash64(x, y, seed: int = 0) -> np.ndarray:
    """Vectorised 64-bit integer hash of lattice coordinates and a seed."""
    x = _i64(x)
    y = _i64(y)
    with np.errstate(over="ignore"):
        h = x * _INT64(374761393) + y * _INT64(668265263) + _INT64(int(seed) & 0xFFFFFFFF) * _INT64(1442695040)
        h &= _MASK31
        h ^= h >> _INT64(13)
        h *= _INT64(1274126177)
        h &= _MASK31
        h ^= h >> _INT64(16)
    return h


def hash01(x, y, seed: int = 0) -> np.ndarray:
    """Deterministic pseudo-random floats in [0, 1) for integer coordinates."""
    h = hash64(x, y, seed)
    return (h & _INT64(0xFFFFFF)).astype(np.float64) / 16777216.0


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def lerp(a, b, t):
    return a + (b - a) * t


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def clamp01(v):
    return min(1.0, max(0.0, v))


def value_noise2(x, y, seed: int = 0, scale: float = 32.0) -> np.ndarray:
    """Smooth value noise in [0, 1] on an infinite lattice."""
    x = np.asarray(x, dtype=np.float64) / float(scale)
    y = np.asarray(y, dtype=np.float64) / float(scale)
    x0 = np.floor(x)
    y0 = np.floor(y)
    tx = x - x0
    ty = y - y0
    x0i = x0.astype(np.int64)
    y0i = y0.astype(np.int64)
    sx = smoothstep(tx)
    sy = smoothstep(ty)
    v00 = hash01(x0i, y0i, seed)
    v10 = hash01(x0i + 1, y0i, seed)
    v01 = hash01(x0i, y0i + 1, seed)
    v11 = hash01(x0i + 1, y0i + 1, seed)
    top = lerp(v00, v10, sx)
    bottom = lerp(v01, v11, sx)
    return lerp(top, bottom, sy)


def fbm2(x, y, seed: int = 0, scale: float = 64.0, octaves: int = 4,
         gain: float = 0.5, lacunarity: float = 2.0) -> np.ndarray:
    """Fractal Brownian motion built from value noise; roughly in [0, 1]."""
    total = np.zeros_like(np.asarray(x, dtype=np.float64))
    amp = 1.0
    norm = 0.0
    freq_scale = float(scale)
    for i in range(int(octaves)):
        total = total + amp * value_noise2(x, y, seed=seed + 101 * i, scale=freq_scale)
        norm += amp
        amp *= gain
        freq_scale /= lacunarity
    return total / max(norm, 1e-9)


def unit_dir(dx: float, dy: float):
    """Normalise a displacement to the [-1, 1] range (Chebyshev style)."""
    m = max(abs(dx), abs(dy), 1.0)
    return dx / m, dy / m