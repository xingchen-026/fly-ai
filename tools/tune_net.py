"""Scan LIF parameters (syn_gain / noise) and report equilibrium activity.

Usage:
    uv run python tools/tune_net.py
"""
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.neural import ConnectomeNet  # noqa: E402

STEPS = 20  # neural steps per game tick


def run(cfg, syn_gain: float, noise: float, ticks: int = 180):
    c = deepcopy(cfg)
    c["neural"]["syn_gain"] = syn_gain
    c["neural"]["lif"]["noise_sigma"] = noise
    rng = np.random.default_rng(3)
    net = ConnectomeNet(c, ROOT / "data", np.random.default_rng(11))
    F = int(c["neural"]["n_features"])
    quiet = np.zeros(F, dtype=np.float32)
    burst = (rng.random(F) < 0.12).astype(np.float32)
    rates = {}
    for phase, feats in (("quiet", quiet), ("quiet", quiet), ("burst", burst)):
        for _ in range(ticks // 3):
            net.encode(feats)
            net.step(STEPS, net._iext, rng)
            net.tick_update()
        rates[phase] = (float(net.mean_dn_hz()), float(net.rate[net.input_idx].mean()),
                        float((net.rate > 1.0).sum()), net.gain)
    return rates


def main():
    cfg = json.load(open(ROOT / "src" / "config.json", encoding="utf-8"))
    print(f"{'syn_gain':>8} {'noise':>6} | {'quiet DN':>8} {'quiet in':>8} {'active':>6} {'gain':>6} | "
          f"{'burst DN':>8} {'burst in':>8} {'active':>6} {'gain':>6}")
    for syn_gain in (8.0, 12.0, 18.0, 26.0, 42.0):
        for noise in (1.0, 1.6, 2.2):
            r = run(cfg, syn_gain, noise)
            q, b = r["quiet"], r["burst"]
            print(f"{syn_gain:8.1f} {noise:6.2f} | {q[0]:8.2f} {q[1]:8.2f} {q[2]:6.0f} {q[3]:6.2f} | "
                  f"{b[0]:8.2f} {b[1]:8.2f} {b[2]:6.0f} {b[3]:6.2f}")


if __name__ == "__main__":
    main()