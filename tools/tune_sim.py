"""Scan LIF excitation gain against the full simulation and report behaviour.

Usage:
    uv run python tools/tune_sim.py 12 18 24 30 42 --ticks 1500 --seed 1
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import ACTIONS_CN  # noqa: E402
from src.main import Simulation, load_config  # noqa: E402


def run_one(cfg, ticks: int, seed: int):
    sim = Simulation(cfg, seed=seed)
    sensed = {"n": 0, "sum": 0.0, "input_hz": 0.0, "samples": 0}

    orig_sense = sim.agent.sense

    def wrapped(*a, **k):
        f = orig_sense(*a, **k)
        sensed["n"] += 1
        sensed["sum"] += float(f.sum())
        return f

    sim.agent.sense = wrapped  # type: ignore
    for _ in range(ticks):
        sim.tick()
        sensed["input_hz"] += float(sim.net.rate[sim.net.input_idx].mean())
        sensed["samples"] += 1
    s = sim.stats()
    a = s["agent"]
    hist = list(sim.history)[-300:]
    out = {
        "deaths": s["deaths"], "hp": a["hp"], "energy": a["energy"], "thirst": a["thirst"],
        "temp": a["temp"], "hz_last": st.fmean(h[5] for h in hist) if hist else 0.0,
        "gain": s["gain"], "gain_min": min(h[6] for h in sim.history), "gain_max": max(h[6] for h in sim.history),
        "input_hz": sensed["input_hz"] / max(sensed["samples"], 1),
        "feat_active": sensed["sum"] / max(sensed["n"], 1),
        "actions": s["action_counts"], "events": s["event_counts"],
    }
    sim.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gains", nargs="*", type=float, default=[12, 18, 24, 30, 42])
    ap.add_argument("--noise", type=float, default=None)
    ap.add_argument("--ticks", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    base, _ = load_config()
    base.setdefault("population", {})["size"] = 1
    print(f"{'gain':>6} {'noise':>6} | {'deaths':>6} {'hp':>5} {'en':>5} {'th':>5} {'T':>5} | "
          f"{'DN Hz':>6} {'in Hz':>6} {'feat':>5} | {'homeo g':>8} | actions " + "/".join(ACTIONS_CN))
    for g in args.gains:
        cfg = deepcopy(base)
        cfg["neural"]["syn_gain"] = float(g)
        if args.noise is not None:
            cfg["neural"]["lif"]["noise_sigma"] = float(args.noise)
        r = run_one(cfg, args.ticks, args.seed)
        act = "/".join(str(int(x)) for x in r["actions"])
        print(f"{g:6.1f} {cfg['neural']['lif']['noise_sigma']:6.2f} | {r['deaths']:6d} {r['hp']:5.1f} "
              f"{r['energy']:5.1f} {r['thirst']:5.1f} {r['temp']:5.1f} | {r['hz_last']:6.2f} {r['input_hz']:6.2f} "
              f"{r['feat_active']:5.1f} | {r['gain_min']:.2f}..{r['gain_max']:.2f} | {act}")


if __name__ == "__main__":
    main()