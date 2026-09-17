"""A/B ablation: how much does the connectome read-out shape behaviour?

Usage:
    uv run python tools/ablation.py --ticks 3000 --seed 1
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import ACTIONS_CN  # noqa: E402
from src.main import Simulation, load_config  # noqa: E402


def run(cfg, ticks: int, seed: int) -> dict:
    sim = Simulation(cfg, seed=seed)
    for _ in range(ticks):
        sim.tick()
    s = sim.stats()
    acts = np.array(s["action_counts"], dtype=float)
    p = acts / max(acts.sum(), 1.0)
    ent = float(-(p * np.log(p + 1e-9)).sum())
    out = {"deaths": s["deaths"], "hp": s["agent"]["hp"], "energy": s["agent"]["energy"],
           "hz": s["net_hz"], "gain": s["gain"], "entropy": ent,
           "actions": {ACTIONS_CN[i]: int(acts[i]) for i in range(len(ACTIONS_CN))},
           "fires": s["fire_tiles"], "events": s["event_counts"]}
    sim.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    base, _ = load_config()
    base.setdefault("population", {})["size"] = 1
    off = deepcopy(base); off["decision"]["network_gain"] = 0.0
    on = deepcopy(base)
    a = run(off, args.ticks, args.seed)
    b = run(on, args.ticks, args.seed)
    print(f"ticks={args.ticks} seed={args.seed}")
    print(f"{'network_gain':>14} {'deaths':>6} {'HP':>6} {'energy':>7} {'DN Hz':>6} {'gain':>5} {'action entropy':>14}")
    print(f"{'0.00 (ablate)':>14} {a['deaths']:6d} {a['hp']:6.1f} {a['energy']:7.1f} {a['hz']:6.2f} {a['gain']:5.2f} {a['entropy']:14.3f}")
    print(f"{base['decision']['network_gain']:>14.2f} {b['deaths']:6d} {b['hp']:6.1f} {b['energy']:7.1f} {b['hz']:6.2f} {b['gain']:5.2f} {b['entropy']:14.3f}")
    print("\n动作分布（network off）:", a["actions"])
    print("动作分布（network on ）:", b["actions"])


if __name__ == "__main__":
    main()