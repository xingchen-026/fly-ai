"""Does plasticity help? Compare three settings over identical worlds (single agent,
evolution disabled so that only the synaptic/decode plasticity differs).

Usage:
    uv run python tools/learning_ablation.py --ticks 3000 --seed 1

A) no plasticity      - fixed synapses, fixed decode
B) R-STDP only        - reward-modulated STDP eligibility, fixed decode
C) R-STDP + decode    - default: synapses + plastic decode layer
"""
from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.main import Simulation, load_config  # noqa: E402


def variant(base: dict, stdp: bool, neuromod: bool, readout: bool) -> dict:
    c = deepcopy(base)
    c["neural"]["plasticity"]["stdp"]["enabled"] = bool(stdp)
    c["neural"]["plasticity"]["neuromod"]["enabled"] = bool(neuromod)
    c["neural"]["plasticity"]["readout"]["enabled"] = bool(readout)
    return c


def run(cfg: dict, ticks: int, seed: int) -> dict:
    sim = Simulation(cfg, seed=seed)
    for _ in range(ticks):
        sim.tick()
    s = sim.stats()
    a = s["agent"]
    learn = s["learning"]
    acts = np.array(s["action_counts"], dtype=float)
    p = acts / max(acts.sum(), 1.0)
    ent = float(-(p * np.log(p + 1e-9)).sum())
    hist = list(sim.history)[-400:]
    if hist:
        hp = float(np.mean([h[1] for h in hist]))
        energy = float(np.mean([h[2] for h in hist]))
        thirst = float(np.mean([h[3] for h in hist]))
    else:
        hp = energy = thirst = 0.0
    out = {"deaths": s["deaths"], "hp_mean": hp, "energy_mean": energy, "thirst_mean": thirst,
           "eat": s["counters"]["eat"], "hunt": s["counters"]["hunt"],
           "drink": s["counters"]["drink"], "bite": s["counters"]["bite"],
           "poison": s["counters"]["poison"], "fire": s["counters"]["fire"],
           "entropy": ent, "w_change": learn["w_change"], "readout": learn["readout_drift"],
           "hz": s["net_hz"], "distance": a["distance"],
           "counters": dict(s["counters"])}
    sim.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--seeds", default=None, help="comma separated seeds for aggregated runs")
    ap.add_argument("--out", default=None, help="also write the table to this file (utf-8)")
    args = ap.parse_args()
    lines = []
    base, _ = load_config()
    base.setdefault("population", {})["size"] = 1  # keep comparisons single-agent
    # isolate plasticity from neuroevolution: no cloning/mutation during A/B
    base["population"].setdefault("evolution", {})["enabled"] = False
    rows = [
        ("A no-plasticity", variant(base, False, False, False)),
        ("B R-STDP", variant(base, True, True, False)),
        ("C R-STDP+decode", variant(base, True, True, True)),
    ]
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else [args.seed]

    def emit(line=""):
        print(line)
        lines.append(line)
    if len(seeds) == 1:
        emit(f"ticks={args.ticks} seed={seeds[0]}")
        emit(f"{'setting':<18}{'deaths':>7}{'meanHP':>8}{'energy':>8}{'water':>7}"
              f"{'eat':>5}{'hunt':>5}{'bite':>5}{'dist':>6}{'entropy':>8}{'dw%':>7}{'decode':>8}")
    else:
        emit(f"ticks={args.ticks} seeds={seeds}  (mean +/- std)")
        emit(f"{'setting':<18}{'deaths':>8}{'meanHP':>12}{'energy':>12}{'eat':>10}{'hunt':>8}"
              f"{'bite':>9}{'entropy':>14}{'dw%':>11}{'decode':>10}")
    import atexit
    if args.out:
        atexit.register(_write_out, args.out, lines)
    for name, cfg in rows:
        rs = [run(cfg, args.ticks, s) for s in seeds]
        if len(seeds) == 1:
            r = rs[0]
            emit(f"{name:<18}{r['deaths']:>7}{r['hp_mean']:>8.1f}{r['energy_mean']:>8.1f}"
                  f"{r['thirst_mean']:>7.1f}{r['eat']:>5}{r['hunt']:>5}{r['bite']:>5}"
                  f"{r['distance']:>6}{r['entropy']:>8.3f}{r['w_change'] * 100:>7.2f}{r['readout']:>8.3f}")
        else:
            def agg(key):
                v = np.array([x[key] for x in rs], dtype=float)
                return v.mean(), v.std()
            d, hp, en = agg("deaths"), agg("hp_mean"), agg("energy_mean")
            eat, hunt, bite = agg("eat"), agg("hunt"), agg("bite")
            ent, dw, dec = agg("entropy"), agg("w_change"), agg("readout")
            emit(f"{name:<18}{d[0]:>5.1f}±{d[1]:<2.1f}{hp[0]:>8.1f}±{hp[1]:<3.1f}"
                  f"{en[0]:>8.1f}±{en[1]:<3.1f}{eat[0]:>6.1f}±{eat[1]:<3.1f}"
                  f"{hunt[0]:>5.1f}±{hunt[1]:<2.1f}{bite[0]:>6.1f}±{bite[1]:<3.1f}"
                  f"{ent[0]:>9.3f}±{ent[1]:<4.3f}{dw[0] * 100:>8.2f}±{dw[1] * 100:<3.2f}"
                  f"{dec[0]:>7.3f}±{dec[1]:<4.3f}")


def _write_out(path, lines):
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()