"""Head-less simulation smoke test: runs, validates numeric sanity, prints stats.

Usage:
    uv run python tools/smoke_test.py --ticks 2000 --seed 7 [--log logs/smoke.csv]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import SPECIES  # noqa: E402
from src.main import Simulation, load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    cfg, _ = load_config()
    log_path = Path(args.log) if args.log else None
    sim = Simulation(cfg, seed=args.seed, log_path=log_path)
    failures = []
    for _ in range(args.ticks):
        sim.tick()
        st = sim.agent.stats()
        for key in ("hp", "energy", "thirst", "fatigue"):
            v = float(st[key])
            if not np.isfinite(v) or not (0.0 <= v <= 100.0 + 1e-6):
                failures.append(f"tick {sim.tick_count}: {key}={v} out of range")
        if not (np.isfinite(st["temp"]) and -40.0 <= st["temp"] <= 70.0):
            failures.append(f"tick {sim.tick_count}: temp={st['temp']} out of range")
        if not np.isfinite(sim.net.mean_dn_hz()):
            failures.append(f"tick {sim.tick_count}: network rate NaN")
        if not (0.0 <= float(st["poison"]) <= float(sim.agent.max_poison) + 1e-6):
            failures.append(f"tick {sim.tick_count}: poison={st['poison']} out of range")
        if not (0.0 <= sim.net.gain <= sim.net.gain_max + 1e-6):
            failures.append(f"tick {sim.tick_count}: homeostasis gain={sim.net.gain}")
        if sim.net.stdp_on:
            w = sim.net.W.data
            if not np.all(np.isfinite(w)):
                failures.append(f"tick {sim.tick_count}: non-finite synaptic weight")
            if float(w.min()) < float(sim.net.w_lo.min()) - 1e-3 or \
               float(w.max()) > float(sim.net.w_hi.max()) + 1e-3:
                failures.append(f"tick {sim.tick_count}: weight outside clamp bounds")

    s = sim.stats()
    every = int(cfg["decision"]["decide_every_ticks"])
    base_decisions = (args.ticks + every - 1) // every
    actions = sum(s["action_counts"])
    # every respawn forces one extra first decision for that member
    if not (base_decisions <= actions <= base_decisions + s["deaths"]):
        failures.append(f"decision count mismatch: actions={actions} expected "
                        f"{base_decisions}..{base_decisions + s['deaths']}")
    sim.close()
    for key, n in s["creatures_by_species"].items():
        cap = int(SPECIES[key]["max"] * 1.3) + 4
        if n > cap:
            failures.append(f"species {key} population {n} exceeds cap {cap}")
    if log_path is not None:
        rows = sum(1 for _ in open(log_path, encoding="utf-8")) - 1
        if rows != args.ticks:
            failures.append(f"log rows={rows} expected={args.ticks}")
    a = s["agent"]
    print(f"ticks={args.ticks} seed={args.seed}  存活 tick={a['age']}  死亡 {s['deaths']} 次")
    print(f"HP {a['hp']:.1f} | 能量 {a['energy']:.1f} | 口渴 {a['thirst']:.1f} | "
          f"体温 {a['temp']:.1f}°C | 疲劳 {a['fatigue']:.1f}")
    pop = s.get("population", {})
    print(f"网络 DN {s['net_hz']:.2f} Hz | 自稳态增益 {s['gain']:.2f} | 事件 {s['event_counts']}")
    print(f"种群 {pop.get('size', 1)} 个体 | 存活 {pop.get('alive', 1)} | "
          f"最高世代 {max(pop.get('generations', [0]))} | 死亡 {pop.get('deaths_total', 0)}")
    from src.agent import ACTIONS_CN
    print(f"动作计数 {dict(zip(ACTIONS_CN, s['action_counts']))}")
    if failures:
        print("\nFAILED:")
        for f in failures[:10]:
            print(" -", f)
        return 1
    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())