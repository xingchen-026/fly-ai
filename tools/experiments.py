"""Parallel grid experiments: run `run_headless` over config grids x seeds.

Examples:
    # learning rate sweep, 3 seeds, 6000 ticks, 4 worker processes
    uv run python tools/experiments.py --set neural.plasticity.neuromod.lr=0,0.25,0.5 \
        --seeds 1,2,3 --ticks 6000 --jobs 4

    # two-dimensional grid + sort by energy, custom output dir
    uv run python tools/experiments.py --set decision.network_gain=0.3,0.65 \
        --set decision.instinct_gain=0.5,1.0 --seeds 1,2 --ticks 4000 \
        --sort energy --out logs/exp_netgain

Outputs (under --out, default logs/experiments_<timestamp>):
    runs.csv      one row per (config combo, seed) with final metrics
    curves.csv    snapshot metrics over ticks for every run
    report.html   self-contained summary table + SVG learning curves (no deps)
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from copy import deepcopy
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.main import Simulation, apply_overrides, load_config  # noqa: E402

FINAL_KEYS = ["deaths", "hp", "energy", "thirst", "temp", "distance", "hz", "gain",
              "w_change", "readout_drift", "da_ema", "entropy",
              "eat", "drink", "hunt", "bite", "poison", "fire"]
CURVE_KEYS = ["hp", "energy", "thirst", "mean_dn_hz", "da", "da_ema", "w_change",
              "deaths", "n_eat", "n_bite"]


def run_one(task):
    """Worker: build cfg, run a headless simulation, return final + curve metrics."""
    cfg, seed, ticks, snap_every, label = task
    sim = Simulation(cfg, seed=seed, snapshot_every=snap_every)
    t0 = time.perf_counter()
    for _ in range(ticks):
        sim.tick()
    elapsed = time.perf_counter() - t0
    s = sim.stats()
    a = s["agent"]
    learn = s["learning"]
    acts = np.array(s["action_counts"], dtype=float)
    p = acts / max(acts.sum(), 1.0)
    ent = float(-(p * np.log(p + 1e-9)).sum())
    row = {"label": label, "seed": seed, "ticks": ticks,
           "deaths": s["deaths"], "hp": a["hp"], "energy": a["energy"],
           "thirst": a["thirst"], "temp": a["temp"], "distance": a["distance"],
           "hz": s["net_hz"], "gain": s["gain"], "w_change": learn["w_change"],
           "readout_drift": learn["readout_drift"], "da_ema": learn["da_ema"],
           "entropy": ent, "ticks_per_s": ticks / max(elapsed, 1e-9)}
    for k in ("eat", "drink", "hunt", "bite", "poison", "fire"):
        row[k] = s["counters"][k]
    curve = [{"label": label, "seed": seed, "tick": snap.get("tick", 0),
              **{k: snap.get(k, 0.0) for k in CURVE_KEYS}} for snap in sim.snapshots]
    sim.close()
    return row, curve


def parse_grid(items):
    names, value_lists = [], []
    for item in items or []:
        key, raw = item.split("=", 1)
        vals = [v for v in raw.split(",") if v != ""]
        names.append(key)
        value_lists.append(vals)
    combos = list(product(*value_lists)) if names else [()]
    out = []
    for combo in combos:
        overrides = [f"{n}={v}" for n, v in zip(names, combo)]
        label = " | ".join(overrides) if overrides else "default"
        out.append((label, overrides))
    return out


def svg_lines(series, width=860, height=220, title="", y_label="", colors=None):
    """Tiny dependency-free SVG line chart."""
    colors = colors or ["#e05c5c", "#f0b348", "#52b0ec", "#b080e0", "#8cc88c",
                        "#ff78d2", "#78d2c8", "#c8c878"]
    pts_all = [v for _, xs in series for v in xs]
    if not pts_all:
        return ""
    lo, hi = min(pts_all), max(pts_all)
    if hi - lo < 1e-9:
        hi = lo + 1.0
    n = max(len(xs) for _, xs in series)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'style="background:#12141a;border:1px solid #2a2e3a">',
             f'<text x="10" y="18" fill="#c8cede" font-size="13">{title}</text>',
             f'<text x="10" y="{height - 6}" fill="#78809a" font-size="11">{y_label} '
             f'[{lo:.2f} .. {hi:.2f}]</text>']
    for i, (name, xs) in enumerate(series):
        if len(xs) < 2:
            continue
        pts = []
        for j, v in enumerate(xs):
            x = 40 + (width - 60) * (j / (n - 1))
            y = 30 + (height - 60) * (1 - (v - lo) / (hi - lo))
            pts.append(f"{x:.1f},{y:.1f}")
        col = colors[i % len(colors)]
        parts.append(f'<polyline fill="none" stroke="{col}" stroke-width="1.8" points="{" ".join(pts)}"/>')
        parts.append(f'<text x="{width - 260}" y="{34 + 14 * i}" fill="{col}" font-size="11">{name}</text>')
    parts.append("</svg>")
    return "".join(parts)


def write_report(out_dir: Path, rows, curves, labels, args):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "runs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / "curves.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(curves[0].keys()))
        w.writeheader()
        w.writerows(curves)

    def agg(label, key):
        v = np.array([r[key] for r in rows if r["label"] == label], dtype=float)
        return float(v.mean()), float(v.std()), v.size

    html = [f"<html><head><meta charset='utf-8'><title>fly-AI experiments</title>"
            f"<style>body{{background:#0e1016;color:#c8cede;font-family:Segoe UI,Microsoft YaHei,sans-serif;"
            f"padding:18px}}table{{border-collapse:collapse;font-size:13px}}"
            f"th,td{{border:1px solid #2a2e3a;padding:4px 8px}}th{{background:#1a1e2a}}</style></head><body>",
            f"<h2>fly-AI 网格实验（ticks={args.ticks} seeds={args.seeds}）</h2>",
            "<table><tr><th>配置</th><th>死亡</th><th>HP</th><th>能量</th><th>口渴</th>"
            "<th>进食</th><th>捕猎</th><th>被咬</th><th>Δw%</th><th>动作熵</th><th>tick/s</th></tr>"]
    for label in labels:
        cells = [f"{agg(label, 'deaths')[0]:.1f}", f"{agg(label, 'hp')[0]:.1f}",
                 f"{agg(label, 'energy')[0]:.1f}", f"{agg(label, 'thirst')[0]:.1f}",
                 f"{agg(label, 'eat')[0]:.1f}", f"{agg(label, 'hunt')[0]:.1f}",
                 f"{agg(label, 'bite')[0]:.1f}", f"{agg(label, 'w_change')[0] * 100:.2f}",
                 f"{agg(label, 'entropy')[0]:.3f}", f"{agg(label, 'ticks_per_s')[0]:.0f}"]
        html.append(f"<tr><td>{label}</td>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    html.append("</table><br/>")

    for key, y_label in (("hp", "平均 HP"), ("energy", "平均能量"), ("n_eat", "累计进食"),
                         ("da_ema", "平均 DA"), ("w_change", "突触改变比例")):
        series = []
        for label in labels:
            runs = [c for c in curves if c["label"] == label]
            by_tick = {}
            for c in runs:
                by_tick.setdefault(c["tick"], []).append(c[key])
            ticks = sorted(by_tick)
            series.append((label.replace(" | ", " "), [float(np.mean(by_tick[t])) for t in ticks]))
        html.append(svg_lines(series, title=f"{y_label} 随 tick 变化", y_label=y_label))
        html.append("<br/>")
    html.append("</body></html>")
    (out_dir / "report.html").write_text("\n".join(html), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--ticks", type=int, default=6000)
    ap.add_argument("--jobs", type=int, default=0, help="0 = cpu_count-1")
    ap.add_argument("--snapshot-every", type=int, default=500)
    ap.add_argument("--sort", default="deaths")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    base, _ = load_config()
    if not any(o.split("=", 1)[0].startswith("population.") for o in (args.overrides or [])):
        base.setdefault("population", {})["size"] = 1  # fast single-agent default
    combos = parse_grid(args.overrides)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else ROOT / "logs" / f"experiments_{stamp}"
    jobs = args.jobs or max(1, (os_cpu := (__import__("os").cpu_count() or 2)) - 1)
    print(f"grid: {len(combos)} 组配置 x {len(seeds)} seeds = {len(combos) * len(seeds)} 次运行"
          f" | ticks={args.ticks} | jobs={jobs}")
    for label, _ in combos:
        print("  -", label)
    tasks = []
    for label, overrides in combos:
        for seed in seeds:
            cfg = deepcopy(base)
            apply_overrides(cfg, overrides)
            tasks.append((cfg, seed, args.ticks, args.snapshot_every, label))
    t0 = time.perf_counter()
    results = None
    if jobs > 1:
        try:
            with Pool(processes=jobs) as pool:
                results = pool.map(run_one, tasks)
        except (PermissionError, OSError) as exc:  # sandboxed shells can forbid pipes
            print(f"[warn] 多进程不可用（{exc}），自动回退为串行执行")
            results = None
    if results is None:
        results = []
        for i, task in enumerate(tasks, 1):
            print(f"  [{i}/{len(tasks)}] {task[4]}  seed={task[1]}", flush=True)
            results.append(run_one(task))
    rows = [r for r, _ in results]
    curves = [c for _, cs in results for c in cs]
    labels = [label for label, _ in combos]
    write_report(out_dir, rows, curves, labels, args)
    dt = time.perf_counter() - t0

    def agg(label, key):
        v = np.array([r[key] for r in rows if r["label"] == label], dtype=float)
        return v.mean(), v.std()

    print(f"\n{'配置':<44}{'死亡':>7}{'HP':>7}{'能量':>8}{'进食':>7}{'被咬':>7}"
          f"{'Δw%':>8}{'tick/s':>8}")
    order = sorted(labels, key=lambda lb: agg(lb, args.sort)[0])
    for label in order:
        cells = []
        for key, fmt in (("deaths", "{:.1f}"), ("hp", "{:.1f}"), ("energy", "{:.1f}"),
                         ("eat", "{:.1f}"), ("bite", "{:.1f}"), ("w_change", "{:.2f}"),
                         ("ticks_per_s", "{:.0f}")):
            m, s = agg(label, key)
            if key == "w_change":
                m, s = m * 100, s * 100
            cells.append(fmt.format(m) + (f"±{fmt.format(s)}" if len(seeds) > 1 else ""))
        print(f"{label[:42]:<44}" + "".join(f"{c:>9}" for c in cells))
    print(f"\n总耗时 {dt:.1f}s | 输出目录 {out_dir}")
    print(f"  runs.csv / curves.csv / report.html")


if __name__ == "__main__":
    main()