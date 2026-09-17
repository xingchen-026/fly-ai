"""World / event statistics without the neural network (fast).

Usage:
    uv run python tools/inspect_world.py --ticks 20000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.events import EventSystem  # noqa: E402
from src.main import load_config  # noqa: E402
from src.world import BIOME_NAMES_CN, World  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    cfg, _ = load_config()
    world = World(cfg, args.seed)
    rng = np.random.default_rng(args.seed + 17)
    events = EventSystem(cfg, world, rng)
    sx, sy = world.find_spawn(np.random.default_rng(args.seed))

    print("=== 生物群系占比（1024x1024 采样） ===")
    gx, gy = np.meshgrid(np.arange(-512, 512, 4), np.arange(-512, 512, 4))
    codes = world.biome_codes(gx.ravel(), gy.ravel())
    frac = np.bincount(codes, minlength=7) / codes.size
    for name, f in zip(BIOME_NAMES_CN, frac):
        print(f"  {name:4s} {f * 100:5.1f}%")

    print("\n=== 温度年周期（出生点地表温度） ===")
    tpd = int(cfg["time"]["ticks_per_day"])
    year = 4 * int(cfg["time"]["days_per_season"]) * tpd
    for frac in (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875):
        tick = int(frac * year)
        cal = world.calendar(tick)
        print(f"  第{frac:4.2f}年 {cal['season_cn']}季  {world.temperature(sx, sy, tick):6.1f}°C"
              f"  光照 {world.light_level(tick):.2f}")

    print("\n=== 浆果密度（512x512 采样） ===")
    xs = np.random.default_rng(5).integers(-256, 256, 40000)
    ys = np.random.default_rng(6).integers(-256, 256, 40000)
    hit = sum(1 for i in range(40000) if world.has_berry(int(xs[i]), int(ys[i]), 0))
    print(f"  期望密度约 {hit / 40000 * 100:.2f}% （草地 10% / 森林 17% 基准）")

    counts = {"天气": 0, "野火点燃": 0, "火势蔓延": 0, "野火熄灭": 0,
              "疫病": 0, "果实爆发": 0, "迁徙": 0, "天敌": 0}
    keywords = {"天气": "天气转为", "野火点燃": "燃起野火", "火势蔓延": "火势蔓延",
                "野火熄灭": "野火熄灭", "疫病": "疫病", "果实爆发": "果实爆发",
                "迁徙": "兽群", "天敌": "天敌"}
    t0 = time.perf_counter()
    for tick in range(args.ticks):
        msgs, _ = events.update(tick, (sx, sy))
        for m in msgs:
            for key, kw in keywords.items():
                if kw in m:
                    counts[key] += 1
    dt = time.perf_counter() - t0
    cal = world.calendar(args.ticks)
    print(f"\n=== {args.ticks} tick（约 {cal['day']} 天）事件统计（无智能体，{dt:.2f}s） ===")
    for k, v in counts.items():
        print(f"  {k:5s} {v:6d}")
    print(f"  燃烧中格数 {len(world.fire)} | 灰烬格数 {len(world.ash)} | "
          f"疫病区 {len(world.plague)} | 果实爆发格 {len(world.bloom)}")
    print(f"  区块缓存 {len(world._cache)} / {world.cache_max}")


if __name__ == "__main__":
    main()