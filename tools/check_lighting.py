"""Regression test: the map must never blow out and must fade smoothly at night.

Usage:
    uv run python tools/check_lighting.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402

from src.main import Simulation, load_config  # noqa: E402
from src.render import Renderer  # noqa: E402


def main() -> int:
    cfg, _ = load_config()
    sim = Simulation(cfg, seed=1)
    r = Renderer(cfg, sim)
    tpd = int(cfg["time"]["ticks_per_day"])
    rows = []
    worst = 0.0
    for tick in range(0, tpd + 1, 5):
        while sim.tick_count < tick:
            sim.tick()
        r.draw(sim)
        arr = pygame.surfarray.array3d(r.screen).transpose(1, 0, 2)[52:702, 14:664]
        mean = arr.reshape(-1, 3).mean(axis=0)
        white = float(((arr > 235).all(axis=2)).mean())
        worst = max(worst, white)
        rows.append((tick, float(sim.world.light_level(tick)), float(mean.mean()), white))
    print(f"{'tick':>5} {'light':>6} {'map mean':>9} {'near-white':>10}")
    for tick, light, mean, white in rows[::4]:
        print(f"{tick:5d} {light:6.2f} {mean:9.1f} {white * 100:9.3f}%")
    pygame.image.save(r.screen, str(ROOT / "logs" / "lighting_day.png"))
    while sim.tick_count < tpd * 2:
        sim.tick()
    r.draw(sim)
    pygame.image.save(r.screen, str(ROOT / "logs" / "lighting_night.png"))
    arr = pygame.surfarray.array3d(r.screen).transpose(1, 0, 2)[52:702, 14:664]
    night_white = float(((arr > 235).all(axis=2)).mean())
    arr = pygame.surfarray.array3d(r.screen).transpose(1, 0, 2)[52:702, 14:664]
    night_mean = float(arr.reshape(-1, 3).mean())
    print(f"\n白天/夜晚截图: logs/lighting_day.png, logs/lighting_night.png")
    print(f"全天最大过曝像素比例 {worst * 100:.3f}% | 午夜地图均值 {night_mean:.1f} 过曝 {night_white * 100:.3f}%")
    sim.close()
    if worst > 0.002 or night_white > 0.002:
        print("FAILED: map is blowing out")
        return 1
    print("LIGHTING CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())