"""Headless (SDL dummy) render smoke test: draws frames, saves a screenshot.

Usage:
    uv run python tools/smoke_render.py --frames 240 --speed 4
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from src.main import Simulation, load_config  # noqa: E402
from src.render import Renderer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--speed", type=int, default=4)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--shot", default="logs/render_test.png")
    args = ap.parse_args()

    cfg, _ = load_config()
    sim = Simulation(cfg, seed=args.seed)
    r = Renderer(cfg, sim)
    t0 = time.perf_counter()
    for i in range(args.frames):
        for _ in range(args.speed):
            sim.tick()
        r.draw(sim, paused=(i % 97 == 0), speed=args.speed)
    dt = time.perf_counter() - t0
    shot = ROOT / args.shot
    shot.parent.mkdir(parents=True, exist_ok=True)
    import pygame
    pygame.image.save(r.screen, str(shot))

    arr = pygame.surfarray.array3d(r.screen).transpose(1, 0, 2)
    regions = {"map": arr[52:702, 14:664], "heat": arr[52:270, 690:1266],
               "motor": arr[290:432, 690:1266], "stats": arr[579:704, 690:1266],
               "log": arr[708:786, 14:664], "header": arr[0:42, :]}
    print(f"frames={args.frames} speed={args.speed} ticks={sim.tick_count} "
          f"time={dt:.2f}s ({args.frames / dt:.1f} fps draw-only)")
    for k, v in regions.items():
        print(f"  {k:6s} mean RGB {v.reshape(-1, 3).mean(axis=0).round(1)}  std {v.std():.1f}")
    st = sim.stats()
    print("agent:", {k: (round(v, 1) if isinstance(v, (int, float)) else v)
                     for k, v in st['agent'].items() if k not in ('urge',)})
    print("net hz %.2f gain %.2f  fire %d  creatures %d" % (st['net_hz'], st['gain'], st['fire_tiles'], st['creatures']))
    print("screenshot:", shot)
    sim.close()


if __name__ == "__main__":
    main()