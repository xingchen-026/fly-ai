"""Natural event system: global weather, wildfires, plague, bloom, migrations.

Weather is a global state machine weighted by season; local events (fire, plague,
bloom, prey/predator arrivals) are spawned in the neighbourhood of the agent,
which keeps an infinite world cheap to simulate.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from src.agent import SPECIES
from src.world import BERRY_DENSITY, FLAMMABLE, DEEP_WATER, WEATHER_CN


class EventSystem:
    def __init__(self, cfg: dict, world, rng: np.random.Generator, tick0: int = 0):
        self.cfg = cfg["events"]
        self.world = world
        self.rng = rng
        lo, hi = self.cfg["start_clear_ticks"]
        self.weather = {"name": "clear", "intensity": float(rng.uniform(0.2, 0.6)),
                        "end": tick0 + int(rng.integers(lo, hi))}
        self.last_weather_cn = "晴"

    # --- weather ------------------------------------------------------------
    def _pick_weather(self, tick: int) -> str:
        cal = self.world.calendar(tick)
        probs = self.cfg["season_weather_probs"][cal["season"]]
        names = list(probs.keys())
        p = np.array([probs[n] for n in names], dtype=np.float64)
        p = p / p.sum()
        return str(self.rng.choice(names, p=p))

    def weather_factors(self) -> dict:
        eff = self.cfg["weather_effects"][self.weather["name"]]
        k = float(self.weather["intensity"])
        out = {}
        for key, v in eff.items():
            out[key] = float(v) * k if key == "temp" else 1.0 + (float(v) - 1.0) * k
        return out

    @property
    def temp_delta(self) -> float:
        return float(self.cfg["weather_effects"][self.weather["name"]]["temp"]) * float(self.weather["intensity"])

    @property
    def weather_cn(self) -> str:
        return WEATHER_CN.get(self.weather["name"], self.weather["name"])

    # --- per-tick update ----------------------------------------------------
    def update(self, tick: int, agent_pos: Tuple[int, int]):
        msgs: List[str] = []
        directives = {}

        if tick >= self.weather["end"]:
            name = self._pick_weather(tick)
            dur = int(self.rng.integers(self.cfg["weather_min_ticks"], self.cfg["weather_max_ticks"]))
            intensity = float(self.rng.uniform(0.35, 1.0))
            self.weather = {"name": name, "intensity": intensity, "end": tick + dur}
            msgs.append(f"天气转为{self.weather_cn}（强度 {intensity:.0%}，持续 {dur} tick）")

        msgs.extend(self.world.update_overlays(tick, self.rng, self.cfg, self.weather["name"]))

        ax, ay = int(agent_pos[0]), int(agent_pos[1])
        wname = self.weather["name"]

        # --- wildfire ignition ---------------------------------------------
        wf = self.cfg["wildfire"]
        if wf["enabled"] and tick % int(wf["check_every"]) == 0:
            chance = float(wf["chance"])
            if wname in ("heatwave", "drought"):
                chance *= float(wf["dry_boost"])
            if wname in ("rain", "storm"):
                chance = 0.0
            if self.rng.random() < chance:
                spot = self.world.scatter(
                    ax, ay, self.rng, wf["spawn_dist"],
                    lambda x, y: bool(FLAMMABLE[self.world.biome_at(x, y)]) and not self.world.on_fire(x, y, tick))
                if spot is not None:
                    life = int(self.rng.integers(wf["burn_ticks"][0], wf["burn_ticks"][1]))
                    if self.world.start_fire(spot[0], spot[1], tick, life):
                        msgs.append(f"附近 ({spot[0]},{spot[1]}) 燃起野火！")

        # --- plague ---------------------------------------------------------
        pl = self.cfg["plague"]
        if pl["enabled"] and tick % int(pl["check_every"]) == 0 and self.rng.random() < float(pl["chance"]):
            spot = self.world.scatter(ax, ay, self.rng, pl["spawn_dist"],
                                      lambda x, y: self.world.biome_at(x, y) != DEEP_WATER)
            if spot is not None:
                self.world.start_plague(spot[0], spot[1], tick, pl)
                msgs.append(f"附近 ({spot[0]},{spot[1]}) 爆发疫病，植物凋零")

        # --- berry bloom ----------------------------------------------------
        bl = self.cfg["bloom"]
        if bl["enabled"] and tick % int(bl["check_every"]) == 0 and self.rng.random() < float(bl["chance"]):
            spot = self.world.scatter(ax, ay, self.rng, bl["spawn_dist"],
                                      lambda x, y: BERRY_DENSITY[self.world.biome_at(x, y)] > 0.0)
            if spot is not None:
                n = self.world.start_bloom(spot[0], spot[1], tick, bl)
                if n > 0:
                    msgs.append(f"附近 ({spot[0]},{spot[1]}) 果实爆发，浆果丛生（{n} 格）")

        # --- animal movements (seasonally modulated) ------------------------
        season = self.world.calendar(tick)["season"]
        season_cfg = self.cfg.get("season_modulation", {}).get(season, {})
        mig_mult = float(season_cfg.get("migration", 1.0))
        inc_mult = float(season_cfg.get("incursion", 1.0))
        mg = self.cfg["migration"]
        food_species = [k for k, s in SPECIES.items() if s["type"] == "prey"]
        if mg["enabled"] and tick % int(mg["check_every"]) == 0 \
                and self.rng.random() < float(mg["chance"]) * mig_mult:
            n = int(self.rng.integers(mg["count"][0], mg["count"][1] + 1))
            key = str(self.rng.choice(food_species))
            directives.setdefault("spawns", []).append((key, n))
            directives["center"] = (ax, ay)
            directives["dist"] = mg["spawn_dist"]
            msgs.append(f"兽群迁徙：{n} 只{SPECIES[key]['cn']}进入附近区域")

        inc = self.cfg["incursion"]
        danger_species = [k for k, s in SPECIES.items() if s["type"] in ("predator", "ambusher")]
        if inc["enabled"] and tick % int(inc["check_every"]) == 0 \
                and self.rng.random() < float(inc["chance"]) * inc_mult:
            n = int(self.rng.integers(inc["count"][0], inc["count"][1] + 1))
            key = str(self.rng.choice(danger_species))
            directives.setdefault("spawns", []).append((key, n))
            directives["center"] = (ax, ay)
            directives["dist"] = inc["spawn_dist"]
            msgs.append(f"危险！{n} 只天敌（{SPECIES[key]['cn']}）闯入附近")

        return msgs, directives