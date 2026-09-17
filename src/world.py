"""Infinite noise-generated 2D world: biomes, resources, structures, climate.

The map itself is never stored: terrain, resource placement and berry blooms are
deterministic functions of (x, y, seed). Only transient state (harvested tiles,
fire, ash, plague, bloom, campfires, shelters) lives in sparse dicts, so the
world is effectively infinite while memory stays local to the visited area.

Resources (per tile, at most one):  berry / mushroom / poison berry / wood / stone.
Structures: campfire (warmth + scares wildlife) and shelter (insulation + rest).
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.util import fbm2, hash01, lerp

# --- biome codes -----------------------------------------------------------
DEEP_WATER, WATER, SAND, GRASS, FOREST, ROCK, SWAMP = range(7)
BIOME_NAMES = ["deep_water", "water", "sand", "grass", "forest", "rock", "swamp"]
BIOME_NAMES_CN = ["深水", "浅水", "沙地", "草地", "森林", "岩石", "沼泽"]
BIOME_INDEX = {name: i for i, name in enumerate(BIOME_NAMES)}

WALK_COST = np.array([np.inf, 3.0, 1.15, 1.0, 1.35, 1.9, 2.1], dtype=np.float32)
FLAMMABLE = np.array([0, 0, 0, 1, 1, 0, 1], dtype=bool)
BERRY_DENSITY = np.array([0.0, 0.0, 0.004, 0.10, 0.17, 0.0, 0.05], dtype=np.float32)
INSULATION = np.array([0.4, 0.6, 0.8, 1.0, 1.55, 1.9, 1.15], dtype=np.float32)

# --- resources -------------------------------------------------------------
RES_NONE, RES_BERRY, RES_MUSHROOM, RES_POISON, RES_WOOD, RES_STONE = range(6)
RESOURCE_CN = {RES_NONE: "", RES_BERRY: "浆果", RES_MUSHROOM: "蘑菇",
               RES_POISON: "毒浆果", RES_WOOD: "木材", RES_STONE: "石料"}
# code -> (allowed biomes, density, regrow ticks, hash seed offset)
RESOURCE_TABLE = {
    RES_WOOD: (("forest", "swamp"), 0.095, 2200, 5101),
    RES_STONE: (("rock", "sand"), 0.35, 4000, 5202),
    RES_MUSHROOM: (("forest", "swamp"), 0.06, 1500, 5303),
    RES_POISON: (("forest", "swamp"), 0.05, 1300, 5404),
    RES_BERRY: (("grass", "forest", "swamp", "sand"), 0.0, 900, 3003),
}
RESOURCE_ORDER = (RES_WOOD, RES_STONE, RES_MUSHROOM, RES_POISON, RES_BERRY)
FOOD_RESOURCES = (RES_BERRY, RES_MUSHROOM)

STRUCTURE_CN = {"campfire": "营火", "shelter": "庇护所"}

_NOISE_SEEDS = {"elevation": 1001, "moisture": 2002, "berry": 3003}
_SEASON_KEYS = ["spring", "summer", "autumn", "winter"]
_SEASON_CN = {"spring": "春", "summer": "夏", "autumn": "秋", "winter": "冬"}
WEATHER_CN = {"clear": "晴", "rain": "降雨", "storm": "暴风雨", "drought": "干旱",
              "cold_snap": "寒潮", "heatwave": "热浪", "fog": "浓雾"}


class World:
    def __init__(self, cfg: dict, seed: int):
        self.cfg = cfg
        self.bp = cfg["world"]
        self.tp = cfg["time"]
        self.seed = int(seed)
        self.chunk_size = int(self.bp.get("chunk", 32))
        self.cache_max = int(self.bp.get("cache_chunks", 768))
        self._cache: "OrderedDict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]" = OrderedDict()

        self.depleted: Dict[Tuple[int, int], int] = {}
        self.fire: Dict[Tuple[int, int], int] = {}
        self.ash: Dict[Tuple[int, int], int] = {}
        self.bloom: Dict[Tuple[int, int], int] = {}
        self.plague: List[dict] = []
        self.structures: Dict[Tuple[int, int], dict] = {}
        self._last_sweep = -10 ** 9
        self._rng = np.random.default_rng(self.seed + 555)

    # --- generation --------------------------------------------------------
    def _noise(self, x, y, key: str, scale: float) -> np.ndarray:
        p = self.bp[key]
        return fbm2(x, y, seed=self.seed + _NOISE_SEEDS[key], scale=scale,
                    octaves=int(p["octaves"]), gain=float(p["gain"]),
                    lacunarity=float(p["lacunarity"]))

    def biome_codes(self, xs, ys) -> np.ndarray:
        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)
        e = self._noise(xs, ys, "elevation", float(self.bp["elevation"]["scale"]))
        m = self._noise(xs, ys, "moisture", float(self.bp["moisture"]["scale"]))
        code = np.full(e.shape, GRASS, dtype=np.uint8)
        code = np.where(m < 0.30, SAND, code)
        code = np.where(m > 0.52, FOREST, code)
        code = np.where(m > 0.72, SWAMP, code)
        code = np.where(e > float(self.bp["mountain_level"]), ROCK, code)
        code = np.where(e < float(self.bp["shallow_level"]), WATER, code)
        code = np.where(e < float(self.bp["sea_level"]), DEEP_WATER, code)
        return code.astype(np.uint8)

    def _resource_codes(self, codes: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        codes = np.asarray(codes).reshape(-1)
        gx = np.asarray(gx).reshape(-1)
        gy = np.asarray(gy).reshape(-1)
        res = np.zeros(codes.shape, dtype=np.uint8)
        for rtype in RESOURCE_ORDER:
            biomes, density, _regrow, seed_off = RESOURCE_TABLE[rtype]
            allowed = np.isin(codes, [BIOME_INDEX[b] for b in biomes])
            if rtype == RES_BERRY:
                dens = BERRY_DENSITY[codes].astype(np.float64)
            elif rtype == RES_STONE:
                dens = np.where(codes == ROCK, 0.5, np.where(codes == SAND, 0.03, 0.0))
            else:
                dens = np.where(allowed, float(density), 0.0)
            rnd = hash01(gx, gy, self.seed + seed_off)
            pick = (res == RES_NONE) & allowed & (rnd < dens)
            res = np.where(pick, rtype, res)
        return res.astype(np.uint8)

    def _gen_chunk(self, cx: int, cy: int):
        cs = self.chunk_size
        gx, gy = np.meshgrid(cx * cs + np.arange(cs), cy * cs + np.arange(cs))
        codes = self.biome_codes(gx.ravel(), gy.ravel()).reshape(cs, cs)
        res = self._resource_codes(codes, gx.ravel(), gy.ravel()).reshape(cs, cs)
        return codes.astype(np.uint8), res.astype(np.uint8)

    def chunk(self, cx: int, cy: int):
        key = (int(cx), int(cy))
        c = self._cache.get(key)
        if c is not None:
            self._cache.move_to_end(key)
            return c
        c = self._gen_chunk(cx, cy)
        self._cache[key] = c
        if len(self._cache) > self.cache_max:
            self._cache.popitem(last=False)
        return c

    def biome_at(self, x: int, y: int) -> int:
        cs = self.chunk_size
        codes, _ = self.chunk(x // cs, y // cs)
        return int(codes[y % cs, x % cs])

    def resource_base_at(self, x: int, y: int) -> int:
        cs = self.chunk_size
        _, res = self.chunk(x // cs, y // cs)
        return int(res[y % cs, x % cs])

    def window_arrays(self, cx: int, cy: int, r: int, tick: int) -> dict:
        """Bulk sampling for rendering: biomes, resources and transient masks."""
        n = 2 * r + 1
        x0, y0 = cx - r, cy - r
        codes = np.empty((n, n), dtype=np.uint8)
        res = np.empty((n, n), dtype=np.uint8)
        cs = self.chunk_size
        for j in range(n):
            y = y0 + j
            cyc, yy = y // cs, y % cs
            i = 0
            while i < n:
                x = x0 + i
                cxc, xx = x // cs, x % cs
                take = min(cs - xx, n - i)
                c, rr = self.chunk(cxc, cyc)
                codes[j, i:i + take] = c[yy, xx:xx + take]
                res[j, i:i + take] = rr[yy, xx:xx + take]
                i += take

        fire = np.zeros((n, n), dtype=bool)
        ash = np.zeros((n, n), dtype=bool)
        bloom = np.zeros((n, n), dtype=bool)
        dealt = np.zeros((n, n), dtype=bool)
        plague = np.zeros((n, n), dtype=bool)
        campfire = np.zeros((n, n), dtype=bool)
        shelter = np.zeros((n, n), dtype=bool)

        for d, mask in ((self.fire, fire), (self.ash, ash), (self.bloom, bloom)):
            for (x, y), end in d.items():
                if end > tick and x0 <= x < x0 + n and y0 <= y < y0 + n:
                    mask[y - y0, x - x0] = True
        for (x, y), end in self.depleted.items():
            if end > tick and x0 <= x < x0 + n and y0 <= y < y0 + n:
                dealt[y - y0, x - x0] = True
        for (x, y), st in self.structures.items():
            if st["end"] > tick and x0 <= x < x0 + n and y0 <= y < y0 + n:
                if st["kind"] == "campfire":
                    campfire[y - y0, x - x0] = True
                else:
                    shelter[y - y0, x - x0] = True
        for p in self.plague:
            if p["end"] <= tick:
                continue
            ax0, ax1 = max(p["cx"] - p["radius"], x0), min(p["cx"] + p["radius"], x0 + n - 1)
            ay0, ay1 = max(p["cy"] - p["radius"], y0), min(p["cy"] + p["radius"], y0 + n - 1)
            if ax0 <= ax1 and ay0 <= ay1:
                plague[ay0 - y0:ay1 - y0 + 1, ax0 - x0:ax1 - x0 + 1] = True

        # bloom turns empty grass/forest/swamp tiles into extra berries
        empty = (res == RES_NONE) & bloom
        if empty.any():
            res = np.where(empty, RES_BERRY, res)
        res = np.where(dealt | fire | ash | plague, RES_NONE, res).astype(np.uint8)
        return {"codes": codes, "res": res, "berry": (res == RES_BERRY),
                "fire": fire, "ash": ash, "bloom": bloom, "plague": plague,
                "campfire": campfire, "shelter": shelter,
                "x0": x0, "y0": y0, "r": r}

    # --- resource queries ---------------------------------------------------
    def resource_at(self, x: int, y: int, tick: int = 0) -> int:
        """Currently harvestable resource on a tile (0 = none)."""
        if self.on_fire(x, y, tick) or self.is_ash(x, y, tick) or self.in_plague(x, y, tick):
            return RES_NONE
        ready = self.depleted.get((x, y))
        if ready is not None and tick < ready:
            return RES_NONE
        code = self.resource_base_at(x, y)
        if code == RES_NONE and self.in_bloom(x, y, tick):
            if int(self.biome_at(x, y)) in (GRASS, FOREST, SWAMP):
                return RES_BERRY
        return code

    def take_resource(self, x: int, y: int, tick: int) -> int:
        code = self.resource_at(x, y, tick)
        if code == RES_NONE:
            return RES_NONE
        regrow = RESOURCE_TABLE[code][2]
        self.depleted[(x, y)] = tick + int(regrow)
        return code

    def has_berry(self, x: int, y: int, tick: int = 0) -> bool:
        return self.resource_at(x, y, tick) == RES_BERRY

    def take_berry(self, x: int, y: int, tick: int) -> bool:
        return self.take_resource(x, y, tick) == RES_BERRY

    def is_water(self, x: int, y: int) -> bool:
        return int(self.biome_at(x, y)) in (WATER, DEEP_WATER)

    def is_blocked(self, x: int, y: int) -> bool:
        return int(self.biome_at(x, y)) == DEEP_WATER

    def can_drink(self, x: int, y: int) -> bool:
        for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            if int(self.biome_at(x + dx, y + dy)) in (WATER, SWAMP):
                return True
        return False

    def is_ash(self, x: int, y: int, tick: int = 0) -> bool:
        return self.ash.get((x, y), -1) > tick

    def in_bloom(self, x: int, y: int, tick: int = 0) -> bool:
        return self.bloom.get((x, y), -1) > tick

    def on_fire(self, x: int, y: int, tick: int = 0) -> bool:
        return self.fire.get((x, y), -1) > tick

    def in_plague(self, x: int, y: int, tick: int = 0) -> bool:
        for p in self.plague:
            if p["end"] > tick and abs(x - p["cx"]) <= p["radius"] and abs(y - p["cy"]) <= p["radius"]:
                return True
        return False

    # --- structures ---------------------------------------------------------
    def add_structure(self, x: int, y: int, kind: str, tick: int, life: int) -> bool:
        if (x, y) in self.structures and self.structures[(x, y)]["end"] > tick:
            return False
        self.structures[(x, y)] = {"kind": kind, "end": tick + int(life)}
        return True

    def structure_at(self, x: int, y: int, tick: int = 0) -> Optional[dict]:
        st = self.structures.get((x, y))
        if st is None or st["end"] <= tick:
            return None
        return st

    def refuel_campfire(self, x: int, y: int, tick: int, life: int) -> bool:
        st = self.structure_at(x, y, tick)
        if st is None or st["kind"] != "campfire":
            return False
        st["end"] = max(st["end"], tick) + int(life)
        return True

    def campfire_near(self, x: int, y: int, tick: int = 0, r: int = 2) -> bool:
        for j in range(-r, r + 1):
            for i in range(-r, r + 1):
                st = self.structures.get((x + i, y + j))
                if st is not None and st["kind"] == "campfire" and st["end"] > tick:
                    return True
        return False

    # --- overlays / events --------------------------------------------------
    def start_fire(self, x: int, y: int, tick: int, life: int) -> bool:
        if not FLAMMABLE[self.biome_at(x, y)] or self.on_fire(x, y, tick) or self.is_ash(x, y, tick):
            return False
        self.fire[(x, y)] = tick + int(life)
        return True

    def start_plague(self, cx: int, cy: int, tick: int, cfg: dict) -> None:
        self.plague.append({"cx": int(cx), "cy": int(cy), "radius": int(cfg["radius"]),
                            "end": tick + int(self._rng.integers(cfg["life_ticks"][0], cfg["life_ticks"][1]))})

    def start_bloom(self, cx: int, cy: int, tick: int, cfg: dict) -> int:
        r = int(cfg["radius"])
        life = int(self._rng.integers(cfg["life_ticks"][0], cfg["life_ticks"][1]))
        n = 0
        for j in range(-r, r + 1):
            for i in range(-r, r + 1):
                if abs(i) + abs(j) > r + 2:
                    continue
                x, y = cx + i, cy + j
                code = int(self.biome_at(x, y))
                if code in (GRASS, FOREST, SWAMP) and not self.on_fire(x, y, tick):
                    self.bloom[(x, y)] = tick + life
                    n += 1
        return n

    def scatter(self, cx: int, cy: int, rng, dist_range, predicate, tries: int = 40) -> Optional[Tuple[int, int]]:
        lo, hi = int(dist_range[0]), int(dist_range[1])
        for _ in range(tries):
            ang = float(rng.random()) * 2.0 * np.pi
            d = lo + float(rng.random()) * (hi - lo)
            x = int(round(cx + np.cos(ang) * d))
            y = int(round(cy + np.sin(ang) * d))
            if predicate(x, y):
                return x, y
        return None

    def update_overlays(self, tick: int, rng, events_cfg: dict, weather_name: str) -> List[str]:
        msgs: List[str] = []
        wf = events_cfg["wildfire"]

        if tick - self._last_sweep >= int(self.bp.get("sweep_every", 240)):
            self._last_sweep = tick
            for d in (self.depleted, self.ash, self.bloom):
                for k in [k for k, v in d.items() if v <= tick]:
                    del d[k]
            self.plague = [p for p in self.plague if p["end"] > tick]
            gone = [k for k, st in self.structures.items() if st["end"] <= tick]
            for k in gone:
                kind = self.structures[k]["kind"]
                del self.structures[k]
                msgs.append(f"{STRUCTURE_CN.get(kind, kind)}已失效（({k[0]},{k[1]})）")

        if not self.fire or tick % int(wf["spread_every"]) != 0:
            return msgs

        burnt = [k for k, v in self.fire.items() if v <= tick]
        for k in burnt:
            del self.fire[k]
            self.ash[k] = tick + int(self.bp["ash_ticks"])
        if burnt and tick % 240 < int(wf["spread_every"]):
            msgs.append(f"{len(burnt)} 处野火熄灭，地面留下灰烬")

        dry = weather_name in ("clear", "heatwave", "drought")
        chance = float(wf["spread_chance"]) * (1.0 if dry else 0.35)
        if weather_name in ("rain", "storm"):
            chance = 0.0
        if chance <= 0.0 or len(self.fire) >= int(wf["max_fire_tiles"]):
            return msgs

        lo, hi = int(wf["burn_ticks"][0]), int(wf["burn_ticks"][1])
        spread = 0
        for (x, y) in list(self.fire.keys()):
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if len(self.fire) >= int(wf["max_fire_tiles"]):
                    break
                if (nx, ny) in self.fire:
                    continue
                if rng.random() < chance:
                    life = int(rng.integers(lo, hi))
                    if self.start_fire(nx, ny, tick, life):
                        spread += 1
        if spread and tick % 120 < int(wf["spread_every"]):
            msgs.append(f"火势蔓延，已有 {len(self.fire)} 格燃烧")
        return msgs

    # --- climate ------------------------------------------------------------
    def day(self, tick: int) -> int:
        return int(tick // int(self.tp["ticks_per_day"]))

    def season_index(self, tick: int) -> int:
        dps = int(self.tp["days_per_season"])
        return int((self.day(tick) // dps) % 4)

    def calendar(self, tick: int) -> dict:
        day = self.day(tick)
        dps = int(self.tp["days_per_season"])
        idx = self.season_index(tick)
        return {"day": day + 1, "day_of_season": day % dps + 1, "season_idx": idx,
                "season": _SEASON_KEYS[idx], "season_cn": _SEASON_CN[_SEASON_KEYS[idx]],
                "year": day // (4 * dps) + 1, "tick_of_day": tick % int(self.tp["ticks_per_day"])}

    def season_temp_offset(self, tick: int) -> float:
        offsets = self.bp["season_temp_offsets"]
        tpd = int(self.tp["ticks_per_day"])
        yt = 4 * int(self.tp["days_per_season"]) * tpd
        idx = (tick % yt) / yt * 4.0
        i0 = int(np.floor(idx)) % 4
        i1 = (i0 + 1) % 4
        w = 0.5 - 0.5 * np.cos(np.pi * (idx - np.floor(idx)))
        return float(lerp(offsets[i0], offsets[i1], w))

    def light_level(self, tick: int) -> float:
        phase = (tick % int(self.tp["ticks_per_day"])) / float(self.tp["ticks_per_day"])
        return float(np.clip(0.5 - 0.5 * np.cos(2.0 * np.pi * phase), 0.0, 1.0))

    def diurnal_offset(self, tick: int) -> float:
        phase = (tick % int(self.tp["ticks_per_day"])) / float(self.tp["ticks_per_day"])
        return float(-float(self.bp["diurnal_amp"]) * np.cos(2.0 * np.pi * phase))

    def temperature(self, x: int, y: int, tick: int, weather_temp: float = 0.0) -> float:
        code = int(self.biome_at(x, y))
        local = 0.0
        if code in (WATER, DEEP_WATER):
            local = float(self.bp["water_temp_offset"])
        elif code == FOREST:
            local = float(self.bp["forest_temp_offset"])
        elif code == ROCK:
            local = float(self.bp["rock_temp_offset"])
        if self._fire_near(x, y, tick):
            local += 6.0
        if self.campfire_near(x, y, tick, r=2):
            local += float(self.bp.get("campfire_warmth", 9.0))
        return float(self.bp["ambient_base"] + self.season_temp_offset(tick)
                     + self.diurnal_offset(tick) + local + weather_temp)

    def _fire_near(self, x: int, y: int, tick: int) -> bool:
        r = int(self.bp.get("fire_warm_radius", 1))
        for j in range(-r, r + 1):
            for i in range(-r, r + 1):
                if self.on_fire(x + i, y + j, tick):
                    return True
        return False

    # --- spawn --------------------------------------------------------------
    def find_spawn(self, rng, cx: int = 0, cy: int = 0, max_r: int = 80) -> Tuple[int, int]:
        for r in range(4, max_r, 3):
            for _ in range(24):
                x = int(cx + rng.integers(-r, r + 1))
                y = int(cy + rng.integers(-r, r + 1))
                code = int(self.biome_at(x, y))
                if code != DEEP_WATER and WALK_COST[code] < 2.6:
                    return x, y
        return int(cx), int(cy)