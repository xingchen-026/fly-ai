"""Connectome-driven survivor: perception, instincts, actions, metabolism, crafting.

Perception: 203-dim vector = 5x5 vision x 7 channels (water / plant / blocked /
food / danger / poison-food / material) + 14 internal channels (including
inventory and nearby structures) + 14 directional cues (food, water, predator,
prey, shelter, hazard, material).

Actions (8): up, down, left, right, interact (eat/drink/hunt/gather), rest,
make_fire (build or refuel a campfire), build_shelter.

World species:
  prey (fly/aphid)        edible, flee or freeze
  predator (wasp)         chases and stings
  ambusher (spider)       hides, lunges when close
  toxic (caterpillar)     contact poison
  ambient (firefly)       decorative, night only
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.world import (BERRY_DENSITY, DEEP_WATER, FOREST, INSULATION, RES_BERRY,
                       RES_MUSHROOM, RES_NONE, RES_POISON, RES_STONE, RES_WOOD,
                       ROCK, SWAMP, WATER, WALK_COST)

ACTIONS = ["up", "down", "left", "right", "interact", "rest", "make_fire", "build_shelter"]
ACTIONS_CN = ["上移", "下移", "左移", "右移", "觅食/饮水/采集", "休息", "生火/添柴", "搭建庇护所"]
DIR_VEC = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
ACT_INTERACT, ACT_REST, ACT_FIRE, ACT_SHELTER = 4, 5, 6, 7

VISION_R = 2
CUE_R = 4
VIS_CHANNELS = 7
INT_CHANNELS = 14
CUE_CHANNELS = 14  # 7 directions x (dx, dy)
N_FEATURES = (2 * VISION_R + 1) ** 2 * VIS_CHANNELS + INT_CHANNELS + CUE_CHANNELS  # 203

DANGER_TYPES = ("predator", "ambusher", "toxic")
EDIBLE_TYPES = ("prey",)
CUE_KEYS = ("food", "water", "pred", "prey", "shelter", "hazard", "material")

SPECIES: Dict[str, dict] = {
    "fly": {"cn": "果蝇", "type": "prey", "color": (150, 240, 170), "energy": 42.0,
            "move_every": 4, "flee_dist": 2, "catch_chance": 0.75, "cooldown": 0,
            "biomes": ("grass", "forest", "swamp"), "density": 11.0, "max": 28, "min_active": 0,
            "seasonal": {"spring": 1.1, "summer": 1.4, "autumn": 1.1, "winter": 0.2}},
    "aphid": {"cn": "蜜蚜", "type": "prey", "color": (206, 246, 140), "energy": 30.0,
              "move_every": 6, "flee_dist": 2, "catch_chance": 0.82, "cooldown": 0,
              "biomes": ("grass", "forest", "sand"), "density": 6.0, "max": 20, "min_active": 0,
              "seasonal": {"spring": 1.3, "summer": 1.2, "autumn": 0.9, "winter": 0.12}},
    "spider": {"cn": "猎蛛", "type": "ambusher", "color": (170, 100, 232), "energy": 0.0,
               "damage": 8.0, "cooldown": 40, "sense": 3, "ambush_range": 2, "move_every": 4,
               "biomes": ("forest", "rock"), "density": 0.6, "max": 5, "min_active": 0,
               "seasonal": {"spring": 1.0, "summer": 1.2, "autumn": 1.1, "winter": 0.5}},
    "wasp": {"cn": "黄蜂", "type": "predator", "color": (244, 182, 58), "energy": 0.0,
             "damage": 5.0, "cooldown": 24, "sense": 7, "move_every": 4,
             "biomes": ("grass", "forest", "sand"), "density": 0.4, "max": 4, "min_active": 0,
             "seasonal": {"spring": 0.8, "summer": 1.5, "autumn": 1.0, "winter": 0.05}},
    "caterpillar": {"cn": "毒蛾幼虫", "type": "toxic", "color": (214, 120, 224), "energy": 0.0,
                    "damage": 1.2, "poison_ticks": 45.0, "cooldown": 25, "sense": 0,
                    "move_every": 5, "biomes": ("swamp", "grass"), "density": 0.7, "max": 6,
                    "min_active": 1,
                    "seasonal": {"spring": 1.6, "summer": 1.2, "autumn": 0.7, "winter": 0.02}},
    "firefly": {"cn": "萤火虫", "type": "ambient", "color": (232, 255, 150), "energy": 0.0,
                "move_every": 3, "biomes": ("grass", "forest", "swamp"), "density": 2.0,
                "max": 12, "min_active": 0, "night_only": True,
                "seasonal": {"spring": 0.6, "summer": 1.8, "autumn": 0.8, "winter": 0.03}},
}


def _clip01(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else float(v))


def _axis_action(dx: float, dy: float) -> int:
    if abs(dx) >= abs(dy):
        return 3 if dx > 0 else 2
    return 1 if dy > 0 else 0


def _opposite(a: int) -> int:
    return {0: 1, 1: 0, 2: 3, 3: 2}.get(a, a)


class Creature:
    def __init__(self, species: str, x: int, y: int):
        if species not in SPECIES:
            raise KeyError(f"unknown species {species!r}")
        self.species = species
        self.spec = SPECIES[species]
        self.kind = str(self.spec["type"])
        self.x, self.y = int(x), int(y)
        self.cooldown = 0
        self.age = 0

    @property
    def cn(self) -> str:
        return str(self.spec["cn"])

    @property
    def color(self):
        return tuple(self.spec["color"])

    def is_dangerous(self) -> bool:
        return self.kind in DANGER_TYPES

    def is_edible(self) -> bool:
        return self.kind in EDIBLE_TYPES

    def _step_toward(self, world, tx: int, ty: int, rng):
        dx, dy = tx - self.x, ty - self.y
        if abs(dx) >= abs(dy):
            opts = [(1 if dx > 0 else -1, 0), (0, 1 if dy > 0 else -1)]
        else:
            opts = [(0, 1 if dy > 0 else -1), (1 if dx > 0 else -1, 0)]
        if rng.random() < 0.15:
            opts = opts[::-1]
        for ox, oy in opts + [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = self.x + ox, self.y + oy
            if not world.is_blocked(nx, ny):
                self.x, self.y = nx, ny
                return True
        return False

    def _step_random(self, world, rng):
        for _ in range(4):
            ox, oy = [(1, 0), (-1, 0), (0, 1), (0, -1)][int(rng.integers(0, 4))]
            nx, ny = self.x + ox, self.y + oy
            if not world.is_blocked(nx, ny):
                self.x, self.y = nx, ny
                return
        return

    def update(self, agents, cfg: dict, world, rng, tick: int):
        """Move / attack the nearest survivor. Returns (messages, hits, effects).

        hits: list of (agent, damage); effects: dict shared by the successful hit
        (e.g. {"poison": ticks}).
        """
        if not isinstance(agents, (list, tuple)):
            agents = [agents]
        spec = self.spec
        self.age += 1
        msgs: List[str] = []
        hits: List[Tuple[object, float]] = []
        effects: Dict[str, float] = {}
        if self.cooldown > 0:
            self.cooldown -= 1
        if not agents:
            return msgs, hits, effects
        target = min(agents, key=lambda a: max(abs(a.x - self.x), abs(a.y - self.y)))
        dx, dy = target.x - self.x, target.y - self.y
        dist = max(abs(dx), abs(dy))
        me = int(spec.get("move_every", 2))
        # campfires keep wildlife away
        if world.campfire_near(self.x, self.y, tick, r=2):
            if tick % me == 0:
                self._step_toward(world, self.x + dx, self.y + dy, rng)
            return msgs, hits, effects

        if self.kind == "prey":
            if tick % me == 0:
                if dist <= int(spec["flee_dist"]):
                    if rng.random() < 0.7:
                        self._step_toward(world, self.x - dx, self.y - dy, rng)
                elif rng.random() < 0.6:
                    self._step_random(world, rng)
        elif self.kind == "predator":
            if dist <= int(spec["sense"]):
                if tick % me == 0:
                    self._step_toward(world, target.x, target.y, rng)
                if dist <= 1 and self.cooldown == 0:
                    hits.append((target, float(spec["damage"])))
                    self.cooldown = int(spec["cooldown"])
                    msgs.append(f"{self.cn}扑向{_who(target)}，蜇咬了一口！")
        elif self.kind == "ambusher":
            if dist <= int(spec["ambush_range"]):
                if tick % me == 0:
                    self._step_toward(world, target.x, target.y, rng)
                if dist <= 1 and self.cooldown == 0:
                    hits.append((target, float(spec["damage"])))
                    self.cooldown = int(spec["cooldown"])
                    msgs.append(f"{self.cn}从暗处扑出，咬中了{_who(target)}！")
            elif tick % me == 0 and rng.random() < 0.01:
                self._step_random(world, rng)
        elif self.kind == "toxic":
            if dist <= 1 and self.cooldown == 0:
                hits.append((target, float(spec["damage"])))
                effects["poison"] = float(spec["poison_ticks"])
                self.cooldown = int(spec["cooldown"])
                msgs.append(f"{_who(target)}碰到{self.cn}，毒毛刺入皮肤！")
            elif tick % me == 0 and rng.random() < 0.5:
                self._step_random(world, rng)
        else:  # ambient
            if tick % me == 0 and rng.random() < 0.7:
                self._step_random(world, rng)
        return msgs, hits, effects


def _who(agent) -> str:
    name = getattr(agent, "name", None)
    return str(name) if name else "幸存者"


class Agent:
    def __init__(self, cfg: dict, world, rng: np.random.Generator, name: str = "幸存者"):
        self.cfg = cfg["agent"]
        self.dec = cfg["decision"]
        self.cr = cfg["creatures"]
        self.build = self.cfg.get("building", {})
        self.world = world
        self.rng = rng
        self.name = name
        self.max_stat = float(self.cfg["max_stat"])
        self.max_poison = float(self.cfg.get("max_poison_ticks", 60.0))
        self.reward_cfg = self.cfg.get("reward", {})
        self.inv_cap = {k: int(v) for k, v in self.cfg.get("inventory", {}).items()} or {
            "food": 8, "wood": 8, "stone": 6}
        self.traits: dict = {}
        self.x = self.y = 0
        self.action = ACT_REST
        self.poison = 0.0
        self.distance = 0
        self._instinct = np.zeros(len(ACTIONS), dtype=np.float32)
        self._scores = np.zeros(len(ACTIONS), dtype=np.float32)
        self._clear_cues()
        self.reset()

    def _clear_cues(self):
        for k in CUE_KEYS:
            setattr(self, f"_{k}_dir", None)
            setattr(self, f"_{k}_dist", 99)

    # --- life-cycle ---------------------------------------------------------
    def reset(self, x: Optional[int] = None, y: Optional[int] = None):
        s = self.cfg["start"]
        self.x = int(x if x is not None else self.x)
        self.y = int(y if y is not None else self.y)
        self.hp = float(s["hp"])
        self.energy = float(s["energy"])
        self.thirst = float(s["thirst"])
        self.temp = float(s["temp"])
        self.fatigue = float(s["fatigue"])
        self.poison = 0.0
        self.inventory = {"food": 0, "wood": 0, "stone": 0}
        self.alive = True
        self.age = 0
        self.cause = ""
        self.action = ACT_REST
        self._clear_cues()

    # --- perception ---------------------------------------------------------
    def sense(self, creatures, tick: int, vision_factor: float = 1.0) -> np.ndarray:
        world = self.world
        feat = np.zeros(N_FEATURES, dtype=np.float32)
        p = 0
        danger_tiles = {(c.x, c.y) for c in creatures if c.is_dangerous()}
        for dy in range(-VISION_R, VISION_R + 1):
            for dx in range(-VISION_R, VISION_R + 1):
                x, y = self.x + dx, self.y + dy
                code = world.biome_at(x, y)
                res = world.resource_at(x, y, tick)
                danger = 1.0 if (world.on_fire(x, y, tick) or world.in_plague(x, y, tick)) else 0.0
                if danger == 0.0 and (x, y) in danger_tiles:
                    danger = 1.0
                feat[p] = 1.0 if code in (WATER, DEEP_WATER) else 0.0
                feat[p + 1] = 1.0 if BERRY_DENSITY[code] > 0.0 or code in (FOREST, SWAMP) else 0.0
                feat[p + 2] = 1.0 if code == DEEP_WATER else 0.0
                feat[p + 3] = 1.0 if res in (RES_BERRY, RES_MUSHROOM) else 0.0
                feat[p + 4] = danger
                feat[p + 5] = 1.0 if res == RES_POISON else 0.0
                feat[p + 6] = 1.0 if res in (RES_WOOD, RES_STONE) else 0.0
                p += VIS_CHANNELS

        res_here = world.resource_at(self.x, self.y, tick)
        food_here = res_here in (RES_BERRY, RES_MUSHROOM)
        prey_adj = any(c.is_edible() and max(abs(c.x - self.x), abs(c.y - self.y)) <= 1
                       for c in creatures)
        can_drink = world.can_drink(self.x, self.y)
        st = world.structure_at(self.x, self.y, tick)
        shelter_here = 1.0 if (st is not None and st["kind"] == "shelter") else 0.0
        self._campfire_here = bool(st is not None and st["kind"] == "campfire")
        fire_near = 1.0 if world.campfire_near(self.x, self.y, tick, r=2) else 0.0
        feat[p] = _clip01(self.hp / 100.0)
        feat[p + 1] = _clip01(self.energy / 100.0)
        feat[p + 2] = _clip01(self.thirst / 100.0)
        feat[p + 3] = _clip01((self.temp - 5.0) / 30.0)
        feat[p + 4] = _clip01(1.0 - self.fatigue / 100.0)
        self._light = world.light_level(tick)
        feat[p + 5] = self._light
        feat[p + 6] = 1.0 if (food_here or prey_adj) else 0.0
        feat[p + 7] = 1.0 if can_drink else 0.0
        feat[p + 8] = _clip01(1.0 - self.poison / self.max_poison)
        feat[p + 9] = _clip01(self.inventory.get("food", 0) / max(self.inv_cap.get("food", 8), 1))
        feat[p + 10] = _clip01(self.inventory.get("wood", 0) / max(self.inv_cap.get("wood", 8), 1))
        feat[p + 11] = _clip01(self.inventory.get("stone", 0) / max(self.inv_cap.get("stone", 6), 1))
        feat[p + 12] = fire_near
        feat[p + 13] = shelter_here
        p += INT_CHANNELS

        cue_r = max(2, int(round(CUE_R * float(vision_factor))))
        best = {k: (None, 99) for k in CUE_KEYS}
        for dy in range(-cue_r, cue_r + 1):
            for dx in range(-cue_r, cue_r + 1):
                if dx == 0 and dy == 0:
                    continue
                x, y = self.x + dx, self.y + dy
                d = max(abs(dx), abs(dy))
                code = world.biome_at(x, y)
                res = world.resource_at(x, y, tick)
                if d < best["food"][1] and res in (RES_BERRY, RES_MUSHROOM):
                    best["food"] = ((dx, dy), d)
                if d < best["water"][1] and code in (WATER, SWAMP):
                    best["water"] = ((dx, dy), d)
                if d < best["shelter"][1] and code in (FOREST, ROCK):
                    best["shelter"] = ((dx, dy), d)
                if d < best["hazard"][1] and (world.on_fire(x, y, tick) or world.in_plague(x, y, tick)):
                    best["hazard"] = ((dx, dy), d)
                if d < best["material"][1] and res in (RES_WOOD, RES_STONE):
                    best["material"] = ((dx, dy), d)
        creature_r = max(cue_r, int(self.dec["predator_flee_dist"]) + 1)
        for c in creatures:
            dx, dy = c.x - self.x, c.y - self.y
            d = max(abs(dx), abs(dy))
            if d <= creature_r:
                key = "pred" if c.is_dangerous() else ("prey" if c.is_edible() else None)
                if key is not None and d < best[key][1]:
                    best[key] = ((dx, dy), d)
        cues = np.zeros(CUE_CHANNELS, dtype=np.float32)
        for i, key in enumerate(CUE_KEYS):
            pos, d = best[key]
            setattr(self, f"_{key}_dir", pos)
            setattr(self, f"_{key}_dist", d)
            if pos is not None and d > 0:
                cues[2 * i] = pos[0] / d
                cues[2 * i + 1] = pos[1] / d
        feat[p:p + CUE_CHANNELS] = cues

        self._food_here = bool(food_here)
        self._res_here = res_here
        self._prey_here = bool(prey_adj)
        self._can_drink = bool(can_drink)
        self._fire_near = bool(fire_near)
        self._shelter_here = bool(shelter_here)
        self._build_instinct()
        return feat

    # --- decision -----------------------------------------------------------
    def _build_instinct(self) -> None:
        low = float(self.dec["urge_low"])
        need_e = _clip01((low - self.energy) / low)
        need_t = _clip01((low - self.thirst) / low)
        fatigue_hi = _clip01((self.fatigue - 45.0) / 55.0)
        warmth = _clip01((self.temp - 5.0) / 30.0)
        poison_frac = _clip01(self.poison / self.max_poison)
        flee_dist = float(self.dec["predator_flee_dist"])
        fear = _clip01(1.0 - self._pred_dist / flee_dist) if self._pred_dir is not None else 0.0
        wood = int(self.inventory.get("wood", 0))
        stone = int(self.inventory.get("stone", 0))

        s = np.zeros(len(ACTIONS), dtype=np.float32)
        if self._food_here:
            s[ACT_INTERACT] += 2.2 * max(need_e, 0.12)
        if self._can_drink and self.thirst < 98.0:
            s[ACT_INTERACT] += 2.4 * max(need_t, 0.12)
        if self._prey_here and self.energy < 95.0:
            s[ACT_INTERACT] += 1.4 * max(need_e, 0.45)
        if self._res_here == RES_POISON:
            s[ACT_INTERACT] += 1.5 * max(need_e - 0.55, 0.0)  # desperate gamble
        if self._res_here == RES_WOOD and wood < self.inv_cap.get("wood", 8):
            s[ACT_INTERACT] += 0.35
        if self._res_here == RES_STONE and stone < self.inv_cap.get("stone", 6):
            s[ACT_INTERACT] += 0.45

        if need_e > 0.08 and self._food_dir is not None:
            s[_axis_action(*self._food_dir)] += 1.6 * need_e + 0.25
        if need_t > 0.08 and self._water_dir is not None:
            s[_axis_action(*self._water_dir)] += 1.8 * need_t + 0.25
        if need_e > 0.35 and self._prey_dir is not None and self._prey_dist <= 4:
            s[_axis_action(*self._prey_dir)] += 0.9 * need_e
        if fear > 0.0 and self._pred_dir is not None:
            s[_opposite(_axis_action(*self._pred_dir))] += 2.6 * fear
        if self._hazard_dir is not None:
            near = _clip01(1.0 - (self._hazard_dist - 1.0) / max(CUE_R, 1))
            s[_opposite(_axis_action(*self._hazard_dir))] += 3.2 * max(near, 0.4)
        if fatigue_hi > 0.0:
            s[ACT_REST] += 1.8 * fatigue_hi
        if self._shelter_here:
            s[ACT_REST] += 0.25
        if poison_frac > 0.2:
            s[ACT_REST] += 0.8 * poison_frac
        if warmth < 0.35 and self._shelter_dir is not None:
            s[_axis_action(*self._shelter_dir)] += 1.2 * (0.35 - warmth) * 3.0
        if warmth < 0.30 and self._shelter_dir is None:
            s[ACT_REST] += 0.5
        if warmth > 0.95:
            s[ACT_REST] += 0.6 * (warmth - 0.95) * 10.0

        b = self.build
        light = getattr(self, "_light", 0.5)
        w_need = int(b.get("shelter_wood", 4))
        s_need = int(b.get("shelter_stone", 2))
        fire_ready = (wood >= int(b.get("campfire_wood", 2))
                      and stone >= int(b.get("campfire_stone", 1)))
        # action availability mask: block crafting actions that cannot succeed,
        # otherwise the read-out keeps wasting stamina on them
        here_campfire = bool(getattr(self, "_campfire_here", False))
        self._avail = {
            "fire": bool((here_campfire and wood >= 1) or fire_ready),
            "shelter": bool((not self._shelter_here) and wood >= w_need and stone >= s_need),
        }
        # keep gathering until there is enough for a shelter (or a fire)
        missing = (wood < w_need or stone < s_need)
        if missing and self._material_dir is not None:
            s[_axis_action(*self._material_dir)] += 0.5 + 1.1 * max(0.0, 0.65 - warmth)
        if not self._fire_near and fire_ready and (warmth < 0.62 or light < 0.2):
            s[ACT_FIRE] += 1.3 + 1.9 * max(0.0, 0.62 - warmth)
        if (not self._shelter_here and warmth < 0.7
                and wood >= w_need and stone >= s_need):
            s[ACT_SHELTER] += 1.0 + 1.0 * max(0.0, 0.7 - warmth)

        self._instinct = s
        self._urge = {"energy": need_e, "thirst": need_t, "fatigue": fatigue_hi,
                      "fear": fear, "warmth": warmth, "poison": poison_frac}

    def decide(self, net_scores: np.ndarray, rng: np.random.Generator) -> int:
        tr = self.traits
        inst_gain = float(tr.get("instinct_gain", self.dec["instinct_gain"]))
        net_gain = float(tr.get("network_gain", self.dec["network_gain"]))
        noise = float(tr.get("explore_noise", self.dec["explore_noise"]))
        s = inst_gain * self._instinct + net_gain * net_scores.astype(np.float32)
        avail = getattr(self, "_avail", None)
        if avail:
            if not avail.get("fire", False):
                s[ACT_FIRE] = -50.0
            if not avail.get("shelter", False):
                s[ACT_SHELTER] = -50.0
        s[int(self.action)] += float(self.dec["sticky"])
        s += rng.normal(0.0, noise, size=len(ACTIONS)).astype(np.float32)
        self._scores = s
        return int(np.argmax(s))

    # --- acting -------------------------------------------------------------
    def act(self, action: int, creatures, tick: int, rng: np.random.Generator):
        world = self.world
        self.action = int(action)
        msgs: List[str] = []
        tags: List[str] = []
        moved = False
        reward = 0.0
        eat = self.cfg["eat"]
        b = self.build

        def _walkable(nx: int, ny: int) -> bool:
            if world.is_blocked(nx, ny):
                return False
            if world.on_fire(nx, ny, tick) or world.in_plague(nx, ny, tick):
                return False
            return True

        if action in DIR_VEC:
            dx, dy = DIR_VEC[action]
            if not _walkable(self.x + dx, self.y + dy):
                for a in np.argsort(-self._scores):
                    if int(a) in DIR_VEC:
                        ox, oy = DIR_VEC[int(a)]
                        if _walkable(self.x + ox, self.y + oy):
                            self.action = int(a)
                            dx, dy = ox, oy
                            moved = True
                            break
            else:
                moved = True
            if moved:
                self.x += dx
                self.y += dy
                self.distance += 1
                # loose wood / stone is picked up automatically while walking
                res = world.resource_at(self.x, self.y, tick)
                if res in (RES_WOOD, RES_STONE):
                    key = "wood" if res == RES_WOOD else "stone"
                    if self.inventory.get(key, 0) < self.inv_cap.get(key, 8):
                        world.take_resource(self.x, self.y, tick)
                        self.inventory[key] = self.inventory.get(key, 0) + 1
                        reward += float(b.get("reward_material", 0.04)) * 0.5
                        tags.append("gather")
                        msgs.append(f"顺手拾取{'木材' if key == 'wood' else '石料'}（{self.inventory[key]}）")
            else:
                msgs.append("前方无法通行")

        elif action == ACT_INTERACT:
            done = False
            res = world.resource_at(self.x, self.y, tick)
            if res in (RES_BERRY, RES_MUSHROOM) and self.energy < 95.0:
                world.take_resource(self.x, self.y, tick)
                if res == RES_BERRY:
                    e_gain, t_gain = float(eat["berry_energy"]), float(eat["berry_thirst"])
                else:
                    e_gain, t_gain = float(eat.get("mushroom_energy", 34.0)), 0.0
                gain = min(e_gain, self.max_stat - self.energy)
                self.energy = min(self.max_stat, self.energy + e_gain)
                self.thirst = min(self.max_stat, self.thirst + t_gain)
                reward += float(self.reward_cfg.get("eat_berry", 0.6)) * _clip01(gain / max(e_gain, 1e-6))
                tags.append("eat")
                msgs.append("吃下浆果" if res == RES_BERRY else "吃下蘑菇")
                done = True
            if not done and res == RES_POISON:
                risk_energy = float(self.cfg.get("poison_berry", {}).get("min_energy_to_risk", 35.0))
                if self.energy < risk_energy:
                    world.take_resource(self.x, self.y, tick)
                    pb = self.cfg.get("poison_berry", {})
                    self.energy = min(self.max_stat, self.energy + float(pb.get("energy", 18.0)))
                    self.poison = min(self.max_poison, self.poison + float(pb.get("poison_ticks", 30.0)))
                    reward += float(self.reward_cfg.get("eat_berry", 0.6)) * 0.4
                    tags.append("poison")
                    msgs.append("饥不择食，吞下毒浆果！")
                    done = True
                else:
                    msgs.append("那是毒浆果，忍住了")
            if not done:
                targets = [c for c in creatures if c.is_edible()
                           and max(abs(c.x - self.x), abs(c.y - self.y)) <= 1]
                if targets:
                    c = targets[0]
                    if rng.random() < float(c.spec["catch_chance"]):
                        creatures.remove(c)
                        gain = min(float(c.spec.get("energy", 40.0)), self.max_stat - self.energy)
                        self.energy = min(self.max_stat, self.energy + float(c.spec.get("energy", 40.0)))
                        reward += float(self.reward_cfg.get("hunt", 1.0)) * max(0.35, _clip01(gain / 40.0))
                        tags.append("hunt")
                        msgs.append(f"捕获{c.cn}，饱餐一顿")
                        done = True
                    else:
                        msgs.append(f"{c.cn}灵巧地逃走了")
            if not done and self._can_drink and self.thirst < 99.5:
                need = _clip01((self.max_stat - self.thirst) / 40.0)
                self.thirst = min(self.max_stat, self.thirst + float(eat["drink_thirst"]))
                self.energy = min(self.max_stat, self.energy + float(eat["drink_energy"]))
                self.temp -= 0.8
                reward += float(self.reward_cfg.get("drink", 0.35)) * need
                tags.append("drink")
                msgs.append("饮水解渴")
                done = True
            if not done and res in (RES_WOOD, RES_STONE):
                key = "wood" if res == RES_WOOD else "stone"
                if self.inventory.get(key, 0) < self.inv_cap.get(key, 8):
                    world.take_resource(self.x, self.y, tick)
                    self.inventory[key] = self.inventory.get(key, 0) + 1
                    reward += float(b.get("reward_material", 0.04))
                    tags.append("gather")
                    msgs.append(f"拾取{('木材' if key == 'wood' else '石料')}（{self.inventory[key]}）")
                    done = True
                else:
                    msgs.append("背包已满")
            if not done and self.inventory.get("food", 0) > 0 and self.energy < 95.0:
                self.inventory["food"] -= 1
                self.energy = min(self.max_stat, self.energy + float(eat.get("stored_energy", 24.0)))
                self.thirst = min(self.max_stat, self.thirst + 2.0)
                reward += float(self.reward_cfg.get("eat_berry", 0.6)) * 0.7
                tags.append("eat")
                msgs.append("吃掉储存的食物")
                done = True
            if not done:
                msgs.append("这里没有可采集的东西")

        elif action == ACT_FIRE:
            self.fatigue = min(100.0, self.fatigue + float(b.get("fatigue_cost", 2.0)))
            st = world.structure_at(self.x, self.y, tick)
            if st is not None and st["kind"] == "campfire":
                if self.inventory.get("wood", 0) >= 1:
                    self.inventory["wood"] -= 1
                    world.refuel_campfire(self.x, self.y, tick, int(b.get("campfire_refuel", 900)))
                    reward += float(b.get("reward_refuel", 0.15))
                    tags.append("build")
                    msgs.append("给营火添柴，火更旺了")
                else:
                    msgs.append("没有木材可以添柴")
            elif (self.inventory.get("wood", 0) >= int(b.get("campfire_wood", 2))
                  and self.inventory.get("stone", 0) >= int(b.get("campfire_stone", 1))):
                self.inventory["wood"] -= int(b.get("campfire_wood", 2))
                self.inventory["stone"] -= int(b.get("campfire_stone", 1))
                world.add_structure(self.x, self.y, "campfire", tick, int(b.get("campfire_life", 1500)))
                reward += float(b.get("reward_campfire", 0.3))
                tags.append("build")
                msgs.append("点燃了一堆营火")
            else:
                msgs.append(f"材料不足（需 {int(b.get('campfire_wood', 2))} 木材 + "
                            f"{int(b.get('campfire_stone', 1))} 石料）")

        elif action == ACT_SHELTER:
            self.fatigue = min(100.0, self.fatigue + float(b.get("fatigue_cost", 2.0)))
            if self._shelter_here:
                msgs.append("这里已经有庇护所了")
            elif (self.inventory.get("wood", 0) >= int(b.get("shelter_wood", 4))
                  and self.inventory.get("stone", 0) >= int(b.get("shelter_stone", 2))):
                self.inventory["wood"] -= int(b.get("shelter_wood", 4))
                self.inventory["stone"] -= int(b.get("shelter_stone", 2))
                world.add_structure(self.x, self.y, "shelter", tick, int(b.get("shelter_life", 6000)))
                reward += float(b.get("reward_shelter", 0.4))
                tags.append("build")
                msgs.append("搭起了一座庇护所")
            else:
                msgs.append(f"材料不足（需 {int(b.get('shelter_wood', 4))} 木材 + "
                            f"{int(b.get('shelter_stone', 2))} 石料）")
        return msgs, moved, reward, tags

    def apply_creature_effects(self, effects: dict) -> None:
        if "poison" in effects:
            self.poison = min(self.max_poison, self.poison + float(effects["poison"]))

    # --- metabolism ---------------------------------------------------------
    def metabolise(self, tick: int, weather_temp: float, wf: dict, moved: bool, hazards: dict):
        world = self.world
        m = self.cfg["metabolism"]
        rw = self.reward_cfg
        msgs: List[str] = []
        tags: List[str] = []
        code = world.biome_at(self.x, self.y)
        st = world.structure_at(self.x, self.y, tick)
        sheltered = st is not None and st["kind"] == "shelter"

        cost = float(m["energy_base"]) * wf["energy"]
        if moved:
            cost += float(m["energy_move"]) * wf["move"] * min(float(WALK_COST[code]), 2.5) / 1.5
        if self.action == ACT_REST:
            cost += float(m["energy_rest"]) * (1.4 if sheltered else 1.0)
        cold = max(0.0, 18.0 - self.temp) / 18.0 * float(m["energy_cold"])
        self.energy = max(0.0, self.energy - max(cost, -0.03) - cold)

        thirst = float(m["thirst_base"]) * wf["thirst"]
        if moved:
            thirst += float(m["thirst_move"])
        thirst += max(0.0, self.temp - 30.0) * float(m["thirst_heat"]) * 0.05
        self.thirst = max(0.0, self.thirst - thirst)

        if self.action == ACT_REST:
            bonus = float(b_rest) if (b_rest := self.build.get("shelter_rest_bonus", 1.5)) else 1.0
            self.fatigue = max(0.0, self.fatigue + float(m["fatigue_rest"]) * (bonus if sheltered else 1.0))
        else:
            self.fatigue = min(100.0, self.fatigue + (float(m["fatigue_move"]) if moved else float(m["fatigue_awake"])))

        ambient = world.temperature(self.x, self.y, tick, weather_temp)
        insulation = float(INSULATION[code])
        if sheltered:
            insulation *= float(self.build.get("shelter_insulation", 2.0))
        self.temp += float(m["temp_relax"]) * insulation * (ambient - self.temp)

        dmg = 0.0
        if self.energy <= 0.0:
            dmg += float(m["starve_damage"])
            self.cause = "力竭而死"
            tags.append("starve")
        if self.thirst <= 0.0:
            dmg += float(m["dehydrate_damage"])
            self.cause = "脱水而死"
            tags.append("dehydrate")
        if self.temp < float(m["hypothermia_below"]):
            dmg += float(m["cold_damage"]) * (float(m["hypothermia_below"]) - self.temp) / 5.0
            self.cause = "失温冻死"
            tags.append("cold")
        if self.temp > float(m["hyperthermia_above"]):
            dmg += float(m["heat_damage"]) * (self.temp - float(m["hyperthermia_above"])) / 5.0
            self.cause = "中暑热死"
            tags.append("heat")
        if world.on_fire(self.x, self.y, tick):
            dmg += float(hazards.get("fire", 0.9))
            self.cause = "被野火吞噬"
            tags.append("fire")
        if world.in_plague(self.x, self.y, tick):
            dmg += float(hazards.get("plague", 0.12))
            self.cause = "感染疫病"
            tags.append("plague")
        if self.poison > 0.0:
            dmg += float(self.cfg.get("poison_damage", 0.12))
            self.poison = max(0.0, self.poison - 1.0)
            self.cause = "中毒身亡"
            tags.append("poison")
            if self.poison == 0.0:
                msgs.append("体内毒素代谢完毕")

        regen = 0.0
        if (self.energy > 40.0 and self.thirst > 40.0 and 10.0 <= self.temp <= 38.0
                and self.fatigue < 80.0 and dmg == 0.0):
            regen = float(m["hp_regen"])
        self.hp = min(self.max_stat, self.hp + regen - dmg)
        reward = float(rw.get("damage_scale", -0.06)) * dmg
        if dmg == 0.0 and (self.hp > float(rw.get("tonic_hp", 50.0))
                           and self.energy > float(rw.get("tonic_energy", 50.0))
                           and self.thirst > float(rw.get("tonic_thirst", 50.0))):
            reward += float(rw.get("tonic", 0.01))
        self.age += 1
        if self.hp <= 0.0:
            self.hp = 0.0
            self.alive = False
            reward += float(rw.get("death", -1.0))
            msgs.append(f"死亡：{self.cause or '不明原因'}")
        return msgs, (not self.alive), reward, tags

    # --- reporting ----------------------------------------------------------
    def stats(self) -> dict:
        st = self.world.structure_at(self.x, self.y, 0)
        return {"x": self.x, "y": self.y, "hp": self.hp, "energy": self.energy,
                "thirst": self.thirst, "temp": self.temp, "fatigue": self.fatigue,
                "action": self.action, "alive": self.alive, "age": self.age,
                "poison": self.poison, "distance": self.distance,
                "inventory": dict(self.inventory),
                "has_campfire": self.world.campfire_near(self.x, self.y),
                "has_shelter": bool(st is not None and st["kind"] == "shelter"),
                "name": self.name,
                "urge": getattr(self, "_urge", {})}