"""Fly-AI Survival - entry point and simulation loop.

Run a window:
    uv run python main.py
Headless experiment:
    uv run python main.py --headless --ticks 20000 --seed 1 --log logs/run1.csv

The simulation runs a small population of survivors (config `population.size`),
each with its own connectome brain. Resources are shared, so they compete; when
a survivor dies, the best alive member is cloned with mutation (neuroevolution).

Back-compatible API used by the tools: `sim.agent`, `sim.net`, `sim.counters`,
`sim.action_counts` and `sim.deaths` all refer to the *focused* survivor.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, deque
from pathlib import Path
from typing import List, Optional

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent import ACTIONS_CN, SPECIES, Agent  # noqa: E402
from src.events import EventSystem  # noqa: E402
from src.population import Population, Survivor  # noqa: E402
from src.world import BIOME_NAMES, World  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Optional[str] = None):
    p = Path(path) if path else ROOT / "src" / "config.json"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f), p


def _is_number(x) -> bool:
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False


def apply_overrides(cfg: dict, items) -> None:
    """Apply dotted `a.b.c=value` overrides in place (used by --set)."""
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"bad --set {item!r}, expected key.path=value")
        key, raw = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                raise SystemExit(f"unknown config path {key!r}")
            node = node[part]
        if parts[-1] not in node:
            raise SystemExit(f"unknown config key {key!r}")
        old = node[parts[-1]]
        try:
            if isinstance(old, bool):
                val = str(raw).strip().lower() in ("1", "true", "yes", "on")
            elif isinstance(old, int):
                val = int(float(raw))
            elif isinstance(old, float):
                val = float(raw)
            elif isinstance(old, (list, tuple)):
                val = [float(x) if _is_number(x) else x for x in str(raw).split(",")]
            else:
                val = raw
        except ValueError as exc:
            raise SystemExit(f"bad value for {key}: {raw!r} ({exc})")
        node[parts[-1]] = val


class Simulation:
    def __init__(self, cfg: dict, seed: Optional[int] = None, log_path: Optional[Path] = None,
                 snapshot_every: int = 0):
        self.cfg = cfg
        base = int(cfg["seed"] if seed is None else seed)
        self.world = World(cfg, base)
        self.events = EventSystem(cfg, self.world, np.random.default_rng(base + 17))
        self.events.cfg["season_modulation"] = cfg.get("ecology", {}).get("season", {})
        self.pop = Population(cfg, self.world, ROOT / "data",
                              np.random.default_rng(base + 23),
                              int(cfg.get("population", {}).get("size", 1)))
        self.rng = np.random.default_rng(base + 999)
        self.creatures: List = []
        self.tick_count = 0
        self.decide_every = int(cfg["decision"]["decide_every_ticks"])
        self.neural_steps = int(cfg["time"]["neural_steps_per_tick"])
        self.eco = cfg.get("ecology", {})
        self.hazards = {
            "fire": float(cfg["events"]["wildfire"]["damage_per_tick"]),
            "plague": float(cfg["events"]["plague"]["damage_per_tick"]),
        }
        self.messages = deque(maxlen=9)
        self.history = deque(maxlen=1200)
        self.event_counts = {"weather": 0, "fire": 0, "plague": 0, "bloom": 0,
                             "migration": 0, "incursion": 0}
        self.snapshot_every = int(snapshot_every)
        self.snapshots: list = []
        self.log_file = None
        if log_path is not None:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            self.log_file = open(log_path, "w", encoding="utf-8")
            self.log_file.write("tick,day,season,weather,focus,x,y,hp,energy,thirst,temp,fatigue,poison,"
                                "food,wood,stone,campfires,shelters,action,mean_dn_hz,gain,da,"
                                "w_change,focus_alive,population,generation,best_fitness,deaths,"
                                "creatures,fire_tiles,plague\n")

    # --- back-compatible focus API -----------------------------------------
    @property
    def focus_member(self) -> Survivor:
        return self.pop.members[self.pop.focus % len(self.pop.members)]

    @property
    def agent(self) -> Agent:
        return self.focus_member.agent

    @property
    def net(self):
        return self.focus_member.net

    @property
    def action_counts(self) -> np.ndarray:
        return self.focus_member.action_counts

    @property
    def counters(self) -> dict:
        return self.focus_member.counters

    @property
    def deaths(self) -> int:
        return int(sum(m.deaths for m in self.pop.members))

    def close(self):
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None

    def manual_respawn(self) -> None:
        m = self.focus_member
        rng = np.random.default_rng(int(self.rng.integers(1 << 31)))
        spot = self.world.find_spawn(rng, cx=m.agent.x, cy=m.agent.y, max_r=40)
        m.agent.reset(*spot)
        m.first_sense = True

    # --- ecosystem ----------------------------------------------------------
    def _biome_fraction(self, ax: int, ay: int, radius: int, biomes) -> float:
        n = int(self.eco.get("sample_points", 320))
        xs = ax + self.rng.integers(-radius, radius + 1, n)
        ys = ay + self.rng.integers(-radius, radius + 1, n)
        hit = 0
        for i in range(n):
            if BIOME_NAMES[self.world.biome_at(int(xs[i]), int(ys[i]))] in biomes:
                hit += 1
        return hit / max(n, 1)

    def _winter_dieoff(self, season: str, rate: float) -> None:
        for cr in list(self.creatures):
            spec = SPECIES.get(cr.species, {})
            winter_factor = float(spec.get("seasonal", {}).get("winter", 1.0))
            if winter_factor < 0.35 and self.rng.random() < rate:
                self.creatures.remove(cr)

    def _maintain_ecology(self, tick: int) -> None:
        if not bool(self.eco.get("enabled", True)):
            return
        ax, ay = self.agent.x, self.agent.y
        r_active = float(self.eco.get("active_radius", 34))
        ring = float(np.pi * max(r_active ** 2 - 64.0, 1.0))
        light = self.world.light_level(tick)
        season = self.world.calendar(tick)["season"]
        season_cfg = self.eco.get("season", {}).get(season, {})
        density_mult = float(season_cfg.get("density", 1.0))
        dieoff = float(season_cfg.get("cold_dieoff", 0.0))
        if dieoff > 0.0:
            self._winter_dieoff(season, dieoff)
        for key, state in self.eco.get("species", {}).items():
            if key not in SPECIES or not bool(state.get("enabled", True)):
                continue
            spec = SPECIES[key]
            if spec.get("night_only") and light > 0.25:
                continue
            frac = self._biome_fraction(ax, ay, int(r_active), spec["biomes"])
            seasonal = float(spec.get("seasonal", {}).get(season, 1.0))
            target = spec["density"] * ring / 1000.0 * frac * density_mult * seasonal
            active_min = int(round(float(spec.get("min_active", 0)) * density_mult * seasonal))
            target = min(int(spec["max"]), max(active_min, int(round(target))))
            have_near = sum(1 for c in self.creatures
                            if c.species == key and max(abs(c.x - ax), abs(c.y - ay)) <= r_active)
            have_all = sum(1 for c in self.creatures if c.species == key)
            global_cap = int(round(int(spec["max"]) * 1.3)) + 2
            target = min(target, global_cap - have_all)
            for _ in range(max(0, target - have_near)):
                biomes = spec["biomes"]
                spot = self.world.scatter(
                    ax, ay, self.rng, (8, r_active),
                    lambda x, y: (not self.world.is_blocked(x, y)) and BIOME_NAMES[self.world.biome_at(x, y)] in biomes,
                    tries=25)
                if spot is None:
                    break
                from src.agent import Creature
                self.creatures.append(Creature(key, spot[0], spot[1]))

    def _spawn_creatures(self, directives: dict) -> None:
        from src.agent import Creature
        center = directives.get("center", (self.agent.x, self.agent.y))
        dist = directives.get("dist", [10, 30])
        for key, n in directives.get("spawns", []):
            spec = SPECIES[key]
            have = sum(1 for c in self.creatures if c.species == key)
            for _ in range(min(int(n), max(0, int(spec["max"]) - have))):
                biomes = spec["biomes"]
                spot = self.world.scatter(
                    center[0], center[1], self.rng, dist,
                    lambda x, y: (not self.world.is_blocked(x, y)) and BIOME_NAMES[self.world.biome_at(x, y)] in biomes,
                    tries=30)
                if spot is not None:
                    self.creatures.append(Creature(key, spot[0], spot[1]))

    def _despawn(self) -> None:
        r = float(self.eco.get("despawn_radius", self.cfg["creatures"]["despawn_dist"]))
        light = self.world.light_level(self.tick_count)
        ax, ay = self.agent.x, self.agent.y
        for cr in list(self.creatures):
            far = max(abs(cr.x - ax), abs(cr.y - ay)) > r
            if far or (cr.kind == "ambient" and light > 0.35):
                self.creatures.remove(cr)

    # --- main tick ----------------------------------------------------------
    def tick(self) -> List[str]:
        t = self.tick_count
        msgs, directives = self.events.update(t, (self.agent.x, self.agent.y))
        self._count_events(msgs)
        if directives:
            self._spawn_creatures(directives)
        if t % int(self.eco.get("check_every", 40)) == 0:
            self._maintain_ecology(t)

        wf = self.events.weather_factors()
        decide = (t % self.decide_every == 0)
        members = [m for m in self.pop.members if m.agent.alive]
        for m in members:
            if decide or m.first_sense:
                m.first_sense = False
                m.features = m.agent.sense(self.creatures, t, float(wf["vision"]))
                m.net.encode(m.features)
            m.net.step(self.neural_steps, m.net._iext, self.rng)
            m.reward = 0.0
            m.tags = []
            m.acted = []
            m.moved = False

        # decisions + actions
        for m in members:
            if not (decide or m.first_sense):
                continue
            m.first_sense = False
            if decide:
                action = m.agent.decide(m.net.scores_for_decision(), self.rng)
                acted, moved, reward, tags = m.agent.act(action, self.creatures, t, self.rng)
                m.action_counts[m.agent.action] += 1
                m.acted, m.moved = acted, moved
                m.reward += reward
                m.tags += tags

        # wildlife acts against the nearest survivor
        bite_msgs: List[str] = []
        hits_effects: dict = {}
        by_agent = self.pop.by_agent_id()
        if t % max(1, int(self.cfg["creatures"].get("update_every", 1))) == 0 and members:
            targets = [m.agent for m in members]
            for cr in list(self.creatures):
                m_msgs, hits, eff = cr.update(targets, self.cfg, self.world, self.rng, t)
                bite_msgs.extend(m_msgs)
                for target, dmg in hits:
                    mm = by_agent.get(id(target))
                    target.hp -= float(dmg)
                    if mm is not None:
                        mm.reward += float(mm.agent.reward_cfg.get("damage_scale", -0.06)) * float(dmg)
                        mm.tags.append("bite")
                    if eff:
                        hits_effects[id(target)] = dict(eff)
        for key, eff in hits_effects.items():
            mm = by_agent.get(key)
            if mm is not None:
                mm.agent.apply_creature_effects(eff)
                mm.tags.append("poison")
                mm.reward += float(mm.agent.reward_cfg.get("damage_scale", -0.06)) * 0.5 * float(eff.get("poison", 0.0))
        self._despawn()

        # metabolism + learning update + death handling
        death_msgs: List[str] = []
        fitness_cfg = self.cfg.get("population", {}).get("evolution", {})
        survival_bonus = float(fitness_cfg.get("fitness_survival", 0.002))
        for m in list(members):
            ag = m.agent
            met, died, reward, tags = ag.metabolise(t, self.events.temp_delta, wf, m.moved, self.hazards)
            m.reward += reward
            m.tags += tags
            m.net.tick_update(da=float(np.clip(m.reward, -1.0, 1.0)),
                              action=int(ag.action) if decide else None)
            m.fitness += float(np.clip(m.reward, -1.0, 1.0)) + survival_bonus
            for tag in m.tags:
                if tag in m.counters:
                    m.counters[tag] += 1
            if died or not ag.alive:
                death_msgs += self.pop.handle_death(m)
            if m is self.pop.members[self.pop.focus]:
                prefix = ""
                if len(self.pop.members) > 1:
                    prefix = f"[{m.name}] "
                msgs += [prefix + s for s in (m.acted + met)]

        all_msgs = [x for x in (msgs + bite_msgs + death_msgs) if x]
        self.messages.extend(all_msgs)

        st = self.agent.stats()
        learn = self.net.learning_stats()
        focus = self.focus_member
        pop_stats = self.pop.stats()
        self.history.append((t, st["hp"], st["energy"], st["thirst"], st["temp"],
                             self.net.mean_dn_hz(), self.net.gain,
                             learn["da"], learn["w_change"], pop_stats["alive"]))
        if self.log_file is not None:
            cal = self.world.calendar(t)
            inv = st.get("inventory", {})
            structs = {"campfire": 0, "shelter": 0}
            for s_ in self.world.structures.values():
                structs[s_["kind"]] = structs.get(s_["kind"], 0) + 1
            self.log_file.write(
                f"{t},{cal['day']},{cal['season']},{self.events.weather['name']},"
                f"{self.pop.focus},{st['x']},{st['y']},"
                f"{st['hp']:.2f},{st['energy']:.2f},{st['thirst']:.2f},{st['temp']:.2f},"
                f"{st['fatigue']:.2f},{st['poison']:.1f},{inv.get('food', 0)},{inv.get('wood', 0)},"
                f"{inv.get('stone', 0)},{structs['campfire']},{structs['shelter']},"
                f"{ACTIONS_CN[st['action']]},{self.net.mean_dn_hz():.3f},{self.net.gain:.3f},"
                f"{learn['da']:.3f},{learn['w_change']:.5f},{int(st['alive'])},{pop_stats['alive']},"
                f"{focus.generation},{pop_stats['best_fitness']:.2f},"
                f"{pop_stats['deaths_total']},{len(self.creatures)},{len(self.world.fire)},"
                f"{len(self.world.plague)}\n")
        if self.snapshot_every > 0 and t % self.snapshot_every == 0:
            self.snapshots.append({
                "tick": t, "hp": st["hp"], "energy": st["energy"], "thirst": st["thirst"],
                "mean_dn_hz": self.net.mean_dn_hz(), "da": learn["da"],
                "da_ema": learn["da_ema"], "w_change": learn["w_change"],
                "readout_drift": learn["readout_drift"],
                "alive": pop_stats["alive"], "generation": pop_stats["generation"],
                "best_fitness": pop_stats["best_fitness"],
                **{f"n_{k}": v for k, v in self.counters.items()}, "deaths": pop_stats["deaths_total"]})
        self.tick_count += 1
        return all_msgs

    def _count_events(self, msgs: List[str]):
        for m in msgs:
            if "天气转为" in m:
                self.event_counts["weather"] += 1
            if ("燃起野火" in m) or ("火势蔓延" in m):
                self.event_counts["fire"] += 1
            if "疫病" in m:
                self.event_counts["plague"] += 1
            if "果实爆发" in m:
                self.event_counts["bloom"] += 1
            if "兽群" in m:
                self.event_counts["migration"] += 1
            if "天敌" in m:
                self.event_counts["incursion"] += 1

    # --- reporting ----------------------------------------------------------
    def stats(self) -> dict:
        cal = self.world.calendar(self.tick_count)
        st = self.agent.stats()
        pop_stats = self.pop.stats()
        by_species = Counter(c.species for c in self.creatures)
        members = []
        for m in self.pop.members:
            members.append({"name": m.name, "alive": bool(m.agent.alive), "x": m.agent.x,
                            "y": m.agent.y, "hp": m.agent.hp, "energy": m.agent.energy,
                            "fitness": m.fitness, "generation": m.generation,
                            "focus": m is self.focus_member})
        return {"tick": self.tick_count, "day": cal["day"], "season_cn": cal["season_cn"],
                "year": cal["year"], "weather_cn": self.events.weather_cn,
                "weather": self.events.weather["name"], "intensity": self.events.weather["intensity"],
                "agent": st, "deaths": self.deaths, "creatures": len(self.creatures),
                "creatures_by_species": dict(by_species),
                "fire_tiles": len(self.world.fire), "plague": len(self.world.plague),
                "net_hz": self.net.mean_dn_hz(), "gain": self.net.gain,
                "motor_rates": self.net.motor_rates_hz(), "net_scores": self.net.motor_scores(),
                "learning": self.net.learning_stats(), "counters": dict(self.counters),
                "event_counts": dict(self.event_counts), "messages": list(self.messages),
                "light": self.world.light_level(self.tick_count), "temp_delta": self.events.temp_delta,
                "action_counts": self.action_counts.tolist(),
                "inventory": dict(st.get("inventory", {})),
                "structures": {"campfire": sum(1 for s in self.world.structures.values()
                                               if s["kind"] == "campfire"),
                               "shelter": sum(1 for s in self.world.structures.values()
                                              if s["kind"] == "shelter")},
                "population": pop_stats, "members": members}


# --- runners ----------------------------------------------------------------
def run_headless(cfg: dict, ticks: int, seed: Optional[int], log_path: Optional[Path],
                 quiet: bool = False, snapshot_every: int = 0) -> dict:
    sim = Simulation(cfg, seed=seed, log_path=log_path, snapshot_every=snapshot_every)
    t0 = time.perf_counter()
    print_every = max(240, ticks // 10)
    for _ in range(ticks):
        sim.tick()
        if not quiet and sim.tick_count % print_every == 0:
            s = sim.stats()
            a = s["agent"]
            learn = s["learning"]
            pop = s["population"]
            print(f"[{s['tick']:>6} tick | 第{s['day']:>3}天 {s['season_cn']}  {s['weather_cn']:<4}] "
                  f"HP {a['hp']:5.1f}  能量 {a['energy']:5.1f}  水 {a['thirst']:5.1f}  "
                  f"体温 {a['temp']:5.1f}  世代 {pop['generation']:>2}  "
                  f"存活 {pop['alive']}/{pop['size']}  DA {learn['da']:+.2f}  死亡 {pop['deaths_total']}")
    elapsed = time.perf_counter() - t0
    s = sim.stats()
    s["elapsed"] = elapsed
    s["ticks_per_s"] = ticks / max(elapsed, 1e-9)
    s["snapshots"] = list(sim.snapshots)
    sim.close()
    return s


def run_window(cfg: dict, seed: Optional[int], speed: int, log_path: Optional[Path],
               frames: Optional[int] = None):
    from src.render import Renderer

    sim = Simulation(cfg, seed=seed, log_path=log_path)
    r = Renderer(cfg, sim)
    speeds = list(cfg["render"]["speed_options"])
    idx = min(range(len(speeds)), key=lambda i: abs(speeds[i] - speed))
    paused = False
    running = True
    fps = int(cfg["render"]["fps_cap"])
    frame = 0
    while running:
        running, keys = r.handle_events()
        frame += 1
        if frames is not None and frame > frames:
            break
        if keys.get("quit"):
            break
        if keys.get("pause"):
            paused = not paused
        if keys.get("respawn"):
            sim.manual_respawn()
        if keys.get("focus") is not None:
            sim.pop.focus = int(keys["focus"]) % len(sim.pop.members)
        if keys.get("speed") is not None:
            idx = max(0, min(len(speeds) - 1, idx + keys["speed"]))
        if not paused:
            for _ in range(speeds[idx]):
                sim.tick()
        r.draw(sim, paused=paused, speed=speeds[idx])
        r.clock.tick(fps)
    sim.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fly-AI connectome survival world")
    ap.add_argument("--config", default=None)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--ticks", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--speed", type=int, default=1)
    ap.add_argument("--log", default=None)
    ap.add_argument("--frames", type=int, default=None,
                    help="window mode: quit automatically after N frames (testing)")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    help="config override, e.g. --set neural.plasticity.neuromod.lr=0.5")
    ap.add_argument("--snapshot-every", type=int, default=0,
                    help="record metrics every N ticks (used by tools/experiments.py)")
    args = ap.parse_args(argv)

    cfg, cfg_path = load_config(args.config)
    apply_overrides(cfg, args.overrides)
    log_path = Path(args.log) if args.log else None
    if args.headless:
        s = run_headless(cfg, args.ticks, args.seed, log_path,
                         snapshot_every=args.snapshot_every)
        a = s["agent"]
        learn = s["learning"]
        pop = s["population"]
        print("\n=== 无头运行结束 ===")
        print(f"配置 {cfg_path}  模拟 {s['tick']} tick（第 {s['day']} 天，{s['season_cn']}季）")
        print(f"种群 {pop['size']} 个体 | 存活 {pop['alive']} | 总死亡 {pop['deaths_total']} | "
              f"最高世代 {max(pop['generations'])} | 最佳适应度 {pop['best_fitness']:.1f}")
        print(f"焦点最终 HP {a['hp']:.1f} 能量 {a['energy']:.1f} 水 {a['thirst']:.1f} "
              f"体温 {a['temp']:.1f} 毒素 {a['poison']:.0f}")
        print(f"网络平均放电 {s['net_hz']:.2f} Hz | 稳态增益 {s['gain']:.2f} | "
              f"STDP {learn['stdp']} 解码学习 {learn['readout']}")
        print(f"学习：DA EMA {learn['da_ema']:+.3f} | 资格迹 {learn['elig']:.2e} | "
              f"突触改变 {learn['w_change'] * 100:.2f}% | 解码漂移 {learn['readout_drift']:.3f}")
        print(f"焦点行为计数 {s['counters']}")
        print(f"事件计数 {s['event_counts']} | 生态 {s['creatures_by_species']}")
        print(f"性能 {s['ticks_per_s']:.1f} tick/s 用时 {s['elapsed']:.2f}s")
        return 0
    run_window(cfg, args.seed, args.speed, log_path, frames=args.frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())