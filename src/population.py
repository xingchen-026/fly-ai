"""Survivor population: several connectome-driven agents in one shared world.

Each member owns its own brain (its own LCN structural seed, so the input group,
motor groups and encoder differ between individuals) and its own agent. Survival
is competitive: berries, mushrooms, wood and stone are shared world resources.

When a member dies, a new individual is born by cloning the best alive member
(neuroevolution): connectome weights + plastic decode layer + behavioural traits
are inherited with Gaussian mutation. If evolution is disabled, the same brain
simply reawakens at a new spawn point.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.agent import ACTIONS_CN, Agent
from src.neural import ConnectomeNet

TRAIT_BOUNDS = {"instinct_gain": (0.4, 2.0), "network_gain": (0.0, 1.5),
                "explore_noise": (0.01, 0.25)}


class Survivor:
    def __init__(self, sid: int, net: ConnectomeNet, agent: Agent, generation: int = 0):
        self.id = int(sid)
        self.net = net
        self.agent = agent
        self.generation = int(generation)
        self.fitness = 0.0
        self.deaths = 0
        self.first_sense = True
        self.action_counts = np.zeros(len(ACTIONS_CN), dtype=np.int64)
        self.counters = {k: 0 for k in ("eat", "drink", "hunt", "gather", "build",
                                        "bite", "poison", "fire", "plague", "cold",
                                        "heat", "starve", "dehydrate")}
        # per-tick scratch
        self.reward = 0.0
        self.moved = False
        self.tags: List[str] = []
        self.acted: List[str] = []
        self.features = np.zeros(int(net.nc["n_features"]), dtype=np.float32)

    @property
    def name(self) -> str:
        return self.agent.name


class Population:
    def __init__(self, cfg: dict, world, data_dir: Path, rng: np.random.Generator, size: int):
        self.cfg = cfg
        self.world = world
        self.rng = rng
        self.size = max(1, int(size))
        self.members: List[Survivor] = []
        self.focus = int(cfg.get("population", {}).get("focus", 0)) % self.size
        struct_seed = int(cfg["neural"].get("seed", 7))
        base_traits = dict(cfg["decision"])
        for i in range(self.size):
            net = ConnectomeNet(cfg, data_dir, seed=struct_seed + i)
            if i > 0:  # initial behavioural diversity
                net.readout += 0.12 * rng.standard_normal(net.readout.shape).astype(np.float32)
                np.clip(net.readout, -net.rl_clip, net.rl_clip, out=net.readout)
            agent = Agent(cfg, world, np.random.default_rng(struct_seed * 131 + i * 17 + 5),
                          name=f"幸存者{i + 1}")
            agent.traits = {
                "instinct_gain": float(np.clip(base_traits["instinct_gain"] * (1 + 0.12 * rng.standard_normal()),
                                               *TRAIT_BOUNDS["instinct_gain"])),
                "network_gain": float(np.clip(base_traits["network_gain"] * (1 + 0.15 * rng.standard_normal()),
                                              *TRAIT_BOUNDS["network_gain"])),
                "explore_noise": float(np.clip(base_traits["explore_noise"] * (1 + 0.2 * rng.standard_normal()),
                                               *TRAIT_BOUNDS["explore_noise"])),
            }
            spawn = world.find_spawn(np.random.default_rng(struct_seed * 977 + i * 31))
            agent.reset(*spawn)
            self.members.append(Survivor(i, net, agent))

    # --- helpers ------------------------------------------------------------
    @property
    def alive(self) -> List[Survivor]:
        return [m for m in self.members if m.agent.alive]

    def by_agent_id(self) -> Dict[int, Survivor]:
        return {id(m.agent): m for m in self.members}

    def mark_resense(self) -> None:
        for m in self.members:
            m.first_sense = True

    def best(self, exclude: Optional[Survivor] = None) -> Survivor:
        pool = [m for m in self.members if m is not exclude and m.agent.alive]
        if not pool:
            pool = [m for m in self.members if m is not exclude] or self.members
        return max(pool, key=lambda m: m.fitness)

    # --- life / death / evolution ------------------------------------------
    def handle_death(self, member: Survivor) -> List[str]:
        cfgp = self.cfg.get("population", {})
        ev = cfgp.get("evolution", {})
        member.deaths += 1
        msgs = [f"{member.name} 第 {member.deaths} 次死亡"
                f"（第 {member.generation} 代，适应度 {member.fitness:.1f}）"]
        if bool(ev.get("enabled", True)):
            parent = self.best(exclude=member) if bool(ev.get("clone_best", True)) else member
            new_net = parent.net.clone_with_mutation(
                self.rng,
                rate=float(ev.get("mutation_synapse_rate", 0.004)),
                sigma=float(ev.get("mutation_sigma", 0.10)),
                readout_sigma=float(ev.get("mutation_readout_sigma", 0.10)))
            traits = {}
            for key, bound in TRAIT_BOUNDS.items():
                base = float(parent.agent.traits.get(key, self.cfg["decision"][key]))
                jitter = float(ev.get("mutation_trait_sigma", 0.06)) * float(self.rng.standard_normal())
                traits[key] = float(np.clip(base + jitter, bound[0], bound[1]))
            new_agent = Agent(self.cfg, self.world,
                              np.random.default_rng(int(self.rng.integers(1 << 31))),
                              name=member.name)
            lo, hi = ev.get("spawn_radius", [6, 20])
            px, py = parent.agent.x, parent.agent.y
            spot = self.world.scatter(px, py, self.rng, (lo, hi),
                                      lambda x, y: not self.world.is_blocked(x, y), tries=30) or (px, py)
            new_agent.reset(*spot)
            new_agent.traits = traits
            member.net = new_net
            member.agent = new_agent
            member.generation += 1
            member.fitness = 0.0
            member.first_sense = True
            msgs.append(f"{member.name} 第 {member.generation} 代诞生"
                        f"（亲本 {parent.name}，适应度 {parent.fitness:.1f}，"
                        f"脑变异 {float(ev.get('mutation_sigma', 0.1)):.0%}）")
        else:
            rng = np.random.default_rng(int(self.rng.integers(1 << 31)))
            spot = self.world.find_spawn(rng, cx=member.agent.x, cy=member.agent.y, max_r=40)
            member.agent.reset(*spot)
            member.fitness = 0.0
            member.first_sense = True
            msgs.append(f"{member.name} 在附近重新苏醒")
        return msgs

    # --- reporting ----------------------------------------------------------
    def stats(self) -> dict:
        focus = self.members[self.focus % len(self.members)]
        fits = [m.fitness for m in self.members]
        return {"size": len(self.members), "alive": len(self.alive),
                "focus": self.focus, "focus_name": focus.name,
                "generation": focus.generation,
                "generations": [m.generation for m in self.members],
                "deaths_total": int(sum(m.deaths for m in self.members)),
                "best_fitness": float(max(fits)) if fits else 0.0,
                "fitness": [float(f) for f in fits],
                "focus_fitness": float(focus.fitness),
                "evolution": bool(self.cfg.get("population", {}).get("evolution", {}).get("enabled", True))}