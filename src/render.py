"""Pygame visualiser: map, connectome activity, motor read-out, stats, event log.

Keys:  ESC quit | SPACE pause | R respawn | + / - speed | TAB toggle heat-map
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pygame

from src.agent import ACTIONS_CN
from src.world import BIOME_NAMES_CN

PALETTE = np.array([
    [18, 44, 86],     # deep water
    [36, 92, 150],    # water
    [214, 196, 132],  # sand
    [86, 148, 72],    # grass
    [44, 102, 54],    # forest
    [128, 124, 116],  # rock
    [74, 102, 74],    # swamp
], dtype=np.uint8)

BG = (16, 18, 24)
PANEL = (26, 30, 40)
TEXT = (226, 230, 238)
DIM = (150, 158, 172)
ACCENT = (255, 205, 92)
BAD = (232, 92, 92)

RES_COLORS = {1: (214, 66, 86), 2: (226, 214, 186), 3: (176, 64, 208),
              4: (152, 106, 60), 5: (176, 176, 182)}

STAT_COLORS = {"hp": (226, 84, 84), "energy": (240, 178, 72), "thirst": (82, 176, 236),
               "temp": (176, 128, 224), "fatigue": (140, 200, 140), "poison": (128, 216, 120)}


def _colormap(v: np.ndarray) -> np.ndarray:
    """Black -> blue -> cyan -> yellow -> white for normalised rates."""
    stops = np.array([0.0, 0.22, 0.48, 0.74, 1.0])
    r = np.interp(v, stops, [6, 8, 18, 250, 255])
    g = np.interp(v, stops, [6, 34, 150, 220, 255])
    b = np.interp(v, stops, [20, 130, 215, 60, 255])
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


class Renderer:
    def __init__(self, cfg: dict, sim):
        pygame.init()
        self.cfg = cfg["render"]
        self.cfg_pop = cfg.get("population", {})
        self.w = int(self.cfg["width"])
        self.h = int(self.cfg["height"])
        self.screen = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Fly-AI 生存世界 · 连接组驱动")
        self.clock = pygame.time.Clock()
        self.fonts: Dict[int, pygame.font.Font] = {}
        self._font_names = str(self.cfg["font"]).split(",")
        self.tile = int(self.cfg["tile_px"])
        self.radius = int(self.cfg["map_radius"])
        self.map_px = (2 * self.radius + 1) * self.tile
        self.map_xy = (14, 52)
        self.right_x = self.map_xy[0] + self.map_px + 14
        self.right_w = self.w - self.right_x - 14
        self.refresh_every = int(self.cfg.get("map_refresh_frames", 4))
        self._map_key = None
        self._map = None
        self.show_heat = True
        self._glow = self._make_glow(max(24, self.tile * 7))
        self._dark = pygame.Surface((self.map_px, self.map_px), pygame.SRCALPHA)
        self.heat_rows, self.heat_cols = 28, 84
        self.heat_cell = max(4, self.right_w // self.heat_cols)
        self._heat_surf = None

    # --- helpers ------------------------------------------------------------
    def font(self, size: int) -> pygame.font.Font:
        f = self.fonts.get(size)
        if f is None:
            f = pygame.font.SysFont(self._font_names, size)
            self.fonts[size] = f
        return f

    def blit_text(self, s: str, x: int, y: int, size: int = 15, color=TEXT,
                  right: bool = False, center: bool = False) -> None:
        img = self.font(size).render(s, True, color)
        if right:
            x -= img.get_width()
        if center:
            x -= img.get_width() // 2
        self.screen.blit(img, (x, y))

    def bar(self, x, y, w, h, frac, color, label, value_text):
        frac = max(0.0, min(1.0, float(frac)))
        pygame.draw.rect(self.screen, (44, 48, 60), (x, y, w, h), border_radius=3)
        pygame.draw.rect(self.screen, color, (x, y, int(w * frac), h), border_radius=3)
        pygame.draw.rect(self.screen, (90, 96, 110), (x, y, w, h), 1, border_radius=3)
        self.blit_text(label, x + 6, y + 1, 13, (20, 22, 28))
        self.blit_text(value_text, x + w - 6, y + 1, 13, (240, 242, 248), right=True)

    def _make_glow(self, radius: int) -> pygame.Surface:
        """Warm lantern glow, PRE-MULTIPLIED so BLEND_RGB_ADD never blows out.

        BLEND_RGBA_ADD adds the raw RGB channels and does not attenuate them by
        alpha, so the brightness falloff must already live in the RGB values.
        """
        n = max(8, radius * 2)
        yy, xx = np.mgrid[0:n, 0:n]
        d = np.sqrt((xx - n / 2) ** 2 + (yy - n / 2) ** 2) / (n / 2)
        falloff = np.clip(1.0 - d, 0.0, 1.0) ** 2
        warm = np.array([255.0, 224.0, 150.0])
        rgb = (falloff[..., None] * warm[None, None, :] * 0.25).astype(np.uint8)
        surf = pygame.Surface((n, n), pygame.SRCALPHA)
        p3 = pygame.surfarray.pixels3d(surf)
        p3[:] = rgb.transpose(1, 0, 2)
        del p3
        pa = pygame.surfarray.pixels_alpha(surf)
        pa[:] = 0
        del pa
        return surf

    # --- input --------------------------------------------------------------
    def handle_events(self):
        keys = {"quit": False, "pause": False, "respawn": False, "speed": None,
                "focus": None}
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                keys["quit"] = True
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    keys["quit"] = True
                elif e.key == pygame.K_SPACE:
                    keys["pause"] = True
                elif e.key == pygame.K_r:
                    keys["respawn"] = True
                elif e.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    keys["speed"] = 1
                elif e.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    keys["speed"] = -1
                elif e.key == pygame.K_TAB:
                    self.show_heat = not self.show_heat
                elif pygame.K_1 <= e.key <= pygame.K_9:
                    keys["focus"] = e.key - pygame.K_1
        return (not keys["quit"]), keys

    # --- map ----------------------------------------------------------------
    def _refresh_map(self, sim, tick: int):
        ax, ay = sim.agent.x, sim.agent.y
        key = (ax, ay, tick // max(1, self.refresh_every))
        if key == self._map_key and self._map is not None:
            return self._map
        self._map_key = key
        self._map = sim.world.window_arrays(ax, ay, self.radius, tick)
        return self._map

    def _draw_map(self, sim, tick: int, light: float):
        m = self._refresh_map(sim, tick)
        codes = m["codes"]
        rgb = PALETTE[codes].astype(np.float32)
        # overlaid tile effects
        rgb[m["ash"]] *= 0.45
        rgb[m["plague"]] = rgb[m["plague"]] * 0.55 + np.array([120, 60, 150]) * 0.45
        rgb[m["bloom"]] = rgb[m["bloom"]] * 0.7 + np.array([190, 170, 60]) * 0.3
        fire = m["fire"]
        if fire.any():
            flick = 0.6 + 0.4 * np.sin(tick * 0.6)
            rgb[fire] = np.array([255, 120 + 60 * flick, 40])
        surf = pygame.surfarray.make_surface(np.ascontiguousarray(rgb.transpose(1, 0, 2).astype(np.uint8)))
        surf = pygame.transform.scale(surf, (self.map_px, self.map_px))
        x0, y0 = self.map_xy
        self.screen.blit(surf, (x0, y0))

        res = m["res"]
        ys, xs = np.nonzero(res)
        for i in range(min(len(ys), 1500)):
            code = int(res[int(ys[i]), int(xs[i])])
            col = RES_COLORS.get(code)
            if col is None:
                continue
            px = int(x0 + (int(xs[i]) + 0.5) * self.tile)
            py = int(y0 + (int(ys[i]) + 0.5) * self.tile)
            if code in (4, 5):  # wood / stone: square chunk
                s = max(2, self.tile // 3)
                pygame.draw.rect(self.screen, col, (px - s // 2, py - s // 2, s, s))
            else:               # berry / mushroom / poison berry
                pygame.draw.circle(self.screen, col, (px, py), max(2, self.tile // 4))

        # structures
        for mask, kind in ((m["campfire"], "campfire"), (m["shelter"], "shelter")):
            sys_, sxs = np.nonzero(mask)
            for i in range(len(sys_)):
                px = int(x0 + (int(sxs[i]) + 0.5) * self.tile)
                py = int(y0 + (int(sys_[i]) + 0.5) * self.tile)
                if kind == "campfire":
                    r = max(3, int(self.tile * 0.45))
                    pygame.draw.circle(self.screen, (255, 140 + int(40 * (0.5 + 0.5 * np.sin(tick * 0.8))), 50),
                                       (px, py), r)
                    pygame.draw.circle(self.screen, (90, 50, 20), (px, py), r, 1)
                else:
                    r = max(3, int(self.tile * 0.5))
                    pygame.draw.rect(self.screen, (150, 110, 70), (px - r, py - r, 2 * r, 2 * r))
                    pygame.draw.polygon(self.screen, (110, 78, 50),
                                        [(px - r, py - r), (px, py - r - 3), (px + r, py - r)])

        ax, ay = sim.agent.x - m["x0"], sim.agent.y - m["y0"]
        # creatures by species: circles=prey, triangle=predator, diamond=ambusher,
        # ringed dot=toxic, pulsing dot=firefly
        for c in sim.creatures:
            cx, cy = c.x - m["x0"], c.y - m["y0"]
            if 0 <= cx <= 2 * self.radius and 0 <= cy <= 2 * self.radius:
                px = int(x0 + (cx + 0.5) * self.tile)
                py = int(y0 + (cy + 0.5) * self.tile)
                col = c.color
                r = max(3, self.tile // 2)
                if c.kind == "prey":
                    pygame.draw.circle(self.screen, col, (px, py), max(2, self.tile // 4))
                elif c.kind == "ambient":
                    pulse = 0.5 + 0.5 * np.sin(tick * 0.35 + (c.x * 13 + c.y * 7) % 6)
                    pygame.draw.circle(self.screen, col, (px, py), max(2, int(2 + 2 * pulse)))
                elif c.kind == "predator":
                    pygame.draw.polygon(self.screen, col,
                                        [(px, py - r), (px - r, py + r), (px + r, py + r)])
                elif c.kind == "ambusher":
                    pygame.draw.polygon(self.screen, col,
                                        [(px, py - r), (px - r, py), (px, py + r), (px + r, py)])
                else:  # toxic
                    pygame.draw.circle(self.screen, col, (px, py), max(2, self.tile // 3))
                    pygame.draw.circle(self.screen, (90, 40, 96), (px, py), max(3, self.tile // 2), 1)
        # survivors (focus gets a white ring and a name label)
        colors = self.cfg_pop.get("colors", [[255, 214, 80]])
        focus = sim.focus_member
        px = py = None
        for idx, member in enumerate(sim.pop.members):
            ag = member.agent
            if not ag.alive:
                continue
            mx, my = ag.x - m["x0"], ag.y - m["y0"]
            if not (0 <= mx <= 2 * self.radius and 0 <= my <= 2 * self.radius):
                continue
            px = int(x0 + (mx + 0.5) * self.tile)
            py = int(y0 + (my + 0.5) * self.tile)
            col = tuple(colors[idx % len(colors)])
            r = max(3, self.tile // 2)
            if member is focus:
                pygame.draw.circle(self.screen, (255, 255, 255), (px, py), r + 2, 1)
            pygame.draw.circle(self.screen, col, (px, py), r)
            pygame.draw.circle(self.screen, (40, 34, 20), (px, py), r, 1)
            if ag.poison > 0.0:
                pygame.draw.circle(self.screen, (150, 235, 120), (px, py), r + 1, 2)
            if member is focus:
                self.blit_text(f"{ag.name} 第{member.generation}代", px + r + 3, py - r - 8, 12, (255, 250, 210))
        if px is None:  # focus not visible: fall back to its own position
            px = int(x0 + (ax + 0.5) * self.tile)
            py = int(y0 + (ay + 0.5) * self.tile)

        # smooth day/night curve: no step at the dusk threshold
        darkness = int(np.clip((0.72 - light) / 0.72, 0.0, 1.0) ** 1.15 * 190.0)
        if darkness > 0:
            self._dark.fill((2, 4, 16, darkness))
            self.screen.blit(self._dark, (x0, y0))
        if light < 0.8:
            self.screen.blit(self._glow, (px - self._glow.get_width() // 2,
                                          py - self._glow.get_height() // 2),
                             special_flags=pygame.BLEND_RGB_ADD)
        pygame.draw.rect(self.screen, (70, 78, 92), (x0 - 1, y0 - 1, self.map_px + 2, self.map_px + 2), 1)

    # --- panels -------------------------------------------------------------
    def _draw_heatmap(self, sim):
        x, y = self.right_x, self.map_xy[1]
        self.blit_text("连接组活动（2594 神经元放电率）", x, y - 20, 14, DIM)
        if not self.show_heat:
            self.blit_text("已隐藏（TAB 显示）", x, y + 40, 15, DIM)
            return 0
        grid = sim.net.activity_grid(self.heat_rows, self.heat_cols) / 45.0
        rgb = _colormap(np.clip(grid, 0, 1))
        surf = pygame.surfarray.make_surface(np.ascontiguousarray(rgb.transpose(1, 0, 2)))
        surf = pygame.transform.scale(surf, (self.heat_cols * self.heat_cell, self.heat_rows * self.heat_cell))
        self.screen.blit(surf, (x, y))
        pygame.draw.rect(self.screen, (70, 78, 92), (x - 1, y - 1, surf.get_width() + 2, surf.get_height() + 2), 1)
        learn = sim.net.learning_stats()
        tag = []
        if learn["stdp"]:
            tag.append(f"STDP资格迹 {learn['elig']:.1e}")
        if learn["readout"]:
            tag.append(f"解码漂移 {learn['readout_drift']:.3f}")
        self.blit_text(f"DA {learn['da']:+.2f} | 突触Δ {learn['w_change'] * 100:+.3f}% | " + " | ".join(tag),
                       x, y + surf.get_height() + 4, 12,
                       (120, 230, 170) if learn["da"] > 0 else (240, 140, 170))
        return surf.get_height()

    def _draw_motor(self, sim, x, y):
        st = sim.stats()
        rates = st["motor_rates"]
        scores = st["net_scores"]
        chosen = int(st["agent"]["action"])
        self.blit_text("下行神经元 → 动作解码（放电率 / 网络得分）", x, y, 14, DIM)
        y += 22
        vmax = max(float(np.max(rates)) if len(rates) else 1.0, 0.5)
        for i, name in enumerate(ACTIONS_CN):
            r = float(rates[i])
            frac = r / vmax
            col = ACCENT if i == chosen else (86, 150, 210)
            self.bar(x, y, self.right_w, 14, frac, col, f"{i}", f"{r:4.1f} Hz  z={float(scores[i]):+.2f}")
            self.blit_text(name, x + self.right_w - 118, y, 13,
                           (30, 30, 30) if i == chosen else (220, 224, 232))
            y += 17
        return y

    def _draw_history(self, sim, x, y, w, h):
        self.blit_text("生理与网络历史（最近 300 tick）", x, y, 14, DIM)
        y += 20
        pygame.draw.rect(self.screen, PANEL, (x, y, w, h), border_radius=4)
        hist = list(sim.history)[-300:]
        if len(hist) < 2:
            return y + h
        arr = np.array(hist, dtype=np.float64)[:, 1:]
        n = arr.shape[0]
        xs = np.linspace(x + 2, x + w - 2, n)
        series = [(0, (226, 84, 84), 100.0), (1, (240, 178, 72), 100.0),
                  (2, (82, 176, 236), 100.0), (3, (176, 128, 224), 45.0)]
        for idx, col, scale in series:
            vals = np.clip(arr[:, idx] / scale, 0, 1) * (h - 4)
            pts = [(float(xs[i]), float(y + h - 2 - vals[i])) for i in range(n)]
            pygame.draw.lines(self.screen, col, False, pts, 2)
        hz = arr[:, 4]
        vmax = max(1.0, float(hz.max()))
        vals = np.clip(hz / min(vmax, 30.0), 0, 1) * (h - 4)
        pts = [(float(xs[i]), float(y + h - 2 - vals[i])) for i in range(n)]
        pygame.draw.lines(self.screen, (140, 240, 200), False, pts, 2)
        da = np.clip((arr[:, 7] + 1.0) / 2.0, 0.0, 1.0) * (h - 4)
        pts = [(float(xs[i]), float(y + h - 2 - da[i])) for i in range(n)]
        pygame.draw.lines(self.screen, (255, 120, 210), False, pts, 2)
        legend = [("HP", (226, 84, 84)), ("能量", (240, 178, 72)), ("水", (82, 176, 236)),
                  ("体温", (176, 128, 224)), ("DN Hz", (140, 240, 200)), ("DA", (255, 120, 210))]
        lx = x + 8
        for name, col in legend:
            pygame.draw.rect(self.screen, col, (lx, y + h + 4, 10, 8))
            self.blit_text(name, lx + 13, y + h - 1, 12, DIM)
            lx += 74
        return y + h + 22

    def _draw_stats(self, sim, x, y):
        a = sim.stats()["agent"]
        self.blit_text("生存状态", x, y, 14, DIM)
        y += 20
        rows = [("HP", a["hp"], 100.0, STAT_COLORS["hp"], f"{a['hp']:.0f}"),
                ("能量", a["energy"], 100.0, STAT_COLORS["energy"], f"{a['energy']:.0f}"),
                ("口渴", a["thirst"], 100.0, STAT_COLORS["thirst"], f"{a['thirst']:.0f}"),
                ("体温", a["temp"] + 10.0, 55.0, STAT_COLORS["temp"], f"{a['temp']:.1f}°C"),
                ("疲劳", a["fatigue"], 100.0, STAT_COLORS["fatigue"], f"{a['fatigue']:.0f}"),
                ("毒素", a["poison"], max(float(sim.agent.max_poison), 1.0),
                 STAT_COLORS["poison"], f"{a['poison']:.0f}")]
        for label, v, scale, col, txt in rows:
            self.bar(x, y, self.right_w, 15, v / scale, col, label, txt)
            y += 17
        inv = a.get("inventory", {})
        pop = sim.stats().get("population", {})
        self.blit_text(f"背包  食物 {inv.get('food', 0)}/8  木材 {inv.get('wood', 0)}/8  "
                       f"石料 {inv.get('stone', 0)}/6   "
                       f"营火 {'有' if a.get('has_campfire') else '无'}  "
                       f"庇护所 {'有' if a.get('has_shelter') else '无'}", x, y + 2, 12, DIM)
        fits = pop.get("fitness", [])
        self.blit_text("适应度  " + "  ".join(f"#{i + 1}:{f:.0f}" for i, f in enumerate(fits))
                       + f"   （按 1..{len(fits)} 切换焦点）", x, y + 18, 12, DIM)
        return y + 36

    def _draw_header(self, sim, paused, speed):
        st = sim.stats()
        pygame.draw.rect(self.screen, (22, 26, 34), (0, 0, self.w, 42))
        self.blit_text("Fly-AI 生存世界", 14, 9, 20, ACCENT)
        pop = st.get("population", {})
        self.blit_text(f"第 {st['day']:>3} 天（{st['season_cn']}·第{st['year']}年）  "
                       f"天气 {st['weather_cn']} {st['intensity']:.0%}  "
                       f"种群 {pop.get('alive', 1)}/{pop.get('size', 1)}  "
                       f"焦点 {pop.get('focus_name', '')} 第 {pop.get('generation', 0)} 代  "
                       f"最佳适应度 {pop.get('best_fitness', 0):.0f}  "
                       f"总死亡 {pop.get('deaths_total', 0)}", 220, 12, 15, TEXT)
        tag = "暂停" if paused else f"速度 x{speed}"
        col = BAD if paused else (150, 240, 180)
        self.blit_text(f"{tag}   FPS {self.clock.get_fps():4.1f}", self.w - 14, 12, 16, col, right=True)

    def _draw_log(self, sim, x, y, w, h):
        pygame.draw.rect(self.screen, PANEL, (x, y, w, h), border_radius=4)
        self.blit_text("事件日志 / 自然事件", x + 8, y + 5, 14, DIM)
        self.blit_text("●幸存者  ●猎物(果蝇/蜜蚜)  ▲黄蜂  ◆猎蛛  ◎毒蛾幼虫  ✦萤火虫",
                       x + 170, y + 6, 12, DIM)
        msgs = list(sim.messages)[-5:]
        yy = y + 24
        for m in msgs:
            if len(m) > 44:
                m = m[:43] + "…"
            self.blit_text(m, x + 10, yy, 14, TEXT)
            yy += 15
        st = sim.stats()
        ev = st["event_counts"]
        sp = st["creatures_by_species"]
        names = {"fly": "果蝇", "aphid": "蜜蚜", "spider": "猎蛛",
                 "wasp": "黄蜂", "caterpillar": "毒虫", "firefly": "萤火虫"}
        pop = " ".join(f"{names.get(k, k)}{v}" for k, v in sorted(sp.items())) or "无"
        struct = st.get("structures", {})
        self.blit_text(f"天气 {ev['weather']} | 野火 {ev['fire']} | 疫病 {ev['plague']} | "
                       f"果实 {ev['bloom']} | 迁徙 {ev['migration']} | 天敌 {ev['incursion']} | "
                       f"营火 {struct.get('campfire', 0)} 庇护所 {struct.get('shelter', 0)} | "
                       f"生物 {st['creatures']}（{pop}）", x + 10, y + h - 20, 13, DIM)

    # --- main draw ----------------------------------------------------------
    def draw(self, sim, paused: bool = False, speed: int = 1):
        self.screen.fill(BG)
        tick = sim.tick_count
        self._draw_header(sim, paused, speed)
        self._draw_map(sim, tick, sim.world.light_level(tick))
        hh = 0
        if self.show_heat:
            hh = self._draw_heatmap(sim)
            y = self.map_xy[1] + hh + 26
        else:
            y = self.map_xy[1] + 60
        y = self._draw_motor(sim, self.right_x, y)
        y = self._draw_history(sim, self.right_x, y + 8, self.right_w, 92)
        self._draw_stats(sim, self.right_x, y + 6)
        self._draw_log(sim, self.map_xy[0], self.map_xy[1] + self.map_px + 6,
                       self.map_px, self.h - (self.map_xy[1] + self.map_px) - 14)
        pygame.display.flip()