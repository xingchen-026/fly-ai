"""LIF spiking network from the FlyWire male-CNS connectome mini subset.

The mini connectome (2594 neurons / 76323 synapses) is almost entirely a
recurrent descending-neuron (DN) graph, so it is used as a fixed reservoir:
world features are projected into a subset of DNs, and the mean firing rates of
six disjoint DN readout groups are decoded into the agent's actions.

Plasticity (all optional, reward-modulated):
  * STDP eligibility traces on connectome synapses (excitatory by default),
    accumulated from pre/post spike pairs and gated by a dopamine-like signal.
  * A plastic decode layer (DN group activity -> 6 actions) updated by the same
    neuromodulator, which makes rewarded action patterns stick.

Synapse counts are compressed with a power law, signed by the presynaptic
neuron's predicted neurotransmitter (ACh/Glu excitatory, GABA/histamine
inhibitory), and balanced to a target excitatory/inhibitory ratio.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

NT_SIGN = {
    "acetylcholine": 1.0, "glutamate": 0.8, "gaba": -1.0, "histamine": -1.0,
    "serotonin": 0.5, "octopamine": 0.4, "dopamine": 0.4, "tyramine": 0.4,
    "unclear": 0.0,
}

# raw connectome / annotation cache: loading 43 MB of feathers takes ~0.4 s,
# which matters when many simulations (experiments, evolution) run per process.
_RAW_CACHE: dict = {}


def _load_raw(data_dir: Path, files: dict):
    """Load connectome / annotation / neurotransmitter tables (cached).

    A small derived file `neurotransmitters_mini.feather` (only the 2594 mini
    neurons) is preferred when present, so a fresh clone only needs ~0.5 MB of
    data instead of the 40 MB full prediction table. The full FlyWire tables are
    still supported: put them in `data/` and they are used automatically.
    """
    mini_nt = "neurotransmitters_mini.feather"
    nt_name = mini_nt if (data_dir / mini_nt).exists() else files["neurotransmitters"]
    key = (str(data_dir), str(files["connectome"]), str(files["neurons"]), str(nt_name))
    if key not in _RAW_CACHE:
        mat = np.load(data_dir / files["connectome"], allow_pickle=True)
        W = sparse.csr_matrix(
            (mat["data"].astype(np.float32), mat["indices"], mat["indptr"]),
            shape=(int(mat["shape"][0]), int(mat["shape"][1])))
        ids = np.load(data_dir / "selected_body_ids_mini.npy")
        ann = (pd.read_feather(data_dir / files["neurons"])
               .set_index("bodyId").reindex(ids))
        nt = (pd.read_feather(data_dir / nt_name,
                              columns=["body", "consensus_nt", "predicted_nt", "predicted_nt_confidence"])
              .set_index("body").reindex(ids))
        _RAW_CACHE[key] = (W, ids, ann, nt)
    return _RAW_CACHE[key]


class ConnectomeNet:
    def __init__(self, cfg: dict, data_dir: Path, rng: np.random.Generator | None = None,
                 seed: int | None = None):
        self.cfg = cfg
        self.data_dir = Path(data_dir)
        nc = cfg["neural"]
        self.nc = nc
        self.seed = int(seed if seed is not None else nc.get("seed", 7))
        if rng is None:
            rng = np.random.default_rng(self.seed)
        dt = float(cfg["time"]["neural_dt_ms"])
        self.dt = dt
        self.steps_per_tick = int(cfg["time"]["neural_steps_per_tick"])

        W, ids, ann, nt = _load_raw(data_dir, nc["files"])
        self.ids = ids
        self.ann = ann
        n = self.n = W.shape[0]
        assert W.shape[0] == W.shape[1] == len(ids) == ann.shape[0], "connectome / annotation mismatch"

        # --- signed, compressed weights ------------------------------------
        nt_name = nt["consensus_nt"].fillna("unclear").astype(str).values
        conf = nt["predicted_nt_confidence"].fillna(0.5).astype(float).values
        sign = np.array([NT_SIGN.get(x, 0.0) for x in nt_name], dtype=np.float32)
        sign *= np.clip(0.6 + 0.4 * conf, 0.6, 1.0).astype(np.float32)
        pre_idx = np.repeat(np.arange(n, dtype=np.int64), np.diff(W.indptr).astype(np.int64))
        data = np.power(W.data.astype(np.float32), float(nc["synapse_power"]))
        data *= sign[pre_idx]
        Ws = sparse.csr_matrix((data.astype(np.float32), W.indices, W.indptr), shape=W.shape)

        tot = np.abs(Ws).sum(axis=1).A1
        nz = tot[tot > 0]
        med = float(np.median(nz)) if nz.size else 1.0
        Ws.data *= np.float32(float(nc["syn_gain"]) / max(med, 1e-6))

        pos = float(np.maximum(Ws.data, 0.0).sum())
        neg = float(-np.minimum(Ws.data, 0.0).sum())
        ratio = float(nc.get("inhib_target_ratio", 0.0))
        if ratio > 0.0 and neg > 0.0:
            target = ratio / (1.0 - ratio) * pos
            f = float(np.clip(target / neg, 0.1, 10.0))
            Ws.data = np.where(Ws.data < 0.0, Ws.data * f, Ws.data).astype(np.float32)
        self.W = Ws
        self.WT = Ws.transpose().tocsr()
        self.ei_ratio = float(-np.minimum(Ws.data, 0.0).sum() /
                              max(np.abs(Ws.data).sum(), 1e-6))
        self.n_synapses = int(W.nnz)

        # --- neuron groups --------------------------------------------------
        sc = ann["superclass"].astype(str).values
        dn = np.flatnonzero(sc == "descending_neuron")
        perm = rng.permutation(dn)
        n_in = int(min(int(nc["n_input"]), len(perm) // 3))
        self.input_idx = np.sort(perm[:n_in])
        rest = perm[n_in:]
        g = int(nc["n_motor_groups"])
        self.n_groups = g
        self.motor_idx = [np.sort(rest[i::g]) for i in range(g)]
        self.dn_idx = np.sort(dn)
        self.n_input = int(n_in)

        key = (ann["superclass"].astype(str) + "|" + ann["type"].astype(str) + "|"
               + ann["instance"].astype(str))
        self.display_order = np.argsort(key.values, kind="stable")

        # --- encoder --------------------------------------------------------
        f = int(nc["n_features"])
        r = rng.normal(0.0, 1.0, size=(f, n_in)).astype(np.float32)
        r *= np.float32(float(nc["enc_gain"]) / np.sqrt(f))
        self.enc = r
        self._iext = np.zeros(n, dtype=np.float32)

        # --- LIF state ------------------------------------------------------
        L = nc["lif"]
        self.tau = float(L["tau_m"])
        self.v_rest = float(L["v_rest"])
        self.v_reset = float(L["v_reset"])
        self.v_thresh = float(L["v_thresh"])
        self.noise_sigma = float(L["noise_sigma"])
        self.noise_scale = np.float32(self.noise_sigma * np.sqrt(3.0))  # uniform, same variance
        self.ref_steps = int(max(1, round(float(L["t_refract"]) / dt)))
        self.v = np.full(n, self.v_rest, dtype=np.float32)
        self.ref = np.zeros(n, dtype=np.int16)
        self.spk = np.zeros(n, dtype=np.float32)
        self.rate = np.zeros(n, dtype=np.float32)
        self.alpha_rate = np.float32(dt / float(nc["rate_tau"]))
        self.gain = 1.0
        h = nc["homeostasis"]
        self.h_target = float(h["target_hz"])
        self.h_lr = float(h["lr"])
        self.gain_min = float(h["gain_min"])
        self.gain_max = float(h["gain_max"])
        self.h_enabled = bool(h["enabled"])
        self.t = 0

        # --- plasticity -----------------------------------------------------
        pl = nc.get("plasticity", {})
        s = pl.get("stdp", {})
        nm = pl.get("neuromod", {})
        ro = pl.get("readout", {})
        self.stdp_cfg, self.nm_cfg, self.ro_cfg = s, nm, ro
        self.stdp_on = bool(s.get("enabled", False))
        self.neuromod_on = bool(nm.get("enabled", False))
        self.readout_on = bool(ro.get("enabled", False))
        self.exc_only = bool(s.get("exc_only", True))
        self.a_plus = float(s.get("a_plus", 0.05))
        self.a_minus = float(s.get("a_minus", 0.06))
        self.trace_decay = float(np.exp(-dt / float(s.get("tau_trace", 20.0))))
        self.elig = np.zeros(self.n_synapses, dtype=np.float32)
        self.x_pre = np.zeros(n, dtype=np.float32)
        self.y_post = np.zeros(n, dtype=np.float32)
        self.w0 = self.W.data.copy()      # lineage baseline (forgetting target)
        self.w_ref = self.w0.copy()       # pristine weights (hard clamp bounds)
        self.exc_mask = self.w0 > 0.0
        self.w_cap = float(s.get("w_cap_mult", 3.0))
        self.w_floor = float(s.get("w_floor", 0.02))
        self._refresh_bounds()
        self.forget = float(s.get("forget", 2e-5))
        self.sync_every = int(max(1, pl.get("sync_every_ticks", 2)))
        self._sync_count = 0
        self._weights_dirty = False
        sh = s.get("syn_homeostasis", {})
        self.syn_home_on = bool(sh.get("enabled", True))
        self.syn_home_every = int(max(1, sh.get("every_ticks", 5)))
        self.syn_home_rate = float(sh.get("rate", 0.05))
        self.syn_home_band = (float(sh.get("band_low", 0.8)), float(sh.get("band_high", 1.25)))
        self.pre_of_csr = np.repeat(np.arange(n, dtype=np.int64), np.diff(self.W.indptr).astype(np.int64))
        self.row_target = np.bincount(self.pre_of_csr, weights=np.maximum(self.w0, 0.0),
                                      minlength=n).astype(np.float32)
        self.elig_tau = float(s.get("elig_tau", 900.0))
        self.elig_decay_tick = float(np.exp(-(dt * self.steps_per_tick) / self.elig_tau))
        self.stdp_stride = int(max(1, s.get("stride_ms", 2)))
        self.trace_decay_stride = float(np.exp(-dt * self.stdp_stride / float(s.get("tau_trace", 20.0))))
        self.a_plus_stride = self.a_plus * self.stdp_stride
        self.a_minus_stride = self.a_minus * self.stdp_stride
        self._sub = 0
        self._fire_acc = np.zeros(n, dtype=bool)
        # scratch buffers for the hot plasticity path
        self._dw = np.empty(self.n_synapses, dtype=np.float32)
        self._plastic_mask = self.exc_mask.astype(np.float32) if self.exc_only else None
        self._elig_count = 0
        self.elig_decay_interval = int(max(1, pl.get("elig_decay_every_ticks", 2)))
        self.elig_decay_step = float(np.exp(
            -(dt * self.steps_per_tick * self.elig_decay_interval) / self.elig_tau))
        self.lr = float(nm.get("lr", 0.35))
        self.da_clip = float(nm.get("clip", 1.0))
        self.da = 0.0
        self.da_ema = 0.0
        self.n_plastic_updates = 0
        self.last_z = np.zeros(g, dtype=np.float32)
        self.readout = np.eye(g, dtype=np.float32)
        self.rl_lr = float(ro.get("lr", 0.02))
        self.rl_decay = float(ro.get("decay", 5e-4))
        self.rl_clip = float(ro.get("clip", 2.5))

        # transposed-data index map (WT layout -> W layout), used by STDP and cloning
        pre_of = np.repeat(np.arange(n, dtype=np.int64), np.diff(self.W.indptr).astype(np.int64))
        key_csr = pre_of * n + self.W.indices.astype(np.int64)          # (pre, post), sorted
        pre_of_t = np.repeat(np.arange(n, dtype=np.int64), np.diff(self.WT.indptr).astype(np.int64))
        # WT row index is the postsynaptic neuron, its column index is presynaptic
        key_t = self.WT.indices.astype(np.int64) * n + pre_of_t          # (pre, post) of WT entry
        self.csc_to_csr = np.searchsorted(key_csr, key_t).astype(np.int64)
        assert np.all(key_csr[self.csc_to_csr] == key_t), "connectome transpose mapping failed"
        assert np.allclose(self.WT.data, self.W.data[self.csc_to_csr]), "transpose data map failed"

    # --- sensory encoding ---------------------------------------------------
    def encode(self, features: np.ndarray) -> np.ndarray:
        self._iext.fill(0.0)
        self._iext[self.input_idx] = self.enc.T @ features.astype(np.float32)
        return self._iext

    # --- dynamics -----------------------------------------------------------
    def step(self, n_steps: int, iext: np.ndarray | None, rng: np.random.Generator) -> None:
        leak = np.float32(self.dt / self.tau)
        for _ in range(n_steps):
            syn = self.WT @ self.spk
            drive = syn * np.float32(self.gain)
            if iext is not None:
                drive = drive + iext
            noise = (rng.random(self.n, dtype=np.float32) * 2.0 - 1.0) * self.noise_scale
            self.v += leak * (-(self.v - self.v_rest) + drive) + noise
            np.clip(self.v, -95.0, -20.0, out=self.v)
            self.ref -= 1
            np.maximum(self.ref, 0, out=self.ref)
            fire = (self.v >= self.v_thresh) & (self.ref == 0)
            if fire.any():
                self.v[fire] = self.v_reset
                self.ref[fire] = self.ref_steps
                self.spk[:] = 0.0
                self.spk[fire] = 1.0
            else:
                self.spk[:] = 0.0
            self.rate += self.alpha_rate * (self.spk * 1000.0 - self.rate)
            if self.stdp_on:
                np.logical_or(self._fire_acc, fire, out=self._fire_acc)
                self._sub += 1
                if self._sub >= self.stdp_stride:
                    self._sub = 0
                    self._eligibility_step(self._fire_acc)

    def _eligibility_step(self, fire: np.ndarray) -> None:
        """Event-driven STDP eligibility traces, applied every `stride_ms`.

        The spike mask passed in accumulates all spikes of the stride window and
        the STDP amplitudes are scaled by the stride, which keeps the update
        equivalent while cutting the per-millisecond numpy overhead.
        """
        self.x_pre *= self.trace_decay_stride
        self.y_post *= self.trace_decay_stride
        if not fire.any():
            return
        idx = np.flatnonzero(fire)
        self.x_pre[idx] += 1.0
        self.y_post[idx] += 1.0
        # outgoing (pre spike -> depression), CSR order
        m = self._flat_slices(self.W.indptr, idx)
        if m.size:
            self.elig[m] -= self.a_minus_stride * self.y_post[self.W.indices[m]]
        # incoming (post spike -> potentiation), WT order mapped to CSR layout
        m = self._flat_slices(self.WT.indptr, idx)
        if m.size:
            self.elig[self.csc_to_csr[m]] += self.a_plus_stride * self.x_pre[self.WT.indices[m]]
        fire[:] = False

    @staticmethod
    def _flat_slices(indptr: np.ndarray, neurons: np.ndarray) -> np.ndarray:
        """Concatenated row-slice indices for a set of neurons, vectorised."""
        starts = indptr[neurons]
        counts = indptr[neurons + 1] - starts
        total = int(counts.sum())
        if total == 0:
            return np.empty(0, dtype=np.int64)
        base = np.repeat(starts, counts)
        offs = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts)
        return base + offs

    def _refresh_bounds(self) -> None:
        exc = self.w_ref > 0.0
        if self.exc_only:
            self.w_lo = np.where(exc, self.w_floor, self.w_ref)
            self.w_hi = np.where(exc, self.w_cap * self.w_ref, self.w_ref)
        else:
            self.w_lo = np.where(exc, self.w_floor, self.w_cap * np.minimum(self.w_ref, 0.0))
            self.w_hi = np.where(exc, self.w_cap * self.w_ref, -self.w_floor)
        self.w_lo = self.w_lo.astype(np.float32)
        self.w_hi = self.w_hi.astype(np.float32)

    def clone_with_mutation(self, rng: np.random.Generator, rate: float = 0.004,
                            sigma: float = 0.10, readout_sigma: float = 0.10):
        """Inherit weights/read-out from this brain, then mutate (neuroevolution)."""
        child = ConnectomeNet(self.cfg, self.data_dir, seed=self.seed)
        child.W.data[:] = self.W.data
        child.WT.data[:] = child.W.data[child.csc_to_csr]
        child.w0 = child.W.data.copy()
        child._refresh_bounds()
        child.readout[:, :] = self.readout + readout_sigma * rng.standard_normal(child.readout.shape)
        np.clip(child.readout, -child.rl_clip, child.rl_clip, out=child.readout)
        nnz = child.W.data.size
        k = int(rate * nnz)
        if k > 0:
            idx = rng.choice(nnz, size=k, replace=False)
            child.W.data[idx] *= (1.0 + sigma * rng.standard_normal(k)).astype(np.float32)
        np.clip(child.W.data, child.w_lo, child.w_hi, out=child.W.data)
        child.WT.data[:] = child.W.data[child.csc_to_csr]
        child.elig[:] = 0.0
        child.x_pre[:] = 0.0
        child.y_post[:] = 0.0
        return child

    def _synaptic_scaling(self, w: np.ndarray) -> None:
        """Light homeostatic scaling: keep each neuron's excitatory input budget.

        Redistribution is allowed (STDP can still change individual synapses),
        but the total positive weight per presynaptic neuron is slowly pulled
        back to its initial value, which stops runaway potentiation of the
        reservoir and keeps the decode read-out meaningful.
        """
        pos = np.maximum(w, 0.0)
        cur = np.bincount(self.pre_of_csr, weights=pos, minlength=self.n)
        ratio = self.row_target / np.maximum(cur, 1e-6)
        lo, hi = self.syn_home_band
        ratio = np.clip(ratio, lo, hi)
        factor = (1.0 + self.syn_home_rate * (ratio - 1.0)).astype(np.float32)
        mask = w > 0.0
        w[mask] *= factor[self.pre_of_csr[mask]]

    def _z_scores(self) -> np.ndarray:
        r = np.array([float(self.rate[i].mean()) for i in self.motor_idx], dtype=np.float32)
        return (r - r.mean()) / (r.std() + 1e-3)

    def motor_scores(self) -> np.ndarray:
        """Pure read-out (no side effects); used by rendering/logging."""
        return self.readout @ self._z_scores()

    def scores_for_decision(self) -> np.ndarray:
        """Read-out used when acting; remembers z for the plastic decode layer."""
        z = self._z_scores()
        self.last_z = z
        return self.readout @ z

    def tick_update(self, da: float = 0.0, action: int | None = None) -> None:
        """Homeostasis + reward-modulated plasticity, once per game tick."""
        if self.h_enabled:
            m = float(self.rate[self.dn_idx].mean())
            err = (self.h_target - m) / max(self.h_target, 1e-6)
            self.gain *= float(np.exp(np.clip(self.h_lr * err, -0.08, 0.08)))
            self.gain = float(np.clip(self.gain, self.gain_min, self.gain_max))

        da = float(np.clip(da, -self.da_clip, self.da_clip))
        self.da = da
        self.da_ema = 0.9 * self.da_ema + 0.1 * da

        if self.stdp_on:
            w = self.W.data
            if self.neuromod_on and abs(da) > 1e-4:
                np.multiply(self.elig, np.float32(self.lr * da), out=self._dw)
                if self._plastic_mask is not None:
                    self._dw *= self._plastic_mask
                w += self._dw
                self.n_plastic_updates += 1
                self._weights_dirty = True
            if self.forget > 0.0:
                tmp = self.w0 - w
                tmp *= np.float32(self.forget)
                w += tmp
                self._weights_dirty = True
            if self.syn_home_on and (self.t % self.syn_home_every == 0):
                self._synaptic_scaling(w)
                self._weights_dirty = True
            self._sync_count += 1
            if self._weights_dirty and self._sync_count >= self.sync_every:
                np.clip(w, self.w_lo, self.w_hi, out=w)
                self.WT.data[:] = w[self.csc_to_csr]
                self._sync_count = 0
                self._weights_dirty = False
            self._elig_count += 1
            if self._elig_count >= self.elig_decay_interval:
                self._elig_count = 0
                self.elig *= self.elig_decay_step

        if self.readout_on:
            if self.neuromod_on and action is not None and abs(da) > 1e-4:
                self.readout[:, int(action)] += self.rl_lr * da * self.last_z
            self.readout += self.rl_decay * (np.eye(self.n_groups, dtype=np.float32) - self.readout)
            np.clip(self.readout, -self.rl_clip, self.rl_clip, out=self.readout)

        self.t += 1

    # --- readouts -----------------------------------------------------------
    def motor_rates_hz(self) -> np.ndarray:
        return np.array([float(self.rate[idx].mean()) for idx in self.motor_idx], dtype=np.float32)

    def mean_dn_hz(self) -> float:
        return float(self.rate[self.dn_idx].mean())

    def activity_grid(self, rows: int, cols: int) -> np.ndarray:
        vals = self.rate[self.display_order]
        grid = np.zeros(rows * cols, dtype=np.float32)
        grid[:min(vals.size, grid.size)] = np.clip(vals[:grid.size], 0.0, 60.0)
        return grid.reshape(rows, cols)

    def learning_stats(self, sample: int = 16) -> dict:
        s = max(1, int(sample))
        w = self.W.data[::s]
        w0 = self.w0[::s]
        rel = float(np.mean(np.abs(w - w0)) / max(float(np.mean(np.abs(w0))), 1e-6))
        return {"da": self.da, "da_ema": self.da_ema,
                "elig": float(np.mean(np.abs(self.elig[::s]))) if self.stdp_on else 0.0,
                "w_change": rel,
                "readout_drift": float(np.mean(np.abs(self.readout - np.eye(self.n_groups, dtype=np.float32)))),
                "plastic_updates": self.n_plastic_updates,
                "stdp": self.stdp_on, "readout": self.readout_on}

    def summary(self) -> dict:
        return {"neurons": self.n, "synapses": self.n_synapses, "inputs": self.n_input,
                "motor_groups": [len(i) for i in self.motor_idx],
                "dn": int(self.dn_idx.size), "ei_ratio": round(self.ei_ratio, 3),
                "gain": round(self.gain, 3),
                "stdp": self.stdp_on, "neuromod": self.neuromod_on,
                "readout_plastic": self.readout_on}