"""Traffic-analysis countermeasures applied to packet traces.

They model what a privacy-conscious endpoint (or a VPN / Tor-like tunnel)
could do to hide its application.  We evaluate them against an *adaptive*
adversary that retrains on defended traffic (the realistic worst case for
the defender) and against a *non-adaptive* one that keeps its model trained
on undefended traffic.

Every defence returns ``(t, s, d, mean_added_delay_s)``.
"""
from __future__ import annotations

import numpy as np

from .config import BUFLO_MIN_S, BUFLO_RHO_S, JITTER_MAX_S, MTU, WINDOW_S


def pad_to_mtu(t, s, d, rng=None):
    """Pad every packet to the MTU (constant-size cells).  Removes all
    packet-length information; counts and timing are left intact."""
    return t, np.full_like(s, MTU), d, 0.0


def timing_jitter(t, s, d, rng):
    """Delay each packet by an independent U(0, JITTER_MAX_S) amount and
    re-sort.  Blurs fine-grained timing (ticks, ptime) at a latency cost."""
    delay = rng.uniform(0.0, JITTER_MAX_S, t.size)
    t2 = t + delay
    o = np.argsort(t2, kind="stable")
    t2 = t2[o]
    return t2 - t2[0], s[o], d[o], float(delay.mean())


def pad_and_jitter(t, s, d, rng):
    """Both of the above."""
    t, s, d, delay = timing_jitter(t, s, d, rng)
    t, s, d, _ = pad_to_mtu(t, s, d)
    return t, s, d, delay


def buflo(t, s, d, rng=None, rho: float = BUFLO_RHO_S, tau: float = BUFLO_MIN_S):
    """BuFLO-style constant-rate shaping (Dyer et al., 2012).

    In each direction the tunnel emits one MTU-sized cell every ``rho``
    seconds -- real data if queued, dummy otherwise -- for at least ``tau``
    seconds and until the real queue drains.  The observer therefore sees
    two identical constant-rate cell streams whose only free parameter is
    their length.  Real packets wait in a FIFO queue, so departures are
    ``dep_i = max(arr_i, dep_{i-1} + rho)``, computed in closed form with a
    running maximum.
    """
    ts, ss, ds, delays = [], [], [], []
    for dirn in (0, 1):
        arr = t[d == dirn]
        end = tau
        if arr.size:
            i = np.arange(arr.size)
            # departure on the cell grid: first slot >= the FIFO-constrained time
            dep = i * rho + np.maximum.accumulate(arr - i * rho)
            dep = np.ceil(dep / rho - 1e-9) * rho
            delays.append(dep - arr)
            end = max(tau, float(dep[-1]) + rho)
        cells = np.arange(0.0, min(end, WINDOW_S), rho) + (rho / 2 if dirn else 0.0)
        ts.append(cells)
        ss.append(np.full(cells.size, MTU, dtype=s.dtype))
        ds.append(np.full(cells.size, dirn, dtype=d.dtype))
    t2, s2, d2 = np.concatenate(ts), np.concatenate(ss), np.concatenate(ds)
    o = np.argsort(t2, kind="stable")
    delay = float(np.concatenate(delays).mean()) if delays else 0.0
    return t2[o], s2[o], d2[o], delay


DEFENSES = {
    "pad_mtu": pad_to_mtu,
    "jitter_50ms": timing_jitter,
    "pad_mtu+jitter_50ms": pad_and_jitter,
    "buflo": buflo,
}


def padding_overhead(s_orig: np.ndarray, s_def: np.ndarray) -> float:
    """Relative extra bytes added by the defence (0.0 = none)."""
    return float(s_def.sum() / s_orig.sum() - 1.0)
