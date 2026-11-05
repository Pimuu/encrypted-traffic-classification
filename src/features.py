"""Flow-level feature extraction from packet metadata only.

Input is a packet trace ``(t, s, d)`` (timestamps, IP lengths, directions);
output is a fixed-length vector of statistics in the spirit of
NetFlow/IPFIX/CICFlowMeter records.  No payload is ever touched.

Features are organised into four groups so they can be ablated:

* ``volume``    – how much and how fast (counts, bytes, duration, rates)
* ``size``      – packet-length distribution, overall and per direction
* ``timing``    – inter-arrival times, burstiness, idleness, periodicity
* ``direction`` – asymmetry and turn-taking between up- and downstream
"""
from __future__ import annotations

import numpy as np

SMALL_PKT = 128  # bytes: "control-sized" packet (ACK, keep-alive, input)
LARGE_PKT = 1200  # bytes: "data-sized" packet (near MTU)
BURST_GAP_S = 0.5  # gap that separates two bursts
IDLE_GAP_S = 1.0  # gap counted as idle time
REGULARITY_TOL = 0.15  # |iat - median| <= tol * median counts as "on the beat"


def _stats(x: np.ndarray):
    """mean, std, max of an array; zeros for an empty array."""
    if x.size == 0:
        return 0.0, 0.0, 0.0
    return float(x.mean()), float(x.std()), float(x.max())


def _iat_stats(t: np.ndarray):
    """mean, std, coefficient of variation and regularity of inter-arrivals.

    *Regularity* is the share of inter-arrival times within ±15 % of the
    median one – close to 1 for constant-rate senders (VoIP, game ticks)."""
    if t.size < 2:
        return 0.0, 0.0, 0.0, 0.0
    iat = np.diff(t)
    m, sd = float(iat.mean()), float(iat.std())
    med = float(np.median(iat))
    reg = float(np.mean(np.abs(iat - med) <= REGULARITY_TOL * med)) if med > 0 else 0.0
    return m, sd, sd / m if m > 0 else 0.0, reg


FEATURE_GROUPS: dict[str, list[str]] = {
    "volume": [
        "n_pkts", "n_bytes", "fwd_pkts", "bwd_pkts", "fwd_bytes", "bwd_bytes",
        "duration", "pkt_rate", "byte_rate",
    ],
    "size": [
        "size_mean", "size_std", "size_min", "size_max", "size_q25", "size_q50", "size_q75",
        "fwd_size_mean", "fwd_size_std", "fwd_size_max",
        "bwd_size_mean", "bwd_size_std", "bwd_size_max",
        "frac_small", "frac_large",
    ],
    "timing": [
        "iat_mean", "iat_std", "iat_cv", "iat_median", "iat_max", "iat_regularity",
        "fwd_iat_mean", "fwd_iat_std", "fwd_iat_cv", "fwd_iat_regularity",
        "bwd_iat_mean", "bwd_iat_std", "bwd_iat_cv", "bwd_iat_regularity",
        "burst_rate", "burst_bytes_mean", "burst_bytes_cv", "idle_frac",
    ],
    "direction": [
        "bwd_fwd_byte_ratio", "bwd_pkt_frac", "dir_change_rate",
    ],
}
FEATURE_NAMES: list[str] = [f for g in FEATURE_GROUPS.values() for f in g]


def extract_features(t: np.ndarray, s: np.ndarray, d: np.ndarray) -> dict[str, float]:
    """Compute the full feature vector for one flow (arrays sorted by time)."""
    n = int(t.size)
    s = s.astype(np.float64)
    fwd, bwd = d == 0, d == 1
    sf, sb = s[fwd], s[bwd]
    duration = float(t[-1] - t[0]) if n > 1 else 0.0
    dur_eff = max(duration, 1e-3)  # avoid division by zero for 1-instant flows
    f: dict[str, float] = {}

    # ---- volume ----------------------------------------------------------
    f["n_pkts"] = n
    f["n_bytes"] = float(s.sum())
    f["fwd_pkts"] = int(fwd.sum())
    f["bwd_pkts"] = int(bwd.sum())
    f["fwd_bytes"] = float(sf.sum())
    f["bwd_bytes"] = float(sb.sum())
    f["duration"] = duration
    f["pkt_rate"] = n / dur_eff
    f["byte_rate"] = f["n_bytes"] / dur_eff

    # ---- size ------------------------------------------------------------
    f["size_mean"], f["size_std"], f["size_max"] = _stats(s)
    f["size_min"] = float(s.min())
    f["size_q25"], f["size_q50"], f["size_q75"] = (float(q) for q in np.percentile(s, [25, 50, 75]))
    f["fwd_size_mean"], f["fwd_size_std"], f["fwd_size_max"] = _stats(sf)
    f["bwd_size_mean"], f["bwd_size_std"], f["bwd_size_max"] = _stats(sb)
    f["frac_small"] = float(np.mean(s <= SMALL_PKT))
    f["frac_large"] = float(np.mean(s >= LARGE_PKT))

    # ---- timing ----------------------------------------------------------
    f["iat_mean"], f["iat_std"], f["iat_cv"], f["iat_regularity"] = _iat_stats(t)
    iat = np.diff(t) if n > 1 else np.zeros(1)
    f["iat_median"] = float(np.median(iat))
    f["iat_max"] = float(iat.max())
    f["fwd_iat_mean"], f["fwd_iat_std"], f["fwd_iat_cv"], f["fwd_iat_regularity"] = _iat_stats(t[fwd])
    f["bwd_iat_mean"], f["bwd_iat_std"], f["bwd_iat_cv"], f["bwd_iat_regularity"] = _iat_stats(t[bwd])
    # bursts: maximal runs of packets separated by gaps < BURST_GAP_S
    gaps = np.diff(t) if n > 1 else np.zeros(0)
    cut = np.flatnonzero(gaps > BURST_GAP_S) + 1
    burst_bytes = np.add.reduceat(s, np.r_[0, cut]) if n else np.zeros(1)
    f["burst_rate"] = burst_bytes.size / dur_eff
    bm, bs, _ = _stats(burst_bytes)
    f["burst_bytes_mean"] = bm
    f["burst_bytes_cv"] = bs / bm if bm > 0 else 0.0
    f["idle_frac"] = float(gaps[gaps > IDLE_GAP_S].sum() / dur_eff) if gaps.size else 0.0

    # ---- direction -------------------------------------------------------
    f["bwd_fwd_byte_ratio"] = (f["bwd_bytes"] + 1.0) / (f["fwd_bytes"] + 1.0)
    f["bwd_pkt_frac"] = f["bwd_pkts"] / n
    f["dir_change_rate"] = float(np.count_nonzero(np.diff(d))) / max(n - 1, 1)

    return {k: f[k] for k in FEATURE_NAMES}  # stable column order
