"""Build the flow-feature datasets from the synthetic generator.

One generation pass produces several *views* of exactly the same flows:

* ``flows.csv``                 – full-window features (main experiment)
* ``views/prefix_<N>.csv``      – features of the first N packets (early classification)
* ``views/defense_<name>.csv``  – features after a countermeasure is applied
* ``flows_cellular_*.csv``      – flows drawn under a shifted network profile

Packet traces themselves are discarded after feature extraction (they can
total tens of millions of packets); a handful are kept in
``sample_traces.npz`` for plotting.  Every class gets its own RNG stream
derived from the master seed, so results are bit-for-bit reproducible.
"""
from __future__ import annotations

import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from . import config as C
from .defenses import DEFENSES
from .features import extract_features
from .traffic_generator import generate_flow

META_COLS = ["flow_id", "label", "subtype", "rtt", "cap_down"]


def load_csv(path) -> pd.DataFrame:
    """Read a feature CSV with *exact* (round-trip) float parsing.

    pandas < 3.1 parses floats with a fast parser that can be off by one
    ulp; such differences are enough to move a gradient-boosting split
    threshold and change a few predictions.  Requesting round-trip parsing
    makes results bit-identical across pandas versions (pandas >= 3.1 does
    this by default and deprecates the flag, hence the warning filter)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.read_csv(path, float_precision="round_trip")


def _class_job(args):
    """Generate all flows of one class and compute every view (runs in a worker)."""
    label, n_flows, seed, profile, with_views = args
    rng = np.random.default_rng(seed)
    def_rng = np.random.default_rng(seed + 1)  # separate stream for defences
    main, prefixes, defended, samples = [], {k: [] for k in C.EARLY_PACKET_COUNTS}, {k: [] for k in DEFENSES}, []
    for i in range(n_flows):
        t, s, d, sub, net = generate_flow(label, rng, profile)
        meta = {"flow_id": f"{profile.name}-{label}-{i:05d}", "label": label, "subtype": sub,
                "rtt": net.rtt, "cap_down": net.cap_down}
        main.append({**meta, **extract_features(t, s, d)})
        if not with_views:
            continue
        for k in C.EARLY_PACKET_COUNTS:
            m = min(k, t.size)
            prefixes[k].append({**meta, **extract_features(t[:m], s[:m], d[:m])})
        for name, fn in DEFENSES.items():
            t2, s2, d2, delay = fn(t, s, d, def_rng)
            row = {**meta, **extract_features(t2, s2, d2), "orig_bytes": float(s.sum()), "mean_delay_s": delay}
            defended[name].append(row)
        if i < 3:  # keep a few raw traces for illustration
            samples.append((label, sub, t, s, d))
    return label, main, prefixes, defended, samples


def _seeds(profile_name: str) -> dict[str, int]:
    """Independent, reproducible seed per (profile, class)."""
    base = np.random.SeedSequence([C.SEED, sum(map(ord, profile_name))])
    children = base.spawn(len(C.CLASSES))
    return {c: int(ch.generate_state(1)[0]) for c, ch in zip(C.CLASSES, children)}


def build_datasets(workers: int = 4, verbose: bool = True) -> None:
    """Generate every dataset/view and write them under ``data/``."""
    t0 = time.time()
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    (C.DATA_DIR / "views").mkdir(exist_ok=True)

    # ---- main (broadband) dataset with all views ----------------------------
    seeds = _seeds(C.BROADBAND.name)
    jobs = [(c, C.FLOWS_PER_CLASS[c], seeds[c], C.BROADBAND, True) for c in C.CLASSES]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(_class_job, jobs))
    main = pd.DataFrame([r for o in out for r in o[1]])
    main.to_csv(C.DATA_DIR / "flows.csv", index=False)
    for k in C.EARLY_PACKET_COUNTS:
        pd.DataFrame([r for o in out for r in o[2][k]]).to_csv(C.DATA_DIR / "views" / f"prefix_{k}.csv", index=False)
    for name in DEFENSES:
        pd.DataFrame([r for o in out for r in o[3][name]]).to_csv(
            C.DATA_DIR / "views" / f"defense_{name}.csv", index=False)
    samples = [smp for o in out for smp in o[4]]
    np.savez_compressed(
        C.DATA_DIR / "sample_traces.npz",
        **{f"{lab}__{j}__{sub}__{arr}": a for j, (lab, sub, t, s, d) in enumerate(samples)
           for arr, a in (("t", t), ("s", s), ("d", d))},
    )
    if verbose:
        print(f"[data] broadband: {len(main)} flows in {time.time() - t0:.1f}s")

    # ---- domain-shift (cellular) sets: one for augmentation, one for test --
    seeds = _seeds(C.CELLULAR.name)
    for split, off in (("train", 0), ("test", 7919)):
        jobs = [(c, C.SHIFT_FLOWS_PER_CLASS, seeds[c] + off, C.CELLULAR, False) for c in C.CLASSES]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            out = list(ex.map(_class_job, jobs))
        df = pd.DataFrame([r for o in out for r in o[1]])
        df["flow_id"] = df["flow_id"].str.replace("cellular", f"cellular_{split}")
        df.to_csv(C.DATA_DIR / f"flows_cellular_{split}.csv", index=False)
    if verbose:
        print(f"[data] all datasets written in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    build_datasets()
