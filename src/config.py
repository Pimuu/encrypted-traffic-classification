"""Central configuration for the encrypted-traffic classification pipeline.

Every tunable constant lives here so that an experiment can be reproduced
exactly by re-running ``python -m src.run_pipeline`` with the same values.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
SEED = 42  # master seed; every random stream is derived from it

# --------------------------------------------------------------------------
# Traffic classes and how many flows of each to synthesise.
# The mix is deliberately unbalanced (web is the most common class on real
# networks) so that macro-averaged metrics matter.
# --------------------------------------------------------------------------
CLASSES = ["video", "web", "chat", "file_transfer", "voip", "gaming"]
FLOWS_PER_CLASS = {
    "video": 900,
    "web": 1400,
    "chat": 900,
    "file_transfer": 700,
    "voip": 700,
    "gaming": 700,
}

# Observation window, analogous to the "active timeout" of a NetFlow/IPFIX
# exporter: a flow record is emitted after at most this many seconds.
WINDOW_S = 60.0

# Packet-size constants (bytes, on-the-wire IP packet length)
MTU = 1500
MIN_PKT = 40

# --------------------------------------------------------------------------
# Network conditions. Each flow draws its own RTT / capacity / jitter / loss
# around these medians, so the classifier cannot key on one fixed path.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class NetworkProfile:
    name: str
    rtt_median_s: float  # round-trip time
    down_capacity_median_bps: float  # bottleneck capacity, server->client
    up_down_ratio: float  # uplink capacity as a fraction of downlink
    jitter_s: float  # std-dev of per-packet network delay variation
    loss: float  # probability that a packet triggers a loss-recovery stall


# Default: fixed broadband / campus network
BROADBAND = NetworkProfile("broadband", 0.035, 30e6, 0.25, 0.002, 0.003)
# Domain-shift profile used only for the robustness experiment:
# congested cellular access (higher RTT, lower capacity, more jitter / loss)
CELLULAR = NetworkProfile("cellular", 0.110, 6e6, 0.30, 0.012, 0.02)

# --------------------------------------------------------------------------
# Experiment settings
# --------------------------------------------------------------------------
TEST_SIZE = 0.30  # stratified hold-out fraction
CV_FOLDS = 5  # stratified K-fold on the training portion
EARLY_PACKET_COUNTS = [3, 5, 10, 20, 50, 100, 200]  # prefixes for early classification
SHIFT_FLOWS_PER_CLASS = 200  # size of each cellular split (train-aug / test)

# Countermeasures evaluated in the robustness study
JITTER_MAX_S = 0.050  # uniform per-packet delay added by the timing defence
BUFLO_RHO_S = 0.005  # constant-rate defence: one MTU cell every 5 ms per direction (2.4 Mbit/s)
BUFLO_MIN_S = 10.0  # ... for at least 10 s
