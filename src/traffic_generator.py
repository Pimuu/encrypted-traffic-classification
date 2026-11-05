"""Synthetic packet-level traffic generator.

Each traffic class is modelled as a *generative process* that emits a packet
trace -- a sequence of ``(timestamp, ip_length, direction)`` tuples -- as an
on-path observer would see it for a single encrypted flow (TLS/TCP, QUIC or
SRTP/UDP).  Payload bytes are never produced: the observer only ever sees
lengths, timings and directions, which is exactly the threat model studied.

Conventions
-----------
* ``direction == 0``  -> upstream   (client -> server, "forward")
* ``direction == 1``  -> downstream (server -> client, "backward")
* sizes are IP packet lengths in bytes, clipped to ``[MIN_PKT, MTU]``
* the vantage point is a gateway next to the client, so upstream packets
  appear with the sender's own regularity while downstream packets carry
  network jitter.

The models are deliberately *stochastic and overlapping*: every class has
sub-types (live video, throttled downloads, video calls, ...) and per-flow
network conditions, so the classes are separable only statistically.  The
``subtype`` returned with each trace is kept for error analysis; it is never
used as a feature.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import MIN_PKT, MTU, WINDOW_S, NetworkProfile

TCP_ACK = (52, 66)  # pure TCP ACK sizes (with/without timestamps / SACK)


# ---------------------------------------------------------------------------
# Per-flow network sample
# ---------------------------------------------------------------------------
@dataclass
class FlowNet:
    """Network conditions drawn once per flow from a :class:`NetworkProfile`."""

    rtt: float
    cap_down: float
    cap_up: float
    jitter: float
    loss: float
    ack_every: int  # receiver ACK ratio: 2 = classic delayed ACK, 4/8 = GRO "stretch" ACKs

    @classmethod
    def sample(cls, rng: np.random.Generator, prof: NetworkProfile) -> "FlowNet":
        rtt = float(np.clip(rng.lognormal(np.log(prof.rtt_median_s), 0.5), 0.003, 0.8))
        cap = float(np.clip(rng.lognormal(np.log(prof.down_capacity_median_bps), 0.7), 1e6, 300e6))
        return cls(
            rtt=rtt,
            cap_down=cap,
            cap_up=cap * prof.up_down_ratio * float(rng.uniform(0.6, 1.4)),
            jitter=prof.jitter_s * float(rng.uniform(0.5, 2.0)),
            loss=prof.loss * float(rng.uniform(0.3, 2.0)),
            ack_every=int(rng.choice([1, 2, 2, 2, 3, 4, 8])),
        )


# ---------------------------------------------------------------------------
# Trace builder: accumulate packet "chunks" then sort / clip once
# ---------------------------------------------------------------------------
@dataclass
class Trace:
    t: list = field(default_factory=list)
    s: list = field(default_factory=list)
    d: list = field(default_factory=list)

    def add(self, times, sizes, direction) -> None:
        times, sizes, dirs = np.broadcast_arrays(
            np.atleast_1d(np.asarray(times, dtype=np.float64)),
            np.asarray(sizes, dtype=np.float64),
            np.asarray(direction, dtype=np.int8),
        )
        self.t.append(times)
        self.s.append(sizes)
        self.d.append(dirs)

    def finalize(self, start: float = 0.0, window: float = WINDOW_S):
        """Return time-sorted arrays restricted to ``[start, start+window)``,
        re-based so the first observed packet is at t=0."""
        t = np.concatenate(self.t)
        s = np.clip(np.rint(np.concatenate(self.s)), MIN_PKT, MTU).astype(np.int32)
        d = np.concatenate(self.d).astype(np.int8)
        keep = (t >= start) & (t < start + window)
        t, s, d = t[keep], s[keep], d[keep]
        order = np.argsort(t, kind="stable")
        t, s, d = t[order], s[order], d[order]
        if t.size:
            t = t - t[0]
        return t, s, d


# ---------------------------------------------------------------------------
# Building blocks shared by several classes
# ---------------------------------------------------------------------------
def quic_size(rng) -> int:
    """Max QUIC packet size differs by implementation / path-MTU probing."""
    return int(rng.choice([1252, 1350, 1392, 1452]))


def _ack_sizes(rng, n):
    return rng.integers(TCP_ACK[0], TCP_ACK[1] + 1, n)


def bulk_transfer(
    tr: Trace,
    rng: np.random.Generator,
    t0: float,
    nbytes: float,
    rate_bps: float,
    direction: int,
    net: FlowNet,
    pkt_size: int = MTU,
    transport: str = "tcp",
    slow_start: bool = True,
) -> float:
    """Emit a bulk transfer of ``nbytes`` and return its completion time.

    Packets are full-size (``pkt_size``) except the last one.  For
    ``transport`` "tcp" or "quic" the receiver acknowledges every
    ``net.ack_every``-th data packet in the opposite direction (TCP pure ACKs
    are 52-66 B; QUIC ACK-only packets 45-80 B); "udp" sends no ACKs.  ``slow_start`` reproduces the doubling
    congestion window of a fresh / idle-restarted connection: the first
    windows go out as line-rate trains separated by one RTT.
    """
    payload = pkt_size - 52
    n = max(1, int(np.ceil(nbytes / payload)))
    sizes = np.full(n, pkt_size, dtype=np.int32)
    sizes[-1] = 52 + max(1, int(nbytes - payload * (n - 1)))
    line = net.cap_down if direction == 1 else net.cap_up
    rate = max(min(rate_bps, line), 2e5)
    gap_line = pkt_size * 8.0 / line  # back-to-back spacing at line rate
    gap_rate = pkt_size * 8.0 / rate

    times = np.empty(n)
    i, t, cwnd = 0, t0, 10
    # slow-start phase: windows of cwnd packets, one RTT apart, until the
    # window would exceed the target rate (bandwidth-delay product)
    if slow_start:
        bdp_pkts = rate * net.rtt / (pkt_size * 8.0)
        while i < n and cwnd < bdp_pkts:
            k = min(cwnd, n - i)
            times[i : i + k] = t + np.arange(k) * gap_line
            i += k
            t += net.rtt
            cwnd *= 2
    # steady state: gamma-distributed spacing around the target rate
    if i < n:
        m = n - i
        iat = rng.gamma(3.0, gap_rate / 3.0, m)
        stall = rng.random(m) < net.loss  # loss recovery -> ~1-2 RTT pauses
        iat[stall] += net.rtt * rng.uniform(1.0, 2.0, stall.sum())
        times[i:] = t + np.cumsum(iat)
    # downstream packets cross the network -> add jitter (keep order)
    if direction == 1:
        times = np.maximum.accumulate(times + np.abs(rng.normal(0, net.jitter, n)))
    tr.add(times, sizes, direction)
    k = net.ack_every
    if transport != "udp" and n >= k:
        ack_t = times[k - 1 :: k]
        ack_t = ack_t + rng.uniform(5e-5, 5e-4, ack_t.size)
        if direction == 0:  # ACKs travel back across the network
            ack_t = ack_t + net.rtt
        sizes_ack = _ack_sizes(rng, ack_t.size) if transport == "tcp" else rng.integers(45, 81, ack_t.size)
        tr.add(ack_t, sizes_ack, 1 - direction)
    return float(times[-1])


def tls_handshake(tr: Trace, rng, t0: float, net: FlowNet, quic: bool = False) -> float:
    """TCP 3-way handshake + TLS 1.3 handshake (or QUIC initial exchange)."""
    t = t0
    if not quic:
        tr.add(t, 60, 0)  # SYN
        tr.add(t + net.rtt, 60, 1)  # SYN/ACK
        t += net.rtt
        tr.add(t + 1e-4, 52, 0)  # ACK
        ch = rng.integers(517, 660)  # ClientHello
    else:
        ch = rng.choice([1252, 1350])  # QUIC Initial is padded
    tr.add(t + 2e-4, ch, 0)
    t += net.rtt
    cert = rng.integers(2500, 6000)  # ServerHello + certificate chain
    pkt = quic_size(rng) if quic else MTU
    n = int(np.ceil(cert / (pkt - 52)))
    sizes = np.full(n, pkt)
    sizes[-1] = 52 + cert - (pkt - 52) * (n - 1)
    tr.add(t + np.arange(n) * 2e-4, sizes, 1)
    t += n * 2e-4 + 1e-3
    tr.add(t, rng.integers(80, 140), 0)  # client Finished
    tr.add(t + net.rtt, rng.integers(90, 320, rng.integers(1, 3)), 1)  # session tickets
    return t + 2e-3


def maybe_mid_stream(rng, p: float, max_offset: float) -> float:
    """Long-lived flows are often already running when observation starts:
    with probability ``p`` return a random start offset into the flow."""
    return float(rng.uniform(5.0, max_offset)) if rng.random() < p else 0.0


# ---------------------------------------------------------------------------
# Class generators.  Each returns (Trace, start_offset, subtype).
# ---------------------------------------------------------------------------
def gen_video(rng, net: FlowNet):
    """Adaptive (DASH/HLS-style) streaming over TLS or QUIC.

    A buffering phase fetches several segments back-to-back, then the player
    requests one segment every ``seg`` seconds -> periodic ON/OFF bursts of
    MTU-sized downstream packets with tiny upstream requests/ACKs.
    """
    tr = Trace()
    quic = rng.random() < 0.4
    u = rng.random()
    if u < 0.2:
        return _gen_short_form(rng, net, quic)
    live = u > 0.8
    bitrate = float(np.clip(rng.lognormal(np.log(2.5e6), 0.7), 0.25e6, 12e6))
    seg = float(rng.choice([1.0, 2.0])) if live else float(rng.choice([2.0, 4.0, 5.0, 6.0, 10.0]))
    n_buffer = 0 if live else int(rng.integers(2, 8))
    audio = rng.random() < 0.5  # separate audio track
    pkt = quic_size(rng) if quic else int(rng.choice([1500, 1500, 1460, 1420]))
    offset = maybe_mid_stream(rng, 0.35, 60.0)
    t = tls_handshake(tr, rng, 0.0, net, quic) if offset == 0 else 0.0
    horizon = offset + WINDOW_S
    # ~15 % of VoD sessions get paused: the player stops fetching and only
    # telemetry / keep-alive exchanges remain until (maybe) playback resumes
    paused = (not live) and rng.random() < 0.15
    pause_at = offset + rng.uniform(3.0, 45.0) if paused else np.inf
    k = 0
    while t < horizon:
        if t >= pause_at:
            resume = t + float(rng.lognormal(np.log(20.0), 0.8))
            beats = np.arange(t + rng.uniform(1, 5), min(resume, horizon), rng.uniform(5.0, 15.0))
            tr.add(beats, rng.integers(300, 900, beats.size), 0)
            tr.add(beats + net.rtt, rng.integers(90, 400, beats.size), 1)
            t, pause_at = resume, np.inf
            continue
        t_req = t
        tr.add(t_req, rng.integers(250, 700) if not quic else rng.integers(90, 350), 0)
        seg_bytes = bitrate * seg / 8.0 * rng.lognormal(0.0, 0.25)
        done = bulk_transfer(tr, rng, t_req + net.rtt, seg_bytes, net.cap_down * rng.uniform(0.5, 1.0),
                             1, net, pkt, transport="quic" if quic else "tcp")
        if audio:
            tr.add(done + 0.002, rng.integers(250, 500), 0)
            done = bulk_transfer(tr, rng, done + net.rtt, 128e3 * seg / 8.0, net.cap_down, 1, net, pkt,
                                 transport="quic" if quic else "tcp", slow_start=False)
        if k < n_buffer:
            t = done + rng.exponential(0.05)
        else:
            t = max(done + 0.01, t_req + seg * rng.uniform(0.85, 1.15))
        k += 1
    return tr, offset, "live" if live else ("vod_paused" if paused else "vod")


def _gen_short_form(rng, net: FlowNet, quic: bool):
    """Short-form video feed (swipe-to-next clips): each clip is fetched as
    one or two large progressive downloads, the next clip is prefetched,
    and the user swipes after a heavy-tailed watch time -> irregular large
    bursts, closer to web/file traffic than to periodic DASH."""
    tr = Trace()
    pkt = quic_size(rng) if quic else MTU
    offset = maybe_mid_stream(rng, 0.3, 60.0)
    t = tls_handshake(tr, rng, 0.0, net, quic) if offset == 0 else 0.0
    horizon = offset + WINDOW_S
    while t < horizon:
        tr.add(t, rng.integers(250, 700), 0)
        clip = float(np.clip(rng.lognormal(np.log(1.5e6), 0.6), 2e5, 8e6))
        done = bulk_transfer(tr, rng, t + net.rtt, clip, net.cap_down * rng.uniform(0.4, 1.0), 1, net, pkt,
                             transport="quic" if quic else "tcp")
        if rng.random() < 0.5:  # thumbnail / metadata fetches between clips
            tr.add(done + 0.01, rng.integers(200, 500), 0)
            bulk_transfer(tr, rng, done + 0.01 + net.rtt, rng.lognormal(np.log(20e3), 0.8), net.cap_down, 1, net,
                          pkt, transport="quic" if quic else "tcp", slow_start=False)
        t = done + float(np.clip(rng.lognormal(np.log(8.0), 0.8), 1.0, 60.0))  # watch time
    return tr, offset, "short_form"


def gen_web(rng, net: FlowNet):
    """Web browsing on one HTTPS connection: request/response exchanges of
    heavy-tailed object sizes, grouped into page loads separated by
    user think time, followed by an idle keep-alive and close."""
    tr = Trace()
    quic = rng.random() < 0.3
    pkt = quic_size(rng) if quic else MTU
    t = tls_handshake(tr, rng, 0.0, net, quic)
    if rng.random() < 0.15:
        return _web_polling(tr, rng, net, t, quic)
    n_pages = int(rng.geometric(0.45))
    media_heavy = rng.random() < 0.15  # image galleries, news sites with large media
    obj_median = 60e3 if media_heavy else 8e3
    embedded_video = rng.random() < 0.08  # page with an auto-playing embedded clip
    for page in range(n_pages):
        n_obj = 1 + int(rng.poisson(6))
        for j in range(n_obj):
            if t >= WINDOW_S:
                break
            req = rng.integers(350, 1300) if j == 0 else rng.integers(150, 700)
            tr.add(t, req, 0)
            if rng.random() < 0.05:  # form POST / upload
                bulk_transfer(tr, rng, t + 1e-3, rng.lognormal(np.log(5e3), 1.5), net.cap_up, 0, net, pkt,
                              transport="quic" if quic else "tcp")
            resp = float(np.clip(rng.lognormal(np.log(obj_median), 1.4), 200, 5e6))
            server_think = rng.exponential(0.03)
            done = bulk_transfer(tr, rng, t + net.rtt + server_think, resp, net.cap_down, 1, net, pkt,
                                 transport="quic" if quic else "tcp", slow_start=(j == 0))
            t = done + rng.exponential(0.03)
        if embedded_video and page == 0:
            br = float(rng.uniform(0.6e6, 3e6))
            for _ in range(int(rng.integers(3, 10))):
                tr.add(t, rng.integers(250, 600), 0)
                bulk_transfer(tr, rng, t + net.rtt, br * 2.0 / 8.0, net.cap_down, 1, net, pkt, transport="quic" if quic else "tcp")
                t += 2.0 * rng.uniform(0.9, 1.1)
        t += float(rng.lognormal(np.log(8.0), 1.0))  # user think time
    # idle keep-alive, then close_notify / FIN both ways if still in window
    t_close = t + rng.uniform(2.0, 30.0)
    tr.add([t_close, t_close + net.rtt], [rng.integers(52, 110), rng.integers(52, 110)], [0, 1])
    sub = "embedded_video" if embedded_video else ("media_heavy" if media_heavy else "standard")
    return tr, 0.0, sub


def _web_polling(tr: Trace, rng, net: FlowNet, t: float, quic: bool):
    """Single-page web app that, after the initial load, polls an API
    (XHR / long-poll / server-sent updates) every few seconds.  Small JSON
    exchanges make it resemble a chat connection."""
    pkt = quic_size(rng) if quic else MTU
    t = bulk_transfer(tr, rng, t + net.rtt, rng.lognormal(np.log(150e3), 1.0), net.cap_down, 1, net, pkt,
                      transport="quic" if quic else "tcp")
    period = float(rng.uniform(2.0, 20.0))
    while t < WINDOW_S:
        t += period * rng.uniform(0.7, 1.3)
        tr.add(t, rng.integers(200, 650), 0)
        resp = float(np.clip(rng.lognormal(np.log(900), 0.9), 120, 60e3))
        bulk_transfer(tr, rng, t + net.rtt + rng.exponential(0.02), resp, net.cap_down, 1, net, pkt,
                      transport="quic" if quic else "tcp", slow_start=False)
    return tr, 0.0, "app_polling"


def gen_chat(rng, net: FlowNet):
    """Instant messaging on a persistent TLS connection: sparse, small,
    bidirectional messages with app-level ACKs, typing indicators, receipts,
    periodic heartbeats and occasional media attachments."""
    tr = Trace()
    offset = maybe_mid_stream(rng, 0.6, 120.0)
    horizon = offset + WINDOW_S
    t = tls_handshake(tr, rng, 0.0, net) if offset == 0 else 0.0
    # heartbeats / keep-alives
    hb = float(rng.choice([10.0, 15.0, 20.0, 30.0, 45.0]))
    hb_t = np.arange(rng.uniform(0, hb), horizon, hb)
    tr.add(hb_t, rng.integers(60, 110, hb_t.size), 0)
    tr.add(hb_t + net.rtt, rng.integers(60, 110, hb_t.size), 1)
    # presence / status pushes from the server
    n_pres = rng.poisson(0.02 * horizon)
    tr.add(rng.uniform(0, horizon, n_pres), rng.integers(90, 260, n_pres), 1)
    p_down = rng.uniform(0.4, 0.7)
    attach_p = 0.04
    group = rng.random() < 0.25  # busy group chat -> higher message rate
    while t < horizon:
        # conversation active period
        active_end = t + rng.exponential(25.0)
        rate = rng.uniform(0.3, 1.0) if group else rng.uniform(0.08, 0.4)
        while t < min(active_end, horizon):
            t += rng.exponential(1.0 / rate)
            dirn = int(rng.random() < p_down)
            if rng.random() < 0.5:  # typing indicator precedes the message
                tr.add(t - rng.uniform(0.5, 4.0), rng.integers(85, 140), dirn)
            if rng.random() < attach_p:  # photo / voice note / document
                bulk_transfer(tr, rng, t, rng.lognormal(np.log(120e3), 1.2),
                              net.cap_down if dirn else net.cap_up, dirn, net)
            else:
                n_seg = 1 + int(rng.random() < 0.1)  # occasional long message
                tr.add(t + np.arange(n_seg) * 1e-3, np.clip(rng.lognormal(np.log(260), 0.5, n_seg), 90, 1400), dirn)
            tr.add(t + net.rtt, rng.integers(60, 120), 1 - dirn)  # server/app ACK
            if dirn == 1 and rng.random() < 0.7:  # delivery / read receipt
                tr.add(t + rng.uniform(0.2, 6.0), rng.integers(100, 200), 0)
        t = active_end + rng.exponential(40.0)  # idle period
    return tr, offset, "group" if group else "one_to_one"


def gen_file_transfer(rng, net: FlowNet):
    """Bulk download / upload (cloud storage sync, software update, scp over
    TLS).  Saturates the link with MTU-sized packets; a fraction of servers
    throttle with a token bucket, producing periodic bursts."""
    if rng.random() < 0.2:
        return _file_sync_small(rng, net)
    tr = Trace()
    upload = rng.random() < 0.3
    throttled = rng.random() < 0.2
    dirn = 0 if upload else 1
    offset = maybe_mid_stream(rng, 0.1, 30.0)
    t = tls_handshake(tr, rng, 0.0, net) if offset == 0 else 0.0
    tr.add(t, rng.integers(300, 700), 0)  # GET / PUT request header
    t += net.rtt if not upload else 1e-3
    size = float(np.clip(rng.lognormal(np.log(15e6), 1.6), 1e5, 3e9))
    line = net.cap_up if upload else net.cap_down
    horizon = offset + WINDOW_S
    if throttled:
        rate = float(rng.uniform(0.5e6, 5e6))
        period = float(rng.uniform(0.25, 2.0))
        sent = 0.0
        while t < horizon and sent < size:
            chunk = min(rate * period / 8.0, size - sent)
            bulk_transfer(tr, rng, t, chunk, line, dirn, net, slow_start=False)
            sent += chunk
            t += period * rng.uniform(0.95, 1.05)
    else:
        rate = line * float(rng.uniform(0.5, 0.95))
        # only generate what fits in the observation window (saves memory)
        nbytes = min(size, rate * (horizon - t) / 8.0 * 1.05)
        t = bulk_transfer(tr, rng, t, nbytes, rate, dirn, net)
        if nbytes >= size:  # completed -> server status / close
            tr.add([t + net.rtt, t + 2 * net.rtt], [rng.integers(60, 400), 52], [1 - dirn, dirn])
    sub = ("upload" if upload else "download") + ("_throttled" if throttled else "")
    return tr, offset, sub


def _file_sync_small(rng, net: FlowNet):
    """Cloud-storage client syncing many small files over one connection:
    short bulk transfers (mostly uploads) separated by idle gaps -- the
    request/response rhythm makes it look like web browsing."""
    tr = Trace()
    offset = maybe_mid_stream(rng, 0.3, 60.0)
    t = tls_handshake(tr, rng, 0.0, net) if offset == 0 else 0.0
    horizon = offset + WINDOW_S
    p_up = rng.uniform(0.3, 0.9)
    while t < horizon:
        up = rng.random() < p_up
        tr.add(t, rng.integers(300, 900), 0)  # metadata / PUT header
        size = float(np.clip(rng.lognormal(np.log(80e3), 1.3), 2e3, 20e6))
        if up:
            done = bulk_transfer(tr, rng, t + 1e-3, size, net.cap_up, 0, net)
            tr.add(done + net.rtt, rng.integers(200, 600), 1)  # server response
        else:
            done = bulk_transfer(tr, rng, t + net.rtt, size, net.cap_down, 1, net)
        t = done + float(rng.lognormal(np.log(2.5), 1.1))
    return tr, offset, "sync_small_files"


VOIP_CODECS = [
    # name, packetisation interval (s), size sampler (bytes), supports DTX
    ("opus", 0.020, lambda r, n: r.normal(120, 18, n), True),
    ("g711", 0.020, lambda r, n: np.full(n, 214.0), False),
    ("g729", 0.020, lambda r, n: np.full(n, 74.0), False),
    ("opus40", 0.040, lambda r, n: r.normal(170, 25, n), True),
    ("amr_wb", 0.020, lambda r, n: r.normal(95, 6, n), True),
]


def gen_voip(rng, net: FlowNet):
    """SRTP voice (and optionally video) call over UDP.  Constant-rate packets
    every *ptime* in both directions, talk-spurts with discontinuous
    transmission (DTX) during silence, periodic RTCP reports."""
    tr = Trace()
    name, ptime, size_fn, dtx = VOIP_CODECS[int(rng.integers(len(VOIP_CODECS)))]
    vad = dtx and rng.random() < 0.6
    video = rng.random() < 0.3
    offset = maybe_mid_stream(rng, 0.5, 120.0)
    horizon = offset + WINDOW_S
    if offset == 0:  # STUN / DTLS-SRTP setup
        n = int(rng.integers(4, 9))
        tr.add(np.arange(n) * net.rtt / 2, rng.integers(100, 1200, n), np.arange(n) % 2)
    for dirn in (0, 1):
        t = np.arange(0.05, horizon, ptime)
        t = t + (rng.normal(0, 2e-4, t.size) if dirn == 0 else np.abs(rng.normal(0, net.jitter, t.size)))
        sizes = size_fn(rng, t.size)
        if vad:  # alternate talk-spurts / silences; during silence send SID every 400 ms
            keep = np.zeros(t.size, bool)
            cur, talking = 0.0, rng.random() < 0.5
            dominance = rng.lognormal(0.0, 0.7)  # some parties mostly listen
            while cur < horizon:
                dur = rng.exponential(1.6 * dominance if talking else 1.9 / dominance)
                m = (t >= cur) & (t < cur + dur)
                keep[m] = True if talking else False
                cur += dur
                talking = not talking
            sid = (~keep) & (np.mod(np.round((t - 0.05) / ptime), round(0.4 / ptime)) == 0)
            sizes = np.where(sid, rng.integers(50, 70, t.size), sizes)
            keep |= sid
            t, sizes = t[keep], sizes[keep]
        # occasional loss in the network for downstream
        if dirn == 1:
            keep = rng.random(t.size) > net.loss
            t, sizes = t[keep], sizes[keep]
        tr.add(t, sizes, dirn)
        rtcp = np.arange(rng.uniform(0, 5), horizon, 5.0)
        tr.add(rtcp, rng.integers(80, 160, rtcp.size), dirn)
        if video:
            fps = float(rng.choice([15.0, 24.0, 30.0]))
            frames = np.arange(rng.uniform(0, 0.05), horizon, 1 / fps)
            key = np.zeros(frames.size, bool)
            key[:: int(fps * rng.uniform(2, 5))] = True
            n_pk = np.where(key, rng.integers(8, 20, frames.size), rng.integers(1, 5, frames.size))
            ft = np.repeat(frames, n_pk) + np.concatenate([np.arange(k) * 3e-4 for k in n_pk])
            fs = rng.normal(1000, 200, ft.size)
            if dirn == 1:
                ft = ft + np.abs(rng.normal(0, net.jitter, ft.size))
            tr.add(ft, fs, dirn)
    return tr, offset, ("video_call" if video else "voice") + f"_{name}"


def gen_gaming(rng, net: FlowNet):
    """Real-time multiplayer game over UDP.  The client sends small input
    packets at a fixed tick rate; the server sends larger, variable state
    snapshots.  Matches alternate with low-rate lobby phases, and some
    sessions download map/asset data in bulk."""
    if rng.random() < 0.15:
        return _casual_game(rng, net)
    tr = Trace()
    tick_down = float(rng.choice([20.0, 30.0, 60.0, 64.0, 128.0]))
    tick_up = float(min(128.0, tick_down * rng.choice([0.5, 1.0, 1.0, 2.0])))
    p2p = rng.random() < 0.2  # peer-to-peer / lockstep games: symmetric streams
    base_down = rng.lognormal(np.log(220), 0.45)
    base_up = rng.uniform(55, 150)
    if p2p:
        tick_up = tick_down = float(rng.choice([30.0, 50.0, 60.0]))
        base_down = base_up = float(rng.lognormal(np.log(110), 0.35))
    offset = maybe_mid_stream(rng, 0.6, 120.0)
    horizon = offset + WINDOW_S
    cur = 0.0
    in_match = rng.random() < 0.75
    while cur < horizon:
        dur = rng.uniform(40, 300) if in_match else rng.uniform(10, 60)
        end = min(cur + dur, horizon)
        slow = 1.0 if in_match else float(rng.uniform(4, 10))
        # upstream: client inputs (sent locally -> tight timing)
        tu = np.arange(cur, end, slow / tick_up)
        tu = tu + rng.normal(0, 3e-4, tu.size)
        tr.add(tu, rng.lognormal(np.log(base_up), 0.2, tu.size), 0)
        # downstream: server snapshots, size modulated by in-game activity
        td = np.arange(cur + rng.uniform(0, 0.01), end, slow / tick_down)
        activity = np.exp(np.cumsum(rng.normal(0, 0.05, td.size)).clip(-1, 1))
        sz = rng.lognormal(np.log(base_down * (1 if in_match else 0.4)), 0.35, td.size) * activity
        td = td + np.abs(rng.normal(0, net.jitter, td.size))
        keep = rng.random(td.size) > net.loss
        tr.add(td[keep], sz[keep], 1)
        cur = end
        in_match = not in_match
    if rng.random() < 0.15:  # asset / map download inside the same flow
        bulk_transfer(tr, rng, rng.uniform(0, horizon), rng.lognormal(np.log(300e3), 1.0), net.cap_down, 1, net,
                      1400, transport="udp")
    return tr, offset, ("p2p_" if p2p else "") + f"tick{int(tick_down)}"


def _casual_game(rng, net: FlowNet):
    """Casual / turn-based mobile game talking HTTPS to its backend: sparse
    request/response exchanges with occasional asset downloads (looks like
    web or chat traffic rather than a real-time game)."""
    tr = Trace()
    offset = maybe_mid_stream(rng, 0.5, 120.0)
    t = tls_handshake(tr, rng, 0.0, net) if offset == 0 else 0.0
    horizon = offset + WINDOW_S
    mean_gap = rng.uniform(1.0, 8.0)
    while t < horizon:
        t += rng.exponential(mean_gap)
        tr.add(t, rng.integers(200, 900), 0)
        resp = rng.lognormal(np.log(600), 1.0) if rng.random() > 0.05 else rng.lognormal(np.log(200e3), 1.0)
        bulk_transfer(tr, rng, t + net.rtt, resp, net.cap_down, 1, net, slow_start=False)
    return tr, offset, "casual_tls"


GENERATORS = {
    "video": gen_video,
    "web": gen_web,
    "chat": gen_chat,
    "file_transfer": gen_file_transfer,
    "voip": gen_voip,
    "gaming": gen_gaming,
}


# Remaining session length (median s, log-sigma) for long-lived classes,
# measured from the start of observation.  Web and file transfers end
# naturally inside their generators.
SESSION_LENGTH = {"video": (150.0, 1.1), "chat": (300.0, 1.2), "voip": (120.0, 1.1), "gaming": (500.0, 1.1)}


def generate_flow(label: str, rng: np.random.Generator, profile: NetworkProfile, min_packets: int = 4):
    """Generate one observed flow of class ``label``.

    Returns ``(t, s, d, subtype, net)``.  Flows with fewer than
    ``min_packets`` packets inside the window (e.g. a silent chat captured
    mid-stream) are re-drawn, mirroring the usual practice of discarding
    near-empty flow records.
    """
    while True:
        net = FlowNet.sample(rng, profile)
        tr, offset, sub = GENERATORS[label](rng, net)
        window = WINDOW_S
        if label in SESSION_LENGTH:  # session may end inside the window
            med, sigma = SESSION_LENGTH[label]
            window = float(np.clip(rng.lognormal(np.log(med), sigma), 1.0, WINDOW_S))
        t, s, d = tr.finalize(start=offset, window=window)
        if t.size >= min_packets:
            return t, s, d, sub, net
