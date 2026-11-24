# Copyright (c) 2026 Junyi Peng (pengjy@fit.vut.cz)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Sparse sound events (keyboard, footsteps, ...) mixed into 16 kHz speech."""

import json

import numpy as np
import soundfile as sf
from scipy import signal

SR = 16000


def rms(x):
    return float(np.sqrt(np.mean(x**2)) + 1e-9)


def load_audio16(path):
    """ Read an audio file as mono float32 at 16 kHz """
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        y = signal.resample_poly(y, SR, sr)
    return y.astype(np.float32)


def load_plan(plan_path):
    """ Read the event families

        Args:
            plan_path: JSON list of families {name, scp, prob, event_len_ms,
                rate_hz, burst (optional)}; scp is a "key path" list of
                audio files, prob the weight of the family
        Return:
            list of the families, each with its file list in 'files'
    """
    with open(plan_path) as f:
        families = json.load(f)
    for family in families:
        with open(family['scp']) as f:
            family['files'] = [
                line.split(maxsplit=1)[1].strip() for line in f
                if line.strip()
            ]
    return families


def pick_segment(y, L, rng, tries=4):
    """ Cut L samples from y, drawn among the windows at or above the 70th
        percentile of the window RMS: event recordings are mostly silence
    """
    if len(y) <= L:
        return y.copy()
    stride = max(1, L // 2)
    starts = np.arange(0, len(y) - L + 1, stride)
    if len(starts) <= tries:
        best, best_r = None, -1.0
        for s0 in starts:
            r = rms(y[s0:s0 + L])
            if r > best_r:
                best, best_r = y[s0:s0 + L].copy(), r
        return best if best is not None else y[:L].copy()
    # window RMS from the prefix energy, accumulated in float64
    energy = np.concatenate(([0.0], np.cumsum(y**2, dtype=np.float64)))
    r = np.sqrt(np.maximum(energy[starts + L] - energy[starts], 0) / L) + 1e-9
    thr = np.quantile(r, 0.70)
    cand = starts[r >= thr]
    if len(cand) == 0:
        cand = starts[[int(np.argmax(r))]]
    s = int(cand[int(rng.integers(0, len(cand)))])
    return y[s:s + L].copy()


def fade_edges(seg, ms=5.0):
    n = min(int(ms * SR / 1000), len(seg) // 2)
    if n > 0:
        w = np.sin(np.linspace(0, np.pi / 2, n))**2
        seg[:n] *= w
        seg[-n:] *= w[::-1]
    return seg


def event_times(T, rate, rng, burst=None):
    """ Poisson event times in [0, T) s, at least one; with burst = {len_s,
        gap_s} only inside bursts of these lengths and gaps
    """
    if burst:
        windows, t = [], float(rng.uniform(0, max(burst["gap_s"][1], 0.01)))
        while t < T:
            L = float(rng.uniform(*burst["len_s"]))
            windows.append((t, min(t + L, T)))
            t += L + float(rng.uniform(*burst["gap_s"]))
        if not windows:
            windows = [(0.0, T)]
    else:
        windows = [(0.0, T)]
    times = []
    for a, b in windows:
        n = int(rng.poisson(rate * (b - a)))
        times += list(rng.uniform(a, b, n))
    if not times:
        a, b = windows[int(rng.integers(0, len(windows)))]
        times = [float(rng.uniform(a, b))]
    return sorted(times)


def add_events(x, fam, rng, loader=load_audio16):
    """ Mix short cuts from 1-3 files of a family into x at Poisson times

        Args:
            x (np.ndarray): speech, shape [time], 16 kHz
            fam (dict): files, snr [lo, hi] in dB, event_len_ms [lo, hi],
                rate_hz [lo, hi], burst (optional). The SNR of an event is
                local: against the speech RMS under the event, floored 10 dB
                below the RMS of x
            rng (np.random.Generator)
            loader: path -> waveform at 16 kHz
        Return:
            np.ndarray: the mixture
    """
    T = len(x) / SR
    lo, hi = fam["snr"]
    elen = fam.get("event_len_ms", [100, 500])
    rate = float(rng.uniform(*fam.get("rate_hz", [0.5, 2.0])))
    files = fam["files"]
    nsrc = int(min(len(files), rng.integers(1, 4)))
    srcs = [files[i] for i in rng.choice(len(files), nsrc, replace=False)]
    audio = [loader(f) for f in srcs]
    y = x.copy()
    g = rms(x)
    for t in event_times(T, rate, rng, fam.get("burst")):
        L = int(rng.uniform(*elen) * SR / 1000)
        s = int(t * SR)
        if s >= len(x) - int(0.01 * SR):
            continue
        seg = pick_segment(audio[int(rng.integers(0, len(audio)))], L, rng)
        seg = fade_edges(seg[:len(x) - s])
        L = len(seg)
        if L < int(0.005 * SR) or rms(seg) < 1e-5:
            continue
        ref = max(rms(x[s:s + L]), g * 10**(-10 / 20))
        snr = float(rng.uniform(lo, hi))
        y[s:s + L] += seg / rms(seg) * ref * 10**(-snr / 20)
    return y
