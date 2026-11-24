# Copyright (c) 2026 Junyi Peng (pengjy@fit.vut.cz)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Echo detector and single-tap echo removal: the first step of local/score_recipe.py.

Signal processing on one file: no model, no random number, no state between files. All FFTs run in float64
with explicit casts, so the result does not depend on the NumPy version.
"""
import numpy as np

DELAYS = (1000, 667, 725, 726)  # candidate echo delays in samples at 16 kHz


def autocorr(y: np.ndarray) -> np.ndarray:
    """Normalised autocorrelation ac[k] / ac[0] of the de-meaned signal (float32 de-meaning, float64 FFT of size 2n)."""
    y = y - y.mean()
    n = len(y)
    spectrum = np.fft.rfft(y.astype(np.float64), 2 * n)
    ac = np.fft.irfft(spectrum * np.conj(spectrum))[:n]
    return ac / (ac[0] + 1e-12)


def detect(y: np.ndarray):
    """Samples / 32768 as float32 -> (delay D, autocorrelation r) if the file is flagged, else None."""
    if len(y) >= 4000:
        # Branch A: the autocorrelation peak over the lags 600..1099 lies within 2 samples of a candidate delay
        # and has z > 6 against the lags more than 5 samples away from it. r is the value at the peak lag.
        lags = np.arange(600, 1100)
        seg = autocorr(y)[600:1100]
        peak_lag = int(lags[np.argmax(seg)])
        peak = float(seg.max())
        rest = np.abs(lags - peak_lag) > 5
        z = float((peak - seg[rest].mean()) / (seg[rest].std() + 1e-9))
        nearest = min(DELAYS, key=lambda c: abs(c - peak_lag))
        if abs(peak_lag - nearest) <= 2 and z > 6.0:
            return nearest, peak
    # Branch B: mean log-power cepstrum over Hann frames of 16,384 samples, hop 8,000 (a shorter file is
    # zero-padded to one frame, the tail beyond the last full frame is ignored, frames with a mean square
    # <= 1e-8 are dropped). Flagged if the largest robust z over the lags 480..5599 is > 40, sits exactly on
    # a candidate delay, and the file is longer than that delay.
    size, hop = 16384, 8000
    padded = np.pad(y, (0, max(0, size - len(y))))
    frames = np.lib.stride_tricks.sliding_window_view(padded, size)[::hop]
    frames = frames[np.mean(frames ** 2, axis=1) > 1e-8]
    if not len(frames):
        return None
    windowed = (frames * np.hanning(size)).astype(np.float64)
    log_power = np.log(np.abs(np.fft.rfft(windowed, axis=1)) ** 2 + 1e-10)
    cepstrum = np.fft.irfft(log_power, n=size, axis=1).mean(0)
    seg = np.abs(cepstrum[480:5600])
    median = np.median(seg)
    mad = np.median(np.abs(seg - median)) + 1e-12
    z = float(((seg - median) / mad).max())
    lag = int(np.argmax(seg) + 480)
    delay = 726 if lag == 725 else lag  # a cepstral peak at 725 is removed with delay 726
    if z > 40.0 and lag in DELAYS and len(y) > delay:
        return delay, float(autocorr(y)[delay])
    return None


def echo_gain(r: float) -> float:
    """Gain a of the single-tap model from the autocorrelation r at the delay (r = a / (1 + a^2)), clamped."""
    r = float(np.clip(r, 0.02, 0.499))
    a = (1 - np.sqrt(1 - 4 * r * r)) / (2 * r)
    return float(np.clip(a, 0.1, 0.7))


def remove_echo(y: np.ndarray, delay: int, gain: float) -> np.ndarray:
    """Invert y[t] = x[t] + a * x[t - D] over the whole file; returns int16 samples of the same length."""
    # float64 recursion on the samples as they are (not de-meaned), block by block: within a block of D
    # samples the taps do not overlap, so this is the sample-by-sample recursion.
    x = y.astype(np.float64).copy()
    for start in range(delay, len(x), delay):
        end = min(start + delay, len(x))
        x[start:end] -= gain * x[start - delay:end - delay]
    x32 = np.clip(x, -1, 1).astype(np.float32)
    # Back to 16 bit by a FLOOR to the grid, not a round to nearest: rint(x * 2**31) >> 16.
    v = np.rint(x32.astype(np.float64) * 2147483648.0)
    v = np.clip(v, -2147483648.0, 2147483647.0).astype(np.int64)
    return (v >> 16).astype(np.int16)


def echo_step(pcm16: np.ndarray) -> np.ndarray:
    """int16 samples of one file -> the int16 samples to score: de-echoed if the detector flags the file."""
    y = pcm16.astype(np.float32) / np.float32(32768.0)
    found = detect(y)
    if found is None:
        return pcm16
    delay, r = found
    return remove_echo(y, delay, echo_gain(r))
