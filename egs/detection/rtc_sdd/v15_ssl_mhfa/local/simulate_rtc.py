#!/usr/bin/env python3
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
"""Offline simulation of a real-time-communication (RTC) channel.

Turns a clean ("offline") recording into a simulated transmitted one by replaying
locally what an RTC platform does to speech:

  peak normalisation + random gain
  -> [optional] additive non-speech noise: one file at a global SNR, or a
     family plan (stationary background, or Poisson-timed short events at a
     LOCAL SNR; one list <event_dir>/<family>.scp per family)
  -> [optional] room impulse response, partially wet
  -> [optional] echo: a delayed, attenuated copy of the signal
  -> WebRTC noise suppression + automatic gain control   (webrtc-noise-gain)
  -> codec: Opus with DTX / FEC, bursty packet loss and concealment
            (opuslib, libopus) | G.722 | AAC | G.711 | G.726  (ffmpeg)
  -> digital-zero runs, head / tail trim, duration floored to a frame multiple
  -> ~7 kHz low-pass
  -> peak level with an optional hard limiter (clipping)

The parameters of every stage are drawn per file from a generator seeded with
seed * 1_000_003 + <index of the file in --list> and are written to
<out_root>/sim_params.seed<seed>.jsonl.  A preset (PRESETS below) fixes the
ranges; DATA.md describes the presets.

Usage:
  simulate_rtc.py --preset reverb --seed 52 --list data/train_offline/wav.scp \
      --in_root RTCSDD/wav/train --out_root SIM/train_reverb \
      --noise_scp data/noise/musan_reverb.scp --rir_scp data/noise/rirs.scp
  (--list: first column = path relative to --in_root; outputs keep it.)
"""
import argparse
import json
import os
import subprocess
from multiprocessing import Pool

import numpy as np
import soundfile as sf
from scipy import signal

SR = 16000
FRAME10 = 160          # 10 ms, the frame size of the WebRTC audio processing

# low-pass shape (Hz, dB), peak level range, share of zeroed samples
LOWPASS_DB = [(0, 0.0), (3000, 0.0), (3500, -0.5), (4500, -1.2), (5500, -1.9),
              (6500, -2.3), (7000, -5.0), (7500, -11.0), (8000, -22.0)]
PEAK_TARGET_DBFS = (-4.0, 0.0)
ZERO_RUN_FRAC = (0.01, 0.12)

# extra codecs: name -> (ffmpeg encoder, bitrate choices or None for fixed-rate codecs)
EXTRA_CODECS = {
    "alaw": ("pcm_alaw", None),                     # G.711 A-law, 8 kHz, 64 kbps
    "mulaw": ("pcm_mulaw", None),                   # G.711 mu-law, 8 kHz, 64 kbps
    "g726": ("g726", [16000, 24000, 32000, 40000]),  # G.726 ADPCM, 8 kHz
    "aac": ("aac", [16000, 24000, 32000, 48000, 64000, 96000, 128000]),   # AAC-LC 16-128 kbps
}

# noise families: prob = share of ALL files (the remainder gets no noise); snr in dB (stationary: against the
# utterance RMS; event: LOCAL, against the speech RMS in the event window, floored 10 dB below the utterance RMS)
FAMILIES = [
    dict(name="keyboard", mode="event", prob=0.08, snr=[10, 25], event_len_ms=[40, 160], rate_hz=[6, 14],
         burst={"len_s": [1.0, 4.0], "gap_s": [1.0, 6.0]}),
    dict(name="footsteps", mode="event", prob=0.13, snr=[10, 25], event_len_ms=[100, 400], rate_hz=[1.0, 3.0]),
    dict(name="birds", mode="event", prob=0.09, snr=[10, 25], event_len_ms=[300, 1500], rate_hz=[0.3, 1.0]),
    dict(name="rain", mode="stationary", prob=0.03, snr=[15, 35]),
    dict(name="kitchen", mode="event", prob=0.03, snr=[10, 25], event_len_ms=[300, 2000], rate_hz=[0.2, 0.6]),
    dict(name="office", mode="stationary", prob=0.25, snr=[20, 35]),
]
COMBINE_PROB = 0.15   # probability that a noisy file gets a second family

DEFAULTS = dict(
    noisy_frac=0.0,                 # share of files with one noise file from --noise_scp
    snr=(5.0, 35.0),                # its SNR range, dB
    families=False,                 # use the family plan instead (--event_dir)
    rain_snr=None,                  # override of the rain family SNR range
    reverb_frac=0.0, wet=None,      # share of files convolved with an impulse response; wet-mix range (None: fully wet)
    echo_prob=0.0, echo_delay_ms=(30.0, 300.0), echo_gain_db=(-18.0, -6.0),
    ns_level=None, agc_level=None,  # WebRTC noise suppression / gain level (None: random 1-3 / 1-5)
    loss_max=20,                    # max Opus packet loss, %
    opus_modes=None,                # Opus coding modes to sample (None: libopus decides, 12-32 kbps)
    opus_nb_prob=0.0, opus_nb_bitrate=(4000, 20000),   # 8 kHz narrowband Opus path
    plc_mode=["native"],            # packet-loss concealment flavours to sample
    extra_codec_prob=0.0,           # probability of a codec from EXTRA_CODECS
    zero_prob=0.25, clip_prob=0.15,  # probability of extra zero runs / base probability of clipping
    round_ms=0.0,                   # floor the duration to a multiple of this (0: off)
    legacy=False,                   # earlier sampler: Opus from 8 kbps, plain trims, other zero-run / peak ranges,
                                    # noise not low-passed before mixing
)
_REVERB = dict(reverb_frac=0.7, ns_level=1, loss_max=10, round_ms=100.0)
_EVENT = dict(_REVERB, wet=(0.2, 0.5), families=True, opus_modes=["celt", "silk", "hybrid", "auto"],
              opus_nb_prob=0.25, plc_mode=["silence", "repeat", "native"], extra_codec_prob=0.15)
PRESETS = {
    "noise": dict(noisy_frac=1.0, legacy=True),
    "noise_hard": dict(noisy_frac=1.0, legacy=True, snr=(0.0, 20.0), clip_prob=0.3, zero_prob=0.5, loss_max=30),
    "reverb_noise": dict(_REVERB, wet=(0.25, 0.6), noisy_frac=1.0, snr=(15.0, 35.0)),
    "reverb": dict(_REVERB, wet=(0.2, 0.5), noisy_frac=0.3, snr=(15.0, 35.0)),
    "event_strong": dict(_EVENT, echo_prob=0.3),
    "event": dict(_EVENT, echo_prob=0.2, echo_gain_db=(-24.0, -12.0), opus_nb_bitrate=(8000, 20000)),
    "event_soft": dict(_EVENT, echo_prob=0.1, echo_gain_db=(-24.0, -12.0), opus_nb_bitrate=(8000, 20000),
                       rain_snr=[20, 35]),
}
PLAN = None   # family plan, set once per worker (see _init_worker)


def rms(x):
    return float(np.sqrt(np.mean(x ** 2)) + 1e-9)


def to_int16(x):
    return np.clip(x * 32768.0, -32768, 32767).astype(np.int16)


def from_int16(b):
    return np.frombuffer(b, dtype=np.int16).astype(np.float32) / 32768.0


def load_audio16(path):
    """Read any wav as mono float32 at 16 kHz."""
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        y = signal.resample_poly(y, SR, sr)
    return y.astype(np.float32)


# --- stages -----------------------------------------------------------------
def add_reverb(x, rir, rng):
    """Convolve with a room impulse response (energy-normalised)."""
    rir = rir[: int(1.0 * SR)]
    rir = rir / (np.sqrt(np.sum(rir ** 2)) + 1e-9)
    y = signal.fftconvolve(x, rir, mode="full")[: len(x)]
    return y * (np.sqrt(np.mean(x ** 2)) / (np.sqrt(np.mean(y ** 2)) + 1e-9))


def add_echo(x, delay_ms, gain_db):
    """Add a delayed, attenuated copy of the signal (residual of an imperfect echo canceller)."""
    d = int(delay_ms * SR / 1000)
    y = x.copy()
    if 0 < d < len(x):
        y[d:] += 10 ** (gain_db / 20) * x[:-d]
    return y


def add_noise(x, noise, snr_db, rng):
    if len(noise) < len(x):
        noise = np.tile(noise, int(np.ceil(len(x) / len(noise))))
    s = rng.integers(0, len(noise) - len(x) + 1)
    n = noise[s:s + len(x)]
    n = n / (np.sqrt(np.mean(n ** 2)) + 1e-9) * np.sqrt(np.mean(x ** 2)) * 10 ** (-snr_db / 20)
    return x + n


def pick_segment(y, L, rng, tries=4):
    """Cut an L-sample segment from y: the loudest of `tries` random offsets."""
    if len(y) <= L:
        return y.copy()
    best, best_r = None, -1.0
    for _ in range(tries):
        s0 = int(rng.integers(0, len(y) - L + 1))
        r = rms(y[s0:s0 + L])
        if r > best_r:
            best, best_r = y[s0:s0 + L].copy(), r
    return best


def fade_edges(seg, ms=5.0):
    n = min(int(ms * SR / 1000), len(seg) // 2)
    if n > 0:
        w = np.sin(np.linspace(0, np.pi / 2, n)) ** 2
        seg[:n] *= w
        seg[-n:] *= w[::-1]
    return seg


def event_times(T, rate, rng, burst=None):
    """Poisson event times in [0, T) s, optionally restricted to typing-like bursts (len/gap windows)."""
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
    if not times:                      # a "noisy" file must contain at least one event
        a, b = windows[int(rng.integers(0, len(windows)))]
        times = [float(rng.uniform(a, b))]
    return sorted(times)


def add_events(x, fam, rng):
    """Sparse events: short cuts from 1-3 family files at Poisson times, each scaled to a LOCAL SNR."""
    T = len(x) / SR
    lo, hi = fam["snr"]
    elen = fam.get("event_len_ms", [100, 500])
    rate = float(rng.uniform(*fam.get("rate_hz", [0.5, 2.0])))
    files = fam["files"]
    nsrc = int(min(len(files), rng.integers(1, 4)))
    srcs = [files[i] for i in rng.choice(len(files), nsrc, replace=False)]
    audio = [load_audio16(f) for f in srcs]
    y = x.copy()
    g = rms(x)
    snrs = []
    for t in event_times(T, rate, rng, fam.get("burst")):
        L = int(rng.uniform(*elen) * SR / 1000)
        s = int(t * SR)
        if s >= len(x) - int(0.01 * SR):
            continue
        seg = pick_segment(audio[int(rng.integers(0, len(audio)))], L, rng)
        seg = fade_edges(seg[: len(x) - s])
        L = len(seg)
        if L < int(0.005 * SR) or rms(seg) < 1e-5:
            continue
        ref = max(rms(x[s:s + L]), g * 10 ** (-10 / 20))
        snr = float(rng.uniform(lo, hi))
        y[s:s + L] += seg / rms(seg) * ref * 10 ** (-snr / 20)
        snrs.append(snr)
    return y, dict(mode="event", n_events=len(snrs), snr_db=(float(np.mean(snrs)) if snrs else None),
                   rate_hz=round(rate, 3), noise_files=[os.path.basename(f) for f in srcs])


def add_family_noise(x, fam, rng):
    """One noise family on x (stationary = add_noise with a family file; event = add_events)."""
    if fam.get("mode", "stationary") == "event":
        return add_events(x, fam, rng)
    nf = fam["files"][int(rng.integers(0, len(fam["files"])))]
    n = load_audio16(nf)
    n = lowpass(n, rng) if len(n) < 60 * SR else n
    snr = float(rng.uniform(*fam["snr"]))
    return add_noise(x, n, snr, rng), dict(mode="stationary", n_events=0, snr_db=snr,
                                           noise_files=[os.path.basename(nf)])


def webrtc_apm(x, agc_level, ns_level):
    from webrtc_noise_gain import AudioProcessor
    ap = AudioProcessor(int(agc_level), int(ns_level))
    pcm = to_int16(x).tobytes()
    out = []
    for i in range(0, len(pcm) - 2 * FRAME10 + 1, 2 * FRAME10):
        out.append(ap.Process10ms(pcm[i:i + 2 * FRAME10]).audio)
    return from_int16(b"".join(out))


def opus_toc_mode(pkt):
    """Coding mode from the packet TOC byte (RFC 6716, 3.1): config 0-11 SILK, 12-15 hybrid, 16-31 CELT."""
    if not pkt:
        return None
    c = pkt[0] >> 3
    return "silk" if c < 12 else ("hybrid" if c < 16 else "celt")


def opus_voip(x, bitrate, loss_pct, fec, dtx, rng, mode=None, nb=False, plc="native", vbr=None, info=None):
    """Opus encode/decode with bursty packet loss.

    mode None     : APPLICATION_VOIP at 16 kHz, libopus picks the coding mode itself
    mode "silk"   : SIGNAL_VOICE + BANDWIDTH wideband (narrowband when nb) -> SILK-only at <= 30 kbps
    mode "celt"   : APPLICATION_RESTRICTED_LOWDELAY + SIGNAL_MUSIC -> CELT-only
    mode "hybrid" : encoder run at 48 kHz with BANDWIDTH superwideband + SIGNAL_VOICE
    mode "auto"   : VOIP without a signal hint
    nb            : 8 kHz path (downsample -> Opus narrowband -> upsample)
    plc           : "native" = libopus concealment (decode of an empty packet), "silence" = zero fill,
                    "repeat" = replay the previous decoded frame
    The mode actually produced is read from every packet's TOC byte into info["opus_modes"].
    """
    import opuslib
    from opuslib.api import encoder as E, ctl
    import opuslib.constants as C
    fs = 8000 if nb else (48000 if mode == "hybrid" else SR)
    xin = x if fs == SR else signal.resample_poly(x, fs, SR)
    app = opuslib.APPLICATION_RESTRICTED_LOWDELAY if mode == "celt" else opuslib.APPLICATION_VOIP
    enc = opuslib.Encoder(fs, 1, app)
    st = enc.encoder_state
    E.encoder_ctl(st, ctl.set_bitrate, int(bitrate))
    E.encoder_ctl(st, ctl.set_packet_loss_perc, int(loss_pct))
    E.encoder_ctl(st, ctl.set_inband_fec, int(fec))
    E.encoder_ctl(st, ctl.set_dtx, int(dtx))
    if mode == "silk":
        E.encoder_ctl(st, ctl.set_signal, C.SIGNAL_VOICE)
        E.encoder_ctl(st, ctl.set_bandwidth, C.BANDWIDTH_NARROWBAND if nb else C.BANDWIDTH_WIDEBAND)
    elif mode == "hybrid":
        E.encoder_ctl(st, ctl.set_signal, C.SIGNAL_VOICE)
        E.encoder_ctl(st, ctl.set_bandwidth, C.BANDWIDTH_SUPERWIDEBAND)
    elif mode == "celt":
        E.encoder_ctl(st, ctl.set_signal, C.SIGNAL_MUSIC)
    if vbr is not None:
        E.encoder_ctl(st, ctl.set_vbr, int(vbr))
    dec = opuslib.Decoder(fs, 1)
    frame = fs // 50            # 20 ms
    pcm = to_int16(xin)
    out = []
    modes = {}
    prev = np.zeros(frame, dtype=np.int16).tobytes()
    # bursty losses: a lost packet is followed by another with p_burst
    lost_prev = False
    for i in range(0, len(pcm) - frame + 1, frame):
        pkt = enc.encode(pcm[i:i + frame].tobytes(), frame)
        m = opus_toc_mode(pkt)
        modes[m] = modes.get(m, 0) + 1
        p = 0.5 if lost_prev else loss_pct / 100.0
        if rng.random() < p:
            lost_prev = True
            if plc == "silence":
                out.append(np.zeros(frame, dtype=np.int16).tobytes())
            elif plc == "repeat":
                out.append(prev)
            else:
                out.append(dec.decode(b"", frame))
        else:
            out.append(dec.decode(pkt, frame))
            lost_prev = False
        prev = out[-1]
    y = from_int16(b"".join(out))
    if fs != SR:
        y = signal.resample_poly(y, SR, fs)
    if info is not None:
        info["opus_modes"] = modes
    return y


def ffmpeg_codec(x, codec, bitrate):
    """G.722 / AAC / G.711 / G.726 through ffmpeg (encoder + decoder), raw PCM in/out.

    The 8 kHz codecs resample 16k -> 8k -> 16k inside ffmpeg.
    """
    if codec == "g722":
        enc = ["-acodec", "g722", "-f", "g722"]
        dec_in = ["-f", "g722"]          # the raw g722 demuxer rejects -ar/-ac; it is 16 kHz mono by definition
    elif codec == "aac":
        enc = ["-acodec", "aac", "-b:a", f"{int(bitrate)}", "-f", "adts"]
        dec_in = ["-f", "aac"]
    elif codec == "alaw":
        enc = ["-ar", "8000", "-acodec", "pcm_alaw", "-f", "alaw"]
        dec_in = ["-f", "alaw", "-ar", "8000", "-ac", "1"]
    elif codec == "mulaw":
        enc = ["-ar", "8000", "-acodec", "pcm_mulaw", "-f", "mulaw"]
        dec_in = ["-f", "mulaw", "-ar", "8000", "-ac", "1"]
    elif codec == "g726":
        cs = max(2, min(5, int(bitrate) // 8000))     # code size 2..5 bits = 16..40 kbps
        enc = ["-ar", "8000", "-acodec", "g726", "-b:a", str(cs * 8000), "-f", "g726"]
        dec_in = ["-f", "g726", "-code_size", str(cs), "-ar", "8000", "-ac", "1"]
    else:
        raise ValueError(codec)
    common = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
    p1 = subprocess.run(common + ["-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", "pipe:0"] + enc + ["pipe:1"],
                        input=to_int16(x).tobytes(), stdout=subprocess.PIPE, check=True)
    p2 = subprocess.run(common + dec_in + ["-i", "pipe:0", "-f", "s16le", "-ar", str(SR), "-ac", "1", "pipe:1"],
                        input=p1.stdout, stdout=subprocess.PIPE, check=True)
    y = from_int16(p2.stdout)
    # codecs add delay / priming: lag search over the whole signal, bounded by the length difference
    max_lag = min(4000, max(0, len(y) - len(x)))
    if max_lag > 0:
        c = signal.correlate(y[: len(x) + max_lag], x, mode="valid")[: max_lag + 1]
        lag = int(np.argmax(c))
        y = y[lag:lag + len(x)]
    y = y[: len(x)]
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return y


def zero_runs(x, frac, rng):
    if frac <= 0:
        return x
    y = x.copy()
    total = int(frac * len(y))
    done = 0
    while done < total:
        L = int(rng.integers(int(0.02 * SR), int(0.12 * SR)))
        s = int(rng.integers(0, max(1, len(y) - L)))
        y[s:s + L] = 0.0
        done += L
    return y


def trim(x, head_ms, tail_ms):
    """head_ms < 0 prepends |head_ms| of digital silence; > 0 trims."""
    t = int(tail_ms * SR / 1000)
    if head_ms < 0:
        x = np.concatenate((np.zeros(int(-head_ms * SR / 1000), dtype=x.dtype), x))
        h = 0
    else:
        h = int(head_ms * SR / 1000)
    return x[h: len(x) - t] if len(x) - h - t > SR else x


def sample_head_ms(rng):
    u = rng.random()
    if u < 0.25:
        return -float(rng.uniform(0, 50))       # prepend
    if u < 0.50:
        return float(rng.uniform(0, 120))
    if u < 0.75:
        return float(rng.uniform(120, 350))
    return float(rng.uniform(350, 650))


def lowpass(x, rng):
    jitter = rng.uniform(-1.0, 1.0)
    freqs = [f for f, _ in LOWPASS_DB]
    gains = [10 ** ((g + (jitter if f >= 6500 else 0)) / 20) for f, g in LOWPASS_DB]
    h = signal.firwin2(255, freqs, gains, fs=SR)
    return signal.fftconvolve(x, h, mode="same")


def level_and_limit(x, peak_dbfs, clip, rng):
    peak = np.max(np.abs(x)) + 1e-9
    y = x * (10 ** (peak_dbfs / 20) / peak)
    if clip:  # overdrive then hard-limit: what an aggressive AGC + limiter does
        y = np.clip(y * 10 ** (rng.uniform(1.0, 4.0) / 20), -1.0, 1.0)
    return np.clip(y, -1.0, 1.0)


# --- per-file driver --------------------------------------------------------
def sample_params(rng, noisy, args):
    """Base draws first, in a fixed order; the optional stages draw only when the preset enables them."""
    codec_extra = False
    if args.extra_codec_prob > 0 and rng.random() < args.extra_codec_prob:
        codec = str(rng.choice(list(EXTRA_CODECS)))
        codec_extra = True
    else:
        codec = rng.choice(["opus", "opus", "opus", "g722", "aac"])   # Opus dominates real RTC
    old = args.legacy
    p = dict(
        noisy=bool(noisy),
        snr_db=float(rng.uniform(*args.snr)) if noisy else None,
        agc_level=int(args.agc_level) if args.agc_level is not None else int(rng.integers(1, 6)),
        ns_level=int(args.ns_level) if args.ns_level is not None else int(rng.integers(1, 4)),
        codec=str(codec),
        bitrate=int(rng.choice(([8000] if old else []) + [12000, 16000, 24000, 32000])) if codec == "opus" else (
            int(rng.choice([24000, 32000, 48000])) if codec == "aac" else 64000),
        loss_pct=int(rng.integers(5, args.loss_max + 1)) if codec == "opus" else 0,
        fec=int(rng.integers(0, 2)), dtx=int(rng.integers(0, 2)),
        # Opus DTX / PLC already leave zero runs in ~half the files; inject extra ones in a quarter
        zero_frac=(float(rng.uniform(*((0.0, 0.08) if old else ZERO_RUN_FRAC)))
                   if rng.random() < args.zero_prob else 0.0),
        head_ms=int(rng.integers(0, 400)) if old else sample_head_ms(rng),
        tail_ms=int(rng.integers(0, 150)) if old else float(rng.uniform(0, 250)),
        peak_dbfs=float(rng.uniform(*((-2.0, -0.3) if old else PEAK_TARGET_DBFS))),
        clip=bool(rng.random() < (args.clip_prob * (1.5 if noisy else 0.6))),
        gain_pre_db=float(rng.uniform(-6, 6)),
    )
    if codec_extra:
        choices = EXTRA_CODECS[codec][1]
        p["bitrate"] = int(rng.choice(choices)) if choices else 64000
        p["codec_extra"] = True
    if codec == "opus":
        modes = args.opus_modes
        nb_prob = args.opus_nb_prob
        if nb_prob > 0:
            p["opus_nb"] = bool(rng.random() < nb_prob)
        if modes:
            avail = [m for m in modes if not (p.get("opus_nb") and m == "hybrid")] or ["auto"]
            p["opus_mode"] = str(rng.choice(avail))
            p["vbr"] = int(rng.integers(0, 2))
            if p["opus_mode"] == "hybrid":
                p["fec"] = 0    # libopus falls back to SILK-only whenever in-band FEC is on
        if p.get("opus_nb"):
            lo, hi = args.opus_nb_bitrate
            p["bitrate"] = int(rng.integers(int(lo), int(hi) + 1))
        elif modes:
            p["bitrate"] = int(rng.integers(12000 if p["opus_mode"] == "celt" else 6000, 30001))
        plc = args.plc_mode
        if plc and plc != ["native"]:
            p["plc"] = str(rng.choice(plc))
    return p


def _init_worker(plan):
    global PLAN
    PLAN = plan


def process(job):
    rel, in_root, out_root, seed, noisy, noise_files, args, rir_files = job
    rng = np.random.default_rng(seed)
    x, sr = sf.read(os.path.join(in_root, rel), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    assert sr == SR, sr
    p = sample_params(rng, noisy, args)
    x = x / (np.max(np.abs(x)) + 1e-9) * 10 ** ((-6 + p["gain_pre_db"]) / 20)   # loudness-normalise first
    if PLAN is not None and p["noisy"]:
        # 'noisy' is the list of family names drawn in main (one, plus a second with combine_prob)
        fams = list(noisy)
        if len(PLAN["families"]) > 1 and rng.random() < PLAN.get("combine_prob", 0.0):
            others = [f for f in PLAN["families"] if f["name"] not in fams]
            w = np.array([f["prob"] for f in others], dtype=float)
            fams.append(others[int(rng.choice(len(others), p=w / w.sum()))]["name"])
        details = []
        for name in fams:
            fam = next(f for f in PLAN["families"] if f["name"] == name)
            x, d = add_family_noise(x, fam, rng)
            d["family"] = name
            details.append(d)
        p.update(noise_family="+".join(fams), noise_mode="+".join(d["mode"] for d in details),
                 noise_files=sum((d["noise_files"] for d in details), []), n_events=sum(d["n_events"] for d in details),
                 snr_db=details[0]["snr_db"], noise_details=details)
    elif p["noisy"]:
        nf = noise_files[rng.integers(0, len(noise_files))]
        n, nsr = sf.read(nf, dtype="float32", always_2d=False)
        if n.ndim > 1:
            n = n.mean(axis=1)
        if nsr != SR:
            n = signal.resample_poly(n, SR, nsr)
        # long files: skip the FIR (cost); they are low-passed with the mix later anyway
        n = lowpass(n[: max(len(n), 1)], rng) if len(n) < 60 * SR and not args.legacy else n
        x = add_noise(x, n, p["snr_db"], rng)
        p["noise_file"] = os.path.basename(nf)
    if rir_files and rng.random() < args.reverb_frac:
        rf = rir_files[rng.integers(0, len(rir_files))]
        r, rsr = sf.read(rf, dtype="float32", always_2d=False)
        if r.ndim > 1:
            r = r[:, 0]
        if rsr != SR:
            r = signal.resample_poly(r, SR, rsr)
        y = add_reverb(x, r, rng)
        p["rir_file"] = os.path.basename(rf)
        p["reverb"] = True
        if args.wet is not None:   # partial wet mix: an echo residual instead of a fully reverberant room
            w = float(rng.uniform(*args.wet))
            p["wet"] = round(w, 3)
            y = (1.0 - w) * x + w * y
        x = y
    if args.echo_prob > 0 and rng.random() < args.echo_prob:
        d = float(rng.uniform(*args.echo_delay_ms))
        g = float(rng.uniform(*args.echo_gain_db))
        x = add_echo(x, d, g)
        p.update(echo=True, echo_delay_ms=round(d, 1), echo_gain_db=round(g, 2))
    x = webrtc_apm(x, p["agc_level"], p["ns_level"])
    if p["codec"] == "opus":
        info = {}
        x = opus_voip(x, p["bitrate"], p["loss_pct"], p["fec"], p["dtx"], rng, mode=p.get("opus_mode"),
                      nb=p.get("opus_nb", False), plc=p.get("plc", "native"), vbr=p.get("vbr"), info=info)
        if p.get("opus_mode") or p.get("opus_nb"):
            p["opus_modes"] = info["opus_modes"]
    else:
        x = ffmpeg_codec(x, p["codec"], p["bitrate"])
    x = zero_runs(x, p["zero_frac"], rng)
    x = trim(x, p["head_ms"], p["tail_ms"])
    if args.round_ms > 0:
        n = max(1, int(SR * args.round_ms / 1000))
        x = x[: max(n, len(x) // n * n)]
    x = lowpass(x, rng)
    x = level_and_limit(x, p["peak_dbfs"], p["clip"], rng)
    out = os.path.join(out_root, rel)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sf.write(out, to_int16(x), SR, subtype="PCM_16")
    p.update(utt=rel, seed=int(seed), dur_s=round(len(x) / SR, 3))
    return p


def read_scp(path):
    """Paths of a 'key path' list."""
    return [l.split(None, 1)[1].strip() for l in open(path) if l.strip()]


def load_plan(event_dir, rain_snr):
    """The family plan with the file list of every family read from <event_dir>/<family>.scp."""
    fams = [dict(f, files=read_scp(os.path.join(event_dir, f["name"] + ".scp"))) for f in FAMILIES]
    if rain_snr is not None:
        next(f for f in fams if f["name"] == "rain")["snr"] = rain_snr
    return {"combine_prob": COMBINE_PROB, "families": fams}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True, choices=sorted(PRESETS))
    ap.add_argument("--list", required=True, help="first column: wav path relative to --in_root")
    ap.add_argument("--in_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise_scp", help="'key path' list of non-speech noise wavs")
    ap.add_argument("--rir_scp", help="'key path' list of room impulse responses")
    ap.add_argument("--event_dir", help="directory with one <family>.scp per noise family")
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--skip_existing", action="store_true", help="skip files whose output exists (resume)")
    a = ap.parse_args()
    args = argparse.Namespace(**{**DEFAULTS, **PRESETS[a.preset]})

    rels = [l.split()[0] for l in open(a.list) if l.strip()]
    skip = set()
    if a.skip_existing:   # filtered AFTER the per-file seeds / family draws, so a resume equals a fresh run
        skip = {r for r in rels if os.path.exists(os.path.join(a.out_root, r))}
        print(f"skip_existing: {len(skip)} already done, {len(rels) - len(skip)} to go")
    noise_files = read_scp(a.noise_scp) if args.noisy_frac > 0 else []
    rir_files = read_scp(a.rir_scp) if args.reverb_frac > 0 else []
    plan = load_plan(a.event_dir, args.rain_snr) if args.families else None

    master = np.random.default_rng(a.seed)
    jobs = []
    for i, rel in enumerate(rels):
        if plan is not None:
            u, cum, noisy = master.random(), 0.0, []
            for fam in plan["families"]:
                cum += float(fam["prob"])
                if u < cum:
                    noisy = [fam["name"]]
                    break
        else:
            noisy = master.random() < args.noisy_frac
        if rel in skip:
            continue
        jobs.append((rel, a.in_root, a.out_root, a.seed * 1_000_003 + i, noisy, noise_files, args, rir_files))
    os.makedirs(a.out_root, exist_ok=True)
    with Pool(a.nproc, initializer=_init_worker, initargs=(plan,)) as pool, \
            open(os.path.join(a.out_root, f"sim_params.seed{a.seed}.jsonl"), "a") as fh:
        for n, r in enumerate(pool.imap_unordered(process, jobs, chunksize=4), 1):
            fh.write(json.dumps(r) + "\n")
            if n % 500 == 0:
                print(f"{n}/{len(jobs)}", flush=True)
    print(f"done: {len(jobs)} files -> {a.out_root}")


if __name__ == "__main__":
    main()
