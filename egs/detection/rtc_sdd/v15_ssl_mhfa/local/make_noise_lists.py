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
"""Noise, impulse-response and event lists ('key path') for local/simulate_rtc.py, from public corpora.

  musan_noise.scp   MUSAN: every wav under <musan_dir>/noise
  musan_reverb.scp  the MUSAN noise files listed in --musan_reverb (a file can be listed more than once)
  rirs.scp          RIRS_NOISES: simulated_rirs/{mediumroom,smallroom}, plus the measured responses of
                    real_rirs_isotropic_noises (no noise files, no large rooms) --real_rir_repeat times
  <family>.scp      keyboard, footsteps, birds, rain, kitchen, office: non-speech classes of ESC-50
                    (meta/esc50.csv categories) and FSD50K (dev set labels), in the order of the csv files;
                    the office family starts with the pieces listed in --rnnoise_pieces: 30 s pieces of the
                    RNNoise contributions (raw s16le mono 48 kHz), written at 16 kHz to <out_dir>/rnnoise_wav
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import soundfile as sf
from scipy import signal

SR = 16000
# ESC-50 classes per family (plainly non-speech, no human body sounds)
ESC_FAMILIES = {
    "keyboard": ["keyboard_typing", "mouse_click"],
    "footsteps": ["footsteps", "door_wood_knock"],
    "birds": ["chirping_birds"],
    "rain": ["rain", "wind", "thunderstorm"],
    "kitchen": ["can_opening", "pouring_water", "water_drops"],
    "office": ["clock_tick", "door_wood_creaks"],
}
# FSD50K labels per family
FSD_FAMILIES = {
    "keyboard": ["Computer_keyboard", "Typing", "Typewriter"],
    "footsteps": ["Walk_and_footsteps", "Run", "Knock", "Door", "Drawer_open_or_close"],
    "birds": ["Bird", "Bird_vocalization_and_bird_call_and_bird_song", "Chirp_and_tweet"],
    "rain": ["Rain", "Raindrop", "Thunderstorm", "Water"],
    "kitchen": ["Dishes_and_pots_and_pans", "Cutlery_and_silverware", "Frying_(food)", "Glass",
                "Chink_and_clink", "Cupboard_open_or_close", "Water_tap_and_faucet", "Boiling", "Microwave_oven"],
    "office": ["Printer", "Mechanical_fan", "Scissors", "Writing"],
}
FSD_EXCLUDE = {"Speech", "Human_voice", "Conversation", "Male_speech_and_man_speaking",
               "Female_speech_and_woman_speaking", "Child_speech_and_kid_speaking", "Chatter", "Crowd", "Singing",
               "Human_group_actions", "Laughter", "Shout", "Yell", "Whispering", "Screaming", "Cheering", "Music",
               "Musical_instrument"}
# stationary families need material long enough to loop without an obvious period
STATIONARY = {"rain": 3.0, "office": 3.0}
EVENT_MIN_S = 0.15
PIECE_S = 30.0


def write_scp(path, rows):
    with open(path, "w") as fh:
        for k, p in rows:
            fh.write(f"{k} {p}\n")
    print(f"  {path}: {len(rows)}")


def find_wavs(root, prefix):
    rel = sorted(os.path.relpath(os.path.join(d, f), root)
                 for d, _, fs in os.walk(root) for f in fs if f.endswith(".wav"))
    return [(f"{prefix}/{r}", os.path.join(root, r)) for r in rel]


def rir_rows(rirs_dir, repeat):
    rows = []
    for room in ("mediumroom", "smallroom"):
        rows += find_wavs(os.path.join(rirs_dir, "simulated_rirs", room), room)
    real = [r for r in find_wavs(os.path.join(rirs_dir, "real_rirs_isotropic_noises"), "real")
            if "noise" not in r[0] and "largeroom" not in r[0]]
    return rows + real * repeat


def read_keys(path):
    return [l.split()[0] for l in open(path) if l.strip()]


def rnnoise_rows(rnnoise_dir, out, pieces):
    """The listed pieces "<recording>_<index>" (index = position of the 30 s piece), in list order."""
    wav_dir = os.path.join(out, "rnnoise_wav")
    os.makedirs(wav_dir, exist_ok=True)
    keep = set(pieces)
    n = int(PIECE_S * SR)
    for stem in sorted({p.rsplit("_", 1)[0] for p in pieces}):
        b = open(os.path.join(rnnoise_dir, stem + ".raw"), "rb").read()
        x = np.frombuffer(b[: len(b) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
        y = signal.resample_poly(x, 1, 3)   # 48 kHz -> 16 kHz
        for k in range(0, len(y), n):
            op = os.path.join(wav_dir, f"{stem}_{k // n:02d}.wav")
            if f"{stem}_{k // n:02d}" in keep and not os.path.exists(op):
                sf.write(op, y[k:k + n], SR, subtype="PCM_16")
    return [(f"rnnoise/{p}", os.path.join(wav_dir, p + ".wav")) for p in pieces]


def esc_rows(esc_dir):
    fam = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(esc_dir, "meta", "esc50.csv"))):
        for name, classes in ESC_FAMILIES.items():
            if r["category"] in classes:
                fam[name].append((f"esc50/{r['category']}/{r['filename'][:-4]}",
                                  os.path.join(esc_dir, "audio", r["filename"])))
    return fam


def fsd_rows(fsd_dir):
    fam = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(fsd_dir, "FSD50K.ground_truth", "dev.csv"))):
        labels = set(r["labels"].split(","))
        if labels & FSD_EXCLUDE:
            continue
        hit = [name for name, labs in FSD_FAMILIES.items() if labels & set(labs)]
        if not hit:
            continue
        path = os.path.join(fsd_dir, "FSD50K.dev_audio", f"{r['fname']}.wav")
        dur = sf.info(path).duration
        for name in hit:
            if dur < STATIONARY.get(name, EVENT_MIN_S):
                continue
            fam[name].append((f"fsd50k/{name}/{r['fname']}", path))
    return fam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--musan_dir", required=True, type=os.path.abspath)
    ap.add_argument("--rirs_dir", required=True, type=os.path.abspath, help="RIRS_NOISES")
    ap.add_argument("--esc50_dir", required=True, type=os.path.abspath, help="ESC-50-master")
    ap.add_argument("--fsd50k_dir", required=True, type=os.path.abspath)
    ap.add_argument("--rnnoise_dir", required=True, type=os.path.abspath, help="rnnoise_contributions")
    ap.add_argument("--musan_reverb", default="conf/musan_reverb.txt", help="keys of musan_reverb.scp")
    ap.add_argument("--rnnoise_pieces", default="conf/rnnoise_pieces.txt", help="RNNoise pieces of the office family")
    ap.add_argument("--real_rir_repeat", type=int, default=23)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    musan = find_wavs(os.path.join(a.musan_dir, "noise"), "noise")
    path = dict(musan)
    write_scp(f"{a.out_dir}/musan_noise.scp", musan)
    write_scp(f"{a.out_dir}/musan_reverb.scp", [(k, path[k]) for k in read_keys(a.musan_reverb)])
    write_scp(f"{a.out_dir}/rirs.scp", rir_rows(a.rirs_dir, a.real_rir_repeat))
    esc, fsd = esc_rows(a.esc50_dir), fsd_rows(a.fsd50k_dir)
    rnn = rnnoise_rows(a.rnnoise_dir, os.path.abspath(a.out_dir), read_keys(a.rnnoise_pieces))
    for name in ESC_FAMILIES:
        write_scp(f"{a.out_dir}/{name}.scp", (rnn if name == "office" else []) + esc[name] + fsd[name])


if __name__ == "__main__":
    main()
