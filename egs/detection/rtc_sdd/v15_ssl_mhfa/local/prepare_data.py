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
"""Stage 1: data folders (wav.scp, utt2lab) for the RTC-SDD train and dev sets.

Expected layout of <rtcsdd_dir>:
  train_label.txt, dev_label.txt       "<offline|online>/<en|zh>/<name>.wav <bonafide|spoof>"
  meta/dev_offline_online_pairs.csv    offline_id,online_id  (ids start with "dev/"; online_id may be empty)
  wav/{train,dev}/<the relative paths of the label files>

The utterance key is the relative path of the label file.  Written to <data_dir>:
  train_offline  train_online  dev_offline  dev_online   every labelled recording, in label-file order
  dev_heldout    a fixed random 20 % of the dev offline recordings and their online copies
"""
import argparse
import csv
import os
import random


def write(out_dir, rows):
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/wav.scp", "w") as fw, open(f"{out_dir}/utt2lab", "w") as fl:
        for key, wav, lab in rows:
            fw.write(f"{key} {wav}\n")
            fl.write(f"{key} {lab}\n")
    print(f"{out_dir}: {len(rows)} utts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rtcsdd_dir")
    ap.add_argument("data_dir")
    ap.add_argument("--heldout_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    root = os.path.abspath(a.rtcsdd_dir)

    sets = {}
    for split in ("train", "dev"):
        rows = [(key, f"{root}/wav/{split}/{key}", lab)
                for key, lab in (l.split() for l in open(f"{root}/{split}_label.txt"))]
        for cond in ("offline", "online"):
            sets[f"{split}_{cond}"] = [r for r in rows if r[0].startswith(cond + "/")]

    # hold out whole sources: an offline recording together with its online copy
    offline = sorted(r[0] for r in sets["dev_offline"])
    held = set(random.Random(a.seed).sample(offline, int(a.heldout_frac * len(offline))))
    with open(f"{root}/meta/dev_offline_online_pairs.csv", newline="") as f:
        twin = {r["offline_id"].strip()[4:]: r["online_id"].strip()[4:] for r in csv.DictReader(f)}
    held |= {twin[k] for k in held if twin[k]}
    sets["dev_heldout"] = [r for r in sets["dev_offline"] + sets["dev_online"] if r[0] in held]

    for name, rows in sets.items():
        write(f"{a.data_dir}/{name}", rows)


if __name__ == "__main__":
    main()
