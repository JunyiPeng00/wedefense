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
"""Stage 3: the combined training list and the validation list (wav.scp, utt2lab).

  <data_dir>/train   train online recordings + the simulated copies of the train offline recordings,
                     then the same for dev without the held-out sources (<data_dir>/dev_heldout)
  <data_dir>/val     held-out dev online recordings + the simulated copies of the held-out dev
                     offline recordings in the presets --val_presets

The simulated copies are the lines of conf/sim_variants.txt.  Their key is
<preset>/<key of the offline recording>; the first component of a key ("online" or
the preset) is the group name used by the validation metric.
"""
import argparse
import os


def read(data_dir, name):
    wav = dict(l.split() for l in open(f"{data_dir}/{name}/wav.scp"))
    return sorted((key, wav[key], lab) for key, lab in (l.split() for l in open(f"{data_dir}/{name}/utt2lab")))


def write(out_dir, rows):
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/wav.scp", "w") as fw, open(f"{out_dir}/utt2lab", "w") as fl:
        for key, wav, lab in rows:
            fw.write(f"{key} {wav}\n")
            fl.write(f"{key} {lab}\n")
    print(f"{out_dir}: {len(rows)} utts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("sim_dir")
    ap.add_argument("--variants", default="conf/sim_variants.txt")
    ap.add_argument("--val_presets", nargs="+", default=["reverb_noise", "reverb"])
    a = ap.parse_args()
    sim_dir = os.path.abspath(a.sim_dir)
    variants = [l.split() for l in open(a.variants) if l.strip() and not l.startswith("#")]
    held = {key for key, _, _ in read(a.data_dir, "dev_heldout")}

    train, val = [], []
    for subset in ("train", "dev"):
        online = read(a.data_dir, f"{subset}_online")
        offline = read(a.data_dir, f"{subset}_offline")
        train += [r for r in online if r[0] not in held]
        val += [r for r in online if r[0] in held]
        for preset in (p for s, p, _ in variants if s == subset):
            sim = [(f"{preset}/{key}", f"{sim_dir}/{subset}_{preset}/{key}", lab, key in held)
                   for key, _, lab in offline]
            train += [r[:3] for r in sim if not r[3]]
            if preset in a.val_presets:
                val += [r[:3] for r in sim if r[3]]
    write(f"{a.data_dir}/train", train)
    write(f"{a.data_dir}/val", val)


if __name__ == "__main__":
    main()
