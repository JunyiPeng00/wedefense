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
"""Posteriors of wedefense/bin/infer.py -> score file "<utt> <P(spoof)>".

The column of "spoof" is not fixed: the toolkit numbers the sorted labels of the training
utt2lab, so the same map is rebuilt from that file.

  make_scores.py --posteriors_scp exp/.../posteriors.scp --train_label data/train/utt2lab \
                 --protocol data/dev_heldout/utt2lab --out exp/.../scores.txt
"""
import argparse

import kaldiio


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--posteriors_scp", required=True)
    p.add_argument("--train_label", required=True, help="training utt2lab: fixes the label -> column order")
    p.add_argument("--protocol", required=True, help="first column: utterance ids; the output follows this order")
    p.add_argument("--out", required=True)
    p.add_argument("--spoof_label", default="spoof")
    args = p.parse_args()

    labels = [l.split()[1] for l in open(args.train_label) if l.strip()]
    mapping = {lab: i for i, lab in enumerate(sorted(set(labels)))}
    spoof_col = mapping[args.spoof_label]
    print(f"label->column map {mapping}; using column {spoof_col} as spoof")

    scores = dict(kaldiio.load_scp_sequential(args.posteriors_scp))
    ids = [l.split()[0] for l in open(args.protocol) if l.strip()]
    with open(args.out, "w") as fout:
        for utt in ids:
            fout.write(f"{utt} {float(scores[utt][spoof_col]):.10f}\n")
    print(f"wrote {len(ids)} scores to {args.out}")


if __name__ == "__main__":
    main()
