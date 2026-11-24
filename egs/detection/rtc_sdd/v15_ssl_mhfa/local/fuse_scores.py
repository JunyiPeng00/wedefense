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
"""Stage 8: fuse the score files of several teachers.

  fuse_scores.py --scores a.txt b.txt c.txt [--weights w_a w_b w_c] --out fused.txt

Every file has "<utt> <P(spoof)>" for the same utterances.  Fused score = sum(w_i * p_i) / sum(w_i),
equal weights by default; written in the order of the first file.
"""
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", nargs="+", required=True, help='score files "<utt> <P(spoof)>", one per teacher')
    p.add_argument("--weights", nargs="+", type=float, help="one weight per score file (default: equal)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    tables = [dict(l.split() for l in open(f)) for f in args.scores]
    assert all(t.keys() == tables[0].keys() for t in tables), "the score files list different utterances"
    weights = args.weights or [1.0] * len(tables)
    assert len(weights) == len(tables), "one weight per score file"
    with open(args.out, "w") as out:
        for utt in tables[0]:
            fused = sum(w * float(t[utt]) for w, t in zip(weights, tables)) / sum(weights)
            out.write(f"{utt} {fused:.10f}\n")
    print(f"{args.out}: {len(tables[0])} utts, {len(tables)} teachers, weights {weights}")


if __name__ == "__main__":
    main()
