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
"""Macro-F1 at a fixed threshold: (F1_spoof + F1_bonafide) / 2, a score >= threshold counts as spoof.

  macro_f1.py SCORES LABELS [--threshold 0.5] [--subsets a,b --lam 0.3]

SCORES: "<utt> <P(spoof)>", LABELS: "<utt> <bonafide|spoof>". Prints the overall value and one value
per directory of the utterance ids; with --subsets a,b (first components of the ids) also
lam * Macro-F1(a) + (1 - lam) * Macro-F1(b).
"""
import argparse
from collections import defaultdict

from sklearn.metrics import f1_score

SPOOF, BONAFIDE = "spoof", "bonafide"


def macro_f1(y_true, y_pred):
    return 100.0 * f1_score(y_true, y_pred, average="macro", labels=[SPOOF, BONAFIDE])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("scores")
    p.add_argument("labels")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--subsets", default="", help="two first components of the ids for the weighted value")
    p.add_argument("--lam", type=float, default=0.3, help="weight of the first subset")
    args = p.parse_args()

    labels = dict(line.split() for line in open(args.labels, encoding="utf-8") if line.strip())

    y_true, y_pred = [], []
    by_dir = defaultdict(lambda: ([], []))
    by_subset = defaultdict(lambda: ([], []))
    subsets = args.subsets.split(",") if args.subsets else []
    missing = 0

    for line in open(args.scores, encoding="utf-8"):
        parts = line.split()
        if len(parts) != 2:
            continue
        utt, score = parts[0], float(parts[1])
        if utt not in labels:
            missing += 1
            continue
        pred = SPOOF if score >= args.threshold else BONAFIDE
        truth = labels[utt]
        y_true.append(truth)
        y_pred.append(pred)
        by_dir["/".join(utt.split("/")[:-1]) or "."][0].append(truth)
        by_dir["/".join(utt.split("/")[:-1]) or "."][1].append(pred)
        head = utt.split("/")[0]
        if head in subsets:
            by_subset[head][0].append(truth)
            by_subset[head][1].append(pred)

    if missing:
        print(f"warning: {missing} scored utts have no label")
    print(f"scored {len(y_true)}/{len(labels)} utts at threshold {args.threshold}")
    print(f"overall Macro-F1 = {macro_f1(y_true, y_pred):.2f}")

    if len(subsets) == 2 and all(s in by_subset for s in subsets):
        a, b = subsets
        fa = macro_f1(*by_subset[a])
        fb = macro_f1(*by_subset[b])
        print(f"  {a:<8} Macro-F1 = {fa:.2f}   (n={len(by_subset[a][0])})")
        print(f"  {b:<8} Macro-F1 = {fb:.2f}   (n={len(by_subset[b][0])})")
        print(f"  weighted ({args.lam}*{a} + {1 - args.lam:.1f}*{b}) = {args.lam * fa + (1 - args.lam) * fb:.2f}")

    print("\nbreakdown by directory:")
    for key in sorted(by_dir):
        t, pr = by_dir[key]
        print(f"  {key:<32} n={len(t):>7}  Macro-F1={macro_f1(t, pr):.2f}")


if __name__ == "__main__":
    main()
