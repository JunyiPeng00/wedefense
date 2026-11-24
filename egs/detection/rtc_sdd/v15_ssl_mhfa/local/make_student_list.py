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
"""Stage 9: the training list of a student (wav.scp, utt2lab).

  make_student_list.py --train_dir data/train --wav_scp data/<set>_deecho/wav.scp --scores fused.txt \
                       --out_dir data/train_student [--threshold 0.5] [--copies 3]

<out_dir> = the rows of <train_dir>, then --copies copies of the rows of --wav_scp with the label
"spoof" if the fused score is >= threshold, else "bonafide".  Copy i has the keys tl<i>/<utt>.  The
loader draws rows uniformly, so the number of copies sets the share of the teacher-labelled rows.
"""
import argparse

from make_train_list import write


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True, help="labelled training list: wav.scp, utt2lab")
    ap.add_argument("--wav_scp", required=True, help='"<utt> <wav path>" of the audio without labels')
    ap.add_argument("--scores", required=True, help='fused scores "<utt> <P(spoof)>"')
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--copies", type=int, default=3)
    a = ap.parse_args()

    wav = dict(l.split() for l in open(f"{a.train_dir}/wav.scp"))
    rows = [(key, wav[key], lab) for key, lab in (l.split() for l in open(f"{a.train_dir}/utt2lab"))]
    score = dict(l.split() for l in open(a.scores))
    new = [(utt, path, "spoof" if float(score[utt]) >= a.threshold else "bonafide")
           for utt, path in (l.split() for l in open(a.wav_scp))]
    rows += [(f"tl{i}/{utt}", path, lab) for i in range(a.copies) for utt, path, lab in new]
    write(a.out_dir, rows)
    print(f"teacher-labelled: {len(new)} x {a.copies}, {sum(lab == 'spoof' for _, _, lab in new)} spoof")


if __name__ == "__main__":
    main()
