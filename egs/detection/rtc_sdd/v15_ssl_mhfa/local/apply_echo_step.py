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
"""Stage 8: the echo step (local/echo.py) on a list of 16 kHz mono 16-bit wav files.

  apply_echo_step.py data/<set>/wav.scp data/<set>_deecho

Writes <out_dir>/wav.scp: a file that the detector flags points to its de-echoed copy
<out_dir>/wav/<utt>, every other line is kept.
"""
import os
import sys

import soundfile as sf

from echo import echo_step


def main():
    wav_scp, out_dir = sys.argv[1:]
    os.makedirs(out_dir, exist_ok=True)
    flagged = 0
    with open(f"{out_dir}/wav.scp", "w") as out:
        for line in open(wav_scp):
            utt, path = line.split()
            pcm, rate = sf.read(path, dtype="int16")
            clean = echo_step(pcm)
            if clean is not pcm:
                path = os.path.abspath(f"{out_dir}/wav/{utt}")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                sf.write(path, clean, rate, subtype="PCM_16", format="WAV")
                flagged += 1
            out.write(f"{utt} {path}\n")
    print(f"{out_dir}: echo removed from {flagged} files")


if __name__ == "__main__":
    main()
