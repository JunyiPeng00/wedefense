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
"""Stage 7: score wav files with the test-time recipe.

  score_recipe.py --config exp/<name>/config.yaml --model_path exp/<name>/models/model_N.pt \
                  --wav_scp data/<set>/wav.scp --out scores.txt

Per file (16 kHz mono 16-bit wav): echo step (local/echo.py); then the score is the mean of P(spoof) over
its views: the whole file; for a file shorter than 4.0 s the whole file and the file repeated up to 4.0 s;
for a file longer than 12.0 s the 4.0 s windows at a hop of 1.0 s plus one flush with the end. One forward
pass per view in fp32, features on the CPU. Output: "<utt> <score>" in the order of wav.scp.
"""
import argparse

import numpy as np
import soundfile as sf
import torch

from echo import echo_step
from wedefense.dataset.dataset_utils import apply_cmvn
from wedefense.frontend import frontend_class_dict
from wedefense.models.get_model import get_model
from wedefense.models.projections import get_projection
from wedefense.utils.utils import parse_config_or_kwargs

VIEW = 64000  # 4.0 s: length of the repeated view and of every window
HOP = 16000  # 1.0 s window hop
SHORT_BELOW = 64000  # n <  4.0 s -> mean(whole file, repeated view)
LONG_ABOVE = 192000  # n > 12.0 s -> mean over the windows


def load_model(config, model_path, device):
    """Frontend + back-end + head from the config written by train.py, in plain fp32 and inference mode."""
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data_conf = config["dataset_args"]
    model = get_model(config["model"])(**config["model_args"])
    frontend_args = dict(data_conf[data_conf["frontend"] + "_args"], gpu_fbank=False)
    model.add_module("frontend", frontend_class_dict[data_conf["frontend"]](
        **frontend_args, sample_rate=data_conf["resample_rate"]))
    model.add_module("projection", get_projection(config["projection_args"]))
    model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)
    model.eval().to(device)
    model.projection.to("cpu")  # the head runs on the CPU, as in wedefense/bin/infer.py
    return model


@torch.no_grad()
def view_score(model, config, wave):
    """One waveform (float32 in [-1, 1)) -> P(spoof)."""
    wavs = torch.from_numpy(wave)[None]  # (1, samples)
    features, _ = model.frontend(wavs, torch.LongTensor([wavs.shape[1]]))
    if config["dataset_args"].get("cmvn", True):
        features = apply_cmvn(features, **config["dataset_args"].get("cmvn_args", {}))
    outputs = model(features)
    embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
    logits = model.projection(embeds.cpu(), None).numpy()[0]  # columns: bonafide, spoof (sorted labels)
    shifted = np.exp(logits - np.amax(logits))  # float32 softmax: exp(x - max) / sum
    return float((shifted / np.sum(shifted))[1])


def round10(score):
    return float(f"{score:.10f}")


def file_score(model, config, pcm):
    """int16 samples of one file -> its score: the mean of P(spoof) over the views of the file."""
    pcm = echo_step(pcm)
    n = len(pcm)
    if n > LONG_ABOVE:
        # Windows at 0, hop, 2 * hop, ... while they lie inside the file, plus one flush with the end of the file.
        starts = list(range(0, n - VIEW + 1, HOP))
        if starts[-1] != n - VIEW:
            starts.append(n - VIEW)
        views = [pcm[start:start + VIEW] for start in starts]
    elif n < SHORT_BELOW:
        # The whole file, and the file repeated end to end and cut to the view length.
        views = [pcm, np.tile(pcm, VIEW // n + 1)[:VIEW]]
    else:
        views = [pcm]
    # One forward pass per view on int16 / 32768 as float32; every view score is rounded to 10 decimals
    # before the mean.
    scores = [round10(view_score(model, config, view.astype(np.float32) / np.float32(32768.0))) for view in views]
    return round10(sum(scores) / len(scores))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="config.yaml written by train.py into the experiment directory")
    p.add_argument("--model_path", required=True)
    p.add_argument("--wav_scp", required=True, help='"<utt> <wav path>" per line')
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    config = parse_config_or_kwargs(args.config)
    model = load_model(config, args.model_path, args.device)
    with open(args.out, "w") as out:
        for line in open(args.wav_scp):
            utt, path = line.split()
            pcm, rate = sf.read(path, dtype="int16")
            assert rate == 16000 and pcm.ndim == 1, f"{path}: not a 16 kHz mono file"
            out.write(f"{utt} {file_score(model, config, pcm):.10f}\n")


if __name__ == "__main__":
    main()
