# Copyright (c) 2026 Junyi Peng (pengjy@fit.vut.cz)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Batched torch re-implementation of transformers' SeamlessM4TFeatureExtractor
(the input features of w2v-BERT 2.0).

Follows transformers 4.56.2 feature_extraction_seamless_m4t.py and
audio_utils.spectrogram step by step:
  waveform (float32) * 2**15 -> float64; frames of 400 samples, hop 160, no
  centring; per frame: subtract the frame mean, pre-emphasis 0.97 inside the
  frame (first sample * (1 - 0.97)), povey window, zero-pad to 512, rfft; the
  spectrum is rounded to complex64 (as the numpy code stores it), |.|**2 in
  float64; kaldi mel filter bank (80 bins), floor 1.192092955078125e-07,
  natural log, cast to float32; per-utterance, per-bin normalisation
  (x - mean) / sqrt(var(ddof=1) + 1e-7) over the valid frames; right padding
  with padding_value to the longest utterance rounded up to an even frame
  count; stacking of 2 consecutive frames -> 160 dims; attention mask = mask
  of every second frame (stacked frame j is valid iff j < n_frames // 2).

The window and the mel filter bank are taken from the feature extractor
object. Everything runs on the device of the input tensor, without gradients
and with autocast disabled.
"""

import numpy as np
import torch

FRAME_LENGTH = 400
HOP_LENGTH = 160
FFT_LENGTH = 512
PREEMPHASIS = 0.97
MEL_FLOOR = 1.192092955078125e-07
VAR_EPS = 1e-7


class GpuSeamlessFbank:
    """Callable: (waveforms [B, S] float in [-1, 1], lengths [B] samples) ->
    (input_features [B, T, 160] float32, attention_mask [B, T] int32), as
    SeamlessM4TFeatureExtractor(list of wav[i, :len_i], padding=True)."""

    def __init__(self, feature_extractor):
        fe = feature_extractor
        if fe.stride != 2 or fe.num_mel_bins != 80 or fe.sampling_rate != 16000:
            raise ValueError("Only the w2v-BERT 2.0 setting (16 kHz, 80 mel "
                             "bins, stride 2) is reproduced")
        self.num_mel_bins = fe.num_mel_bins
        self.padding_value = float(fe.padding_value)
        self._window = torch.from_numpy(
            np.asarray(fe.window, dtype=np.float64).copy())
        self._mel = torch.from_numpy(
            np.asarray(fe.mel_filters, dtype=np.float64).copy())
        self._cache = {}

    def _constants(self, device):
        key = str(device)
        if key not in self._cache:
            self._cache[key] = (self._window.to(device), self._mel.to(device))
        return self._cache[key]

    @torch.no_grad()
    def __call__(self, waveforms, lengths):
        with torch.autocast(device_type=waveforms.device.type, enabled=False):
            return self._extract(waveforms, lengths)

    def _extract(self, waveforms, lengths):
        device = waveforms.device
        window, mel_filters = self._constants(device)
        lengths = torch.as_tensor(lengths).to(device=device, dtype=torch.long)
        # [B] valid 10 ms frames
        n = 1 + torch.div(lengths - FRAME_LENGTH,
                          HOP_LENGTH,
                          rounding_mode="floor")
        fmax = int(n.max())
        used = (fmax - 1) * HOP_LENGTH + FRAME_LENGTH
        # float32 waveform * 2**15 (exact in float32), then float64
        x = (waveforms[:, :used].to(torch.float32) * 32768.0).to(torch.float64)
        frames = x.unfold(1, FRAME_LENGTH, HOP_LENGTH)  # [B, fmax, 400] view
        frames = frames - frames.mean(dim=-1, keepdim=True)
        pre = torch.empty_like(frames)
        pre[..., 1:] = frames[..., 1:] - PREEMPHASIS * frames[..., :-1]
        pre[..., 0] = frames[..., 0] * (1 - PREEMPHASIS)
        pre *= window
        spec = torch.fft.rfft(pre, n=FFT_LENGTH, dim=-1)  # [B, fmax, 257]
        # numpy stores the STFT in complex64
        spec = spec.to(torch.complex64).to(torch.complex128)
        power = spec.abs().pow(2.0)
        logmel = torch.clamp(power @ mel_filters,
                             min=MEL_FLOOR).log().to(torch.float32)

        # normalisation over the valid frames, in float32 as numpy does
        # (sums accumulated in float64)
        valid = (torch.arange(fmax, device=device)[None, :] <
                 n[:, None])[..., None]  # [B, fmax, 1]
        count = n[:, None].to(torch.float64)
        mean = ((logmel.to(torch.float64) * valid).sum(1) / count).to(
            torch.float32)  # [B, 80]
        centred = logmel - mean[:, None, :]
        var = ((centred.to(torch.float64).pow(2.0) * valid).sum(1) /
               (count - 1)).to(torch.float32)
        normed = centred / torch.sqrt(var + VAR_EPS)[:, None, :]
        normed = torch.where(valid, normed,
                             normed.new_full((), self.padding_value))
        if fmax % 2:  # pad_to_multiple_of=2
            pad = normed.new_full((normed.shape[0], 1, normed.shape[2]),
                                  self.padding_value)
            normed = torch.cat([normed, pad], dim=1)
        steps = normed.shape[1] // 2
        features = normed.reshape(normed.shape[0], steps,
                                  2 * self.num_mel_bins)
        attention_mask = (torch.arange(steps, device=device)[None, :] <
                          torch.div(n, 2, rounding_mode="floor")[:, None]).to(
                              torch.int32)
        return features, attention_mask
