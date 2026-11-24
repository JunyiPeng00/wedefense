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

import contextlib

import torch
import torch.nn as nn


class HfWav2Vec2Frontend(nn.Module):
    """wav2vec 2.0 frontend (Wav2Vec2Model of transformers, e.g.
    omniASR-W2V-1B): 16 kHz waveforms -> the hidden states of all layers."""

    def __init__(self,
                 model_path: str,
                 frozen: bool = False,
                 frame_shift: int = 20,
                 frame_length: int = 20,
                 sample_rate: int = 16000):
        """
        Args:
            model_path (str):
                Local directory with config.json, preprocessor_config.json
                and model.safetensors of a wav2vec 2.0 model, e.g.
                ylacombe/omniASR_W2V_1B_SSL.

            frozen (bool):
                If True, the encoder is not updated. Default: False

            frame_shift (int), frame_length (int):
                In milliseconds, read by the dataset to turn num_frms into a
                number of samples. The encoder has one frame per 20 ms.

            sample_rate (int):
                Sampling rate of the input. Default: 16000
        """
        super().__init__()
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
        self.sample_rate = sample_rate
        self.frozen = bool(frozen)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_path, local_files_only=True)
        self.upstream = Wav2Vec2Model.from_pretrained(model_path,
                                                      local_files_only=True)
        self.upstream.config.apply_spec_augment = False
        self.upstream.config.layerdrop = 0.0
        if self.frozen:
            self.upstream.requires_grad_(False)
        # never used without SpecAugment
        for name, param in self.upstream.named_parameters():
            if "masked_spec_embed" in name:
                param.requires_grad_(False)

    def output_size(self):
        return self.upstream.config.hidden_size

    def forward(self, input: torch.Tensor, input_lengths: torch.LongTensor):
        """
        Args:
            input (Tensor): waveforms, (B, samples).
            input_lengths (LongTensor): number of valid samples, (B,).

        Returns:
            (Tensor, list): hidden states (B, D, T, N) with D the hidden
            size and N the number of layers + 1, frames after the valid
            ones set to zero; [number of valid frames (B,)].
        """
        lengths = torch.as_tensor(input_lengths).detach().cpu().long()
        # the feature extractor normalises every waveform to zero mean and
        # unit variance, in numpy on the CPU
        waves = [
            input[i, :int(length)].detach().to("cpu", torch.float32).numpy()
            for i, length in enumerate(lengths)
        ]
        batch = self.feature_extractor(waves,
                                       sampling_rate=self.sample_rate,
                                       padding=True,
                                       return_attention_mask=True,
                                       return_tensors="pt")
        parameter = next(self.upstream.parameters())
        input_values = batch["input_values"].to(device=parameter.device,
                                                dtype=parameter.dtype)
        attention_mask = batch["attention_mask"].to(parameter.device)
        if self.frozen:
            self.upstream.eval()
        with torch.no_grad() if self.frozen else contextlib.nullcontext():
            output = self.upstream(input_values=input_values,
                                   attention_mask=attention_mask,
                                   output_hidden_states=True,
                                   return_dict=True)
        layerwise = torch.stack(output.hidden_states,
                                dim=-1).permute(0, 2, 1, 3)
        frame_lengths = self.upstream._get_feat_extract_output_lengths(
            attention_mask.sum(-1)).long()
        num_frames = layerwise.shape[2]
        frame_mask = torch.arange(
            num_frames,
            device=layerwise.device)[None, :] < frame_lengths[:, None]
        layerwise = layerwise.masked_fill(~frame_mask[:, None, :, None], 0.0)
        return layerwise, [frame_lengths]
