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

from wedefense.frontend.w2vbert_fbank_gpu import GpuSeamlessFbank


class W2vBertFrontend(nn.Module):
    """w2v-BERT 2.0 frontend: 16 kHz waveforms -> the hidden states of all
    layers (the input of the first layer and the output of every layer)."""

    def __init__(self,
                 model_path: str,
                 frozen: bool = False,
                 gpu_fbank: bool = False,
                 frame_shift: int = 20,
                 frame_length: int = 20,
                 sample_rate: int = 16000):
        """
        Args:
            model_path (str):
                Local directory with config.json, preprocessor_config.json
                and model.safetensors of facebook/w2v-bert-2.0.

            frozen (bool):
                If True, the encoder is not updated. Default: False

            gpu_fbank (bool):
                If True, the input features are computed on the device of the
                waveforms (w2vbert_fbank_gpu.py); if False, by the feature
                extractor of transformers, in numpy on the CPU. Both give
                the same features up to rounding. Default: False

            frame_shift (int), frame_length (int):
                In milliseconds, read by the dataset to turn num_frms into a
                number of samples. The encoder has one frame per 20 ms.

            sample_rate (int):
                Sampling rate of the input. Default: 16000
        """
        super().__init__()
        # needs transformers >= 4.37
        from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel
        self.sample_rate = sample_rate
        self.frozen = frozen
        self.feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(
            model_path, local_files_only=True)
        self.upstream = Wav2Vec2BertModel.from_pretrained(
            model_path, local_files_only=True)
        self.upstream.config.apply_spec_augment = False
        self.upstream.config.layerdrop = 0.0
        # a plain attribute: no parameter, no buffer, same state_dict
        self.gpu_fbank = GpuSeamlessFbank(
            self.feature_extractor) if gpu_fbank else None
        if self.frozen:
            self.upstream.requires_grad_(False)
        # never used without SpecAugment
        for name, param in self.upstream.named_parameters():
            if "masked_spec_embed" in name:
                param.requires_grad_(False)

    def output_size(self):
        return self.upstream.config.hidden_size

    def prepare_inputs(self, input, input_lengths):
        """Waveforms (B, samples) -> input_features (B, T, 160) and
        attention_mask (B, T) of the encoder."""
        if self.gpu_fbank is not None:
            lengths = torch.as_tensor(input_lengths).detach()
            return self.gpu_fbank(input.detach(), lengths)
        lengths = torch.as_tensor(input_lengths).detach().cpu()
        waveforms = input.detach().to(device="cpu", dtype=torch.float32)
        waves = [
            waveforms[i, :int(length)].numpy()
            for i, length in enumerate(lengths)
        ]
        batch = self.feature_extractor(waves,
                                       sampling_rate=self.sample_rate,
                                       padding=True,
                                       return_attention_mask=True,
                                       return_tensors="pt")
        return batch["input_features"], batch["attention_mask"]

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
        features, attention_mask = self.prepare_inputs(input, input_lengths)
        parameter = next(self.upstream.parameters())
        features = features.to(device=parameter.device, dtype=parameter.dtype)
        attention_mask = attention_mask.to(parameter.device)
        frame_lengths = attention_mask.sum(-1).long()
        if self.frozen:
            self.upstream.eval()
        with torch.no_grad() if self.frozen else contextlib.nullcontext():
            output = self.upstream(input_features=features,
                                   attention_mask=attention_mask,
                                   output_hidden_states=True,
                                   return_dict=True)
        layerwise = torch.stack(output.hidden_states,
                                dim=-1).permute(0, 2, 1, 3)
        layerwise = layerwise.masked_fill(
            ~attention_mask.bool()[:, None, :, None], 0.0)
        return layerwise, [frame_lengths]
