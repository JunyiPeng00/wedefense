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
"""Opus coding with packet loss (opuslib, system library libopus)."""

import numpy as np


def opus_packet_loss(wav, bitrate, loss_pct, rng, sample_rate=16000):
    """ Opus VoIP coding in 20 ms packets with in-band FEC and bursty packet
        loss, concealed by the decoder

        Args:
            wav (np.ndarray): waveform in [-1, 1], shape [time]
            bitrate (int): bits per second
            loss_pct (float): probability in percent of losing a packet after
                a received one; after a lost one it is 50 %
            rng (np.random.Generator): draws the losses
        Return:
            np.ndarray: decoded waveform, a whole number of packets
    """
    import opuslib
    from opuslib.api import ctl, encoder

    enc = opuslib.Encoder(sample_rate, 1, opuslib.APPLICATION_VOIP)
    for request, value in ((ctl.set_bitrate, bitrate),
                           (ctl.set_packet_loss_perc, loss_pct),
                           (ctl.set_inband_fec, 1), (ctl.set_dtx, 0)):
        encoder.encoder_ctl(enc.encoder_state, request, int(value))
    dec = opuslib.Decoder(sample_rate, 1)

    frame = sample_rate // 50
    pcm = np.clip(wav * 32768.0, -32768, 32767).astype(np.int16)
    out, lost = [], False
    for i in range(0, len(pcm) - frame + 1, frame):
        packet = enc.encode(pcm[i:i + frame].tobytes(), frame)
        lost = rng.random() < (0.5 if lost else loss_pct / 100.0)
        out.append(dec.decode(b'' if lost else packet, frame))
    return np.frombuffer(b''.join(out),
                         dtype=np.int16).astype(np.float32) / 32768.0
