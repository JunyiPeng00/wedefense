# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#               2022 Hongji Wang (jijijiang77@gmail.com)
#               2023 Zhengyang Chen (chenzhengyang117@gmail.com)
#               2025 Lin Zhang (partialspoof@gmail.com)
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

import functools
import io
import kaldiio
import json
import logging
import random
import tarfile
from subprocess import PIPE, Popen
from urllib.parse import urlparse

import numpy as np
from scipy import signal
from scipy.io import wavfile
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import torchaudio.transforms as torchaudio_T

import wedefense.dataset.augmentation.rawboost_util as rawboost_util
import wedefense.dataset.augmentation.codec_util as codec_util
import wedefense.dataset.augmentation.event_util as event_util
import wedefense.dataset.augmentation.opus_util as opus_util

import lmdb

AUDIO_FORMAT_SETS = set(['flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'])


def url_opener(data):
    """ Give url or local file, return file descriptor
        Inplace operation.

        Args:
            data(Iterable[str]): url or local file list

        Returns:
            Iterable[{src, stream}]
    """
    for sample in data:
        assert 'src' in sample
        # TODO(Binbin Zhang): support HTTP
        url = sample['src']
        try:
            pr = urlparse(url)
            # local file
            if pr.scheme == '' or pr.scheme == 'file':
                stream = open(url, 'rb')
            # network file, such as HTTP(HDFS/OSS/S3)/HTTPS/SCP
            else:
                cmd = f'wget -q -O - {url}'
                process = Popen(cmd, shell=True, stdout=PIPE)
                sample.update(process=process)
                stream = process.stdout
            sample.update(stream=stream)
            yield sample
        except Exception as ex:
            logging.warning('Failed to open {}'.format(url))


def tar_file_and_group(data):
    """ Expand a stream of open tar files into a stream of tar file contents.
        And groups the file with same prefix

        Args:
            data: Iterable[{src, stream}]

        Returns:
            Iterable[{key, wav, lab, sample_rate}]
    """
    for sample in data:
        assert 'stream' in sample
        stream = tarfile.open(fileobj=sample['stream'], mode="r:*")
        prev_prefix = None
        example = {}
        valid = True
        for tarinfo in stream:
            name = tarinfo.name
            pos = name.rfind('.')
            assert pos > 0
            prefix, postfix = name[:pos], name[pos + 1:]
            if prev_prefix is not None and prefix != prev_prefix:
                example['key'] = prev_prefix
                if valid:
                    yield example
                example = {}
                valid = True
            with stream.extractfile(tarinfo) as file_obj:
                try:
                    if postfix in ['lab']:
                        example[postfix] = file_obj.read().decode(
                            'utf8').strip()
                    elif postfix in AUDIO_FORMAT_SETS:
                        waveform, sample_rate = torchaudio.load(file_obj)
                        example['wav'] = waveform
                        example['sample_rate'] = sample_rate
                    else:
                        example[postfix] = file_obj.read()
                except Exception as ex:
                    valid = False
                    logging.warning('error to parse {}'.format(name))
            prev_prefix = prefix
        if prev_prefix is not None:
            example['key'] = prev_prefix
            yield example
        stream.close()
        if 'process' in sample:
            sample['process'].communicate()
        sample['stream'].close()


def parse_raw(data, lmdb_file_path=None):
    """ Parse key/wav/lab from json line

        Args:
            data: Iterable[str], str is a json line has key/wav/lab
            lmdb_file_path (str, optional): Path to LMDB file containing audio data. 
                If provided, audio will be loaded from LMDB using "wav" as the key.
        Returns:
            Iterable[{key, wav, lab, sample_rate}]
    """

    def read_audio(wav, lmdb_file_path):
        if lmdb_file_path is not None:
            with lmdb.open(lmdb_file_path, readonly=True, lock=False) as env:
                with env.begin() as txn:
                    data = txn.get(wav.encode())
                    waveform, sample_rate = torchaudio.load(io.BytesIO(data))
        else:
            if wav.endswith('|'):
                p = Popen(wav[:-1], shell=True, stdout=PIPE)
                data = p.stdout.read()
                waveform, sample_rate = torchaudio.load(io.BytesIO(data))
            else:
                waveform, sample_rate = torchaudio.load(wav)
        return waveform, sample_rate

    def apply_vad(waveform, sample_rate, vad):
        voice_part_list = []
        for start, end in vad:
            start, end = float(start), float(end)
            start, end = int(start * sample_rate), int(end * sample_rate)
            voice_part_list.append(waveform[:, start:end])
        waveform = torch.cat(voice_part_list, dim=1)
        return waveform, sample_rate

    for sample in data:
        assert 'src' in sample
        json_line = sample['src']
        obj = json.loads(json_line)
        assert 'key' in obj
        assert 'wav' in obj
        assert 'lab' in obj
        key = obj['key']
        wav_file = obj['wav']
        lab = obj['lab']
        try:
            waveform, sample_rate = read_audio(wav_file, lmdb_file_path)
            if 'vad' in obj:
                waveform, sample_rate = apply_vad(waveform, sample_rate,
                                                  obj['vad'])
            example = dict(key=key,
                           lab=lab,
                           wav=waveform,
                           sample_rate=sample_rate)
            yield example
        except Exception as ex:
            logging.warning('Failed to read {}'.format(wav_file))


def parse_feat(data):
    """ Parse key/feat/lab from json line

        Args:
            data: Iterable[str], str is a json line has key/feat/lab

        Returns:
            Iterable[{key, feat, lab}]
    """
    for sample in data:
        assert 'src' in sample
        json_line = sample['src']
        obj = json.loads(json_line)
        assert 'key' in obj
        assert 'feat' in obj
        assert 'lab' in obj
        key = obj['key']
        feat_ark = obj['feat']
        lab = obj['lab']
        try:
            feat = torch.from_numpy(kaldiio.load_mat(feat_ark))
            example = dict(key=key, lab=lab, feat=feat)
            yield example
        except Exception as ex:
            logging.warning('Failed to load {}'.format(feat_ark))


def shuffle(data, shuffle_size=2500):
    """ Local shuffle the data

        Args:
            data: Iterable[{key, wav/feat, lab}]
            shuffle_size: buffer size for shuffle

        Returns:
            Iterable[{key, wav/feat, lab}]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= shuffle_size:
            random.shuffle(buf)
            for x in buf:
                yield x
            buf = []
    # The sample left over
    random.shuffle(buf)
    for x in buf:
        yield x


def lab_to_id(data, lab2id):
    """ Parse lab id

        Args:
            data: Iterable[{key, wav/feat, lab}]
            lab2id: Dict[str, int] i.e.:{'bonafide': 1, 'spoof': 0}

        Returns:
            Iterable[{key, wav/feat, label}]
    """
    for sample in data:
        assert 'lab' in sample
        if sample['lab'] in lab2id:
            label = lab2id[sample['lab']]
        else:
            label = -1
        sample['label'] = label
        yield sample


def resample(data, resample_rate=16000):
    """ Resample data.
        Inplace operation.
        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            resample_rate: target resample rate
        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        if sample_rate != resample_rate:
            sample['sample_rate'] = resample_rate
            sample['wav'] = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=resample_rate)(waveform)
        yield sample


def speed_perturb(data, num_labs):
    """ Apply speed perturb to the data.
        Inplace operation.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    speeds = [1.0, 0.9, 1.1]
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        speed_idx = random.randint(0, 2)
        if speed_idx > 0:
            wav, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform, sample_rate,
                [['speed', str(speeds[speed_idx])], ['rate',
                                                     str(sample_rate)]])
            sample['wav'] = wav
            sample['label'] = sample['label'] + num_labs * speed_idx

        yield sample


def get_random_chunk(data, chunk_len):
    """ Get random chunk

        Args:
            data: torch.Tensor (random len)
            chunk_len: chunk length

        Returns:
            torch.Tensor (exactly chunk_len)
    """
    data_len = len(data)
    data_shape = data.shape
    # random chunk
    if data_len >= chunk_len:
        chunk_start = random.randint(0, data_len - chunk_len)
        data = data[chunk_start:chunk_start + chunk_len]
        # re-clone the data to avoid memory leakage
        if type(data) == torch.Tensor:
            data = data.clone()
        else:  # np.array
            data = data.copy()
    else:
        # padding
        repeat_factor = chunk_len // data_len + 1
        repeat_shape = repeat_factor if len(data_shape) == 1 else (
            repeat_factor, 1)
        if type(data) == torch.Tensor:
            data = data.repeat(repeat_shape)
        else:  # np.array
            data = np.tile(data, repeat_shape)
        data = data[:chunk_len]

    return data


def filter(data,
           min_num_frames=100,
           max_num_frames=800,
           frame_shift=10,
           data_type='shard/raw/feat'):
    """ Filter the utterance with very short duration and random chunk the
        utterance with very long duration.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            min_num_frames: minimum number of frames of acoustic features
            max_num_frames: maximum number of frames of acoustic features
            frame_shift: the frame shift of the acoustic features (ms)
        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert 'key' in sample

        if data_type == 'feat':
            assert 'feat' in sample
            feat = sample['feat']
            if len(feat) < min_num_frames:
                continue
            elif len(feat) > max_num_frames:
                feat = get_random_chunk(feat, max_num_frames)
            sample['feat'] = feat
        else:
            assert 'sample_rate' in sample
            assert 'wav' in sample
            sample_rate = sample['sample_rate']
            wav = sample['wav'][0]

            min_len = int(frame_shift / 1000 * min_num_frames * sample_rate)
            max_len = int(frame_shift / 1000 * max_num_frames * sample_rate)

            if len(wav) < min_len:
                continue
            elif len(wav) > max_len:
                wav = get_random_chunk(wav, max_len)
            sample['wav'] = wav.unsqueeze(0)

        yield sample


def random_chunk(data, chunk_len, data_type='shard/raw/feat'):
    """ Random chunk the data into chunk_len

        Args:
            data: Iterable[{key, wav/feat, label}]
            chunk_len: chunk length for each sample

        Returns:
            Iterable[{key, wav/feat, label}]
    """
    for sample in data:
        assert 'key' in sample

        if data_type == 'feat':
            assert 'feat' in sample
            feat = sample['feat']
            feat = get_random_chunk(feat, chunk_len)
            sample['feat'] = feat
        else:
            assert 'wav' in sample
            wav = sample['wav'][0]
            wav = get_random_chunk(wav, chunk_len)
            sample['wav'] = wav.unsqueeze(0)
        yield sample


def add_reverb_noise(data,
                     reverb_source,
                     noise_source,
                     resample_rate=16000,
                     aug_prob=0.6):
    """ Add reverb & noise aug

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            reverb_source: reverb LMDB data source
            noise_source: noise LMDB data source
            resample_rate: resample rate for reverb/noise data
            aug_prob: aug probability

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert 'wav' in sample
        assert 'key' in sample
        if aug_prob > random.random():
            aug_type = random.randint(1, 2)
            if aug_type == 1:
                # add reverberation
                audio = sample['wav'].numpy()[0]
                audio_len = audio.shape[0]

                _, rir_data = reverb_source.random_one()
                rir_sr, rir_audio = wavfile.read(io.BytesIO(rir_data))
                rir_audio = rir_audio.astype(np.float32)
                if rir_sr != resample_rate:
                    rir_audio = signal.resample(
                        rir_audio,
                        int(len(rir_audio) / rir_sr * resample_rate))
                rir_audio = rir_audio / np.sqrt(np.sum(rir_audio**2))
                # print(sample['key'], audio.shape, rir_audio.shape, audio_len)
                out_audio = signal.convolve(audio, rir_audio,
                                            mode='full')[:audio_len]
            else:
                # add additive noise
                audio = sample['wav'].numpy()[0]
                audio_len = audio.shape[0]
                audio_db = 10 * np.log10(np.mean(audio**2) + 1e-4)

                key, noise_data = noise_source.random_one()
                if key.startswith('noise'):
                    snr_range = [0, 15]
                elif key.startswith('speech'):
                    snr_range = [10, 30]
                elif key.startswith('music'):
                    snr_range = [5, 15]
                else:
                    snr_range = [0, 15]
                noise_sr, noise_audio = wavfile.read(io.BytesIO(noise_data))
                noise_audio = noise_audio.astype(np.float32) / (1 << 15)
                if noise_sr != resample_rate:
                    # Since the noise audio could be very long, it must be
                    # chunked first before resampled (to save time)
                    noise_audio = get_random_chunk(
                        noise_audio, int(audio_len / resample_rate * noise_sr))
                    noise_audio = signal.resample(noise_audio, audio_len)
                else:
                    noise_audio = get_random_chunk(noise_audio, audio_len)
                noise_snr = random.uniform(snr_range[0], snr_range[1])
                noise_db = 10 * np.log10(np.mean(noise_audio**2) + 1e-4)
                noise_audio = np.sqrt(10**(
                    (audio_db - noise_db - noise_snr) / 10)) * noise_audio
                out_audio = audio + noise_audio

            # normalize into [-1, 1]
            out_audio = out_audio / (np.max(np.abs(out_audio)) + 1e-4)
            sample['wav'] = torch.from_numpy(out_audio).unsqueeze(0)

        yield sample


def compute_fbank(data,
                  num_mel_bins=80,
                  frame_length=25,
                  frame_shift=10,
                  dither=1.0):
    """ Extract fbank

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label, sample_rate}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        assert 'key' in sample
        assert 'label' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        waveform = waveform * (
            1 << 15)  # Convert from float [-1.0, 1.0] to int16 [-32767, 32767]
        # Only keep key, feat, label
        mat = kaldi.fbank(waveform,
                          num_mel_bins=num_mel_bins,
                          frame_length=frame_length,
                          frame_shift=frame_shift,
                          dither=dither,
                          sample_frequency=sample_rate,
                          window_type='hamming',
                          use_energy=False)  # [T, F]
        yield dict(key=sample['key'], label=sample['label'], feat=mat)


def compute_lfcc_torchaudio(data,
                            n_fft=512,
                            n_lfcc=20,
                            frame_length=20,
                            frame_shift=10,
                            use_delta=True):
    """ Extract linear frequency cepstral coefficients

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label, sample_rate}]
    """
    compute_delta = torchaudio_T.ComputeDeltas()
    # Initialize LFCC feature extractor (only once for efficiency)
    # Assume all samples share the same sample rate; extract from the first sample  # noqa
    # data_lst = list(data) Too slow
    # sample_rate = data_lst[0]['sample_rate']
    from itertools import chain
    first_sample = next(data)
    data = chain([first_sample], data)
    sample_rate = first_sample['sample_rate']
    lfcc_extractor = torchaudio_T.LFCC(
        sample_rate=sample_rate,
        n_lfcc=n_lfcc,
        speckwargs={
            "n_fft": n_fft,
            "win_length": int(frame_length * sample_rate / 1000),
            "hop_length": int(frame_shift * sample_rate / 1000),
            "center": False
        })

    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        assert 'key' in sample
        assert 'label' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        waveform = waveform * (
            1 << 15)  # Convert from float [-1.0, 1.0] to int16 [-32767, 32767]
        # Only keep key, feat, label

        mat = lfcc_extractor(waveform).squeeze(0).permute(
            1, 0)  # doesn't support energy #[1, F, T] -> [T, F]
        if (use_delta):
            lfcc_delta = compute_delta(mat)
            lfcc_delta_delta = compute_delta(lfcc_delta)
            mat = torch.cat((mat, lfcc_delta, lfcc_delta_delta), 1)

        yield dict(key=sample['key'], label=sample['label'], feat=mat)


def apply_cmvn(data, norm_mean=True, norm_var=False):
    """ Apply CMVN

        Args:
            data: Iterable[{key, feat, label}]

        Returns:
            Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'key' in sample
        assert 'feat' in sample
        assert 'label' in sample
        mat = sample['feat']
        if norm_mean:
            mat = mat - torch.mean(mat, dim=0)
        if norm_var:
            mat = mat / torch.sqrt(torch.var(mat, dim=0) + 1e-8)
        yield dict(key=sample['key'], label=sample['label'], feat=mat)


def spec_aug(data, num_t_mask=1, num_f_mask=1, max_t=10, max_f=8, prob=0.6):
    """ Do spec augmentation
        Inplace operation

        Args:
            data: Iterable[{key, feat, label}]
            num_t_mask: number of time mask to apply
            num_f_mask: number of freq mask to apply
            max_t: max width of time mask
            max_f: max width of freq mask
            prob: prob of spec_aug

        Returns
            Iterable[{key, feat, label}]
    """
    for sample in data:
        if random.random() < prob:
            assert 'feat' in sample
            x = sample['feat']
            assert isinstance(x, torch.Tensor)
            # y = x.clone().detach()
            y = x.detach()  # inplace operation
            max_frames = y.size(0)
            max_freq = y.size(1)
            # time mask
            for i in range(num_t_mask):
                start = random.randint(0, max_frames - 1)
                length = random.randint(1, max_t)
                end = min(max_frames, start + length)
                y[start:end, :] = 0
            # freq mask
            for i in range(num_f_mask):
                start = random.randint(0, max_freq - 1)
                length = random.randint(1, max_f)
                end = min(max_freq, start + length)
                y[:, start:end] = 0
            sample['feat'] = y
        yield sample


def rawboost(data, algo=5):
    """ Process Rawboost

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            resample_rate: resample rate for data
            argo: argo. tag for rawboost, default value is 5.

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        assert 'key' in sample
        sample_rate = sample['sample_rate']
        audio = sample['wav'].numpy()[0]
        device = sample['wav'].device
        audio_len = audio.shape[0]

        args = rawboost_util.get_args_for_rawboost()
        # args = argparse.Namespace(
        #        nBands=5, minF=20, maxF=8000, minBW=100, maxBW=1000,
        #        minCoeff=10, maxCoeff=100, minG=0, maxG=0,
        #        minBiasLinNonLin=5, maxBiasLinNonLin=20, N_f=5,
        #        P=10, g_sd=2, SNRmin=10, SNRmax=40)

        out_audio = rawboost_util.process_Rawboost_feature(
            audio, sample_rate, args, algo)
        # sample['wav'] = torch.from_numpy(out_audio).float().to(device)
        sample['wav'] = torch.from_numpy(out_audio).unsqueeze(0).to(device)

        yield sample


def codec(data, random_seed=42, formats=None, prob=1.0):
    """ Process codec

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            random_seed (int): random seed for random(), default 42;
                None leaves the random generator as it is
            formats (list): codecs to draw from, default
                codec_util.SUPPORTED_CODEC_FOR_AUGMENTATION
            prob (float): probability of coding a sample

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    if random_seed is not None:
        random.seed(random_seed)
    choices = list(
        formats) if formats else codec_util.SUPPORTED_CODEC_FOR_AUGMENTATION

    for sample in data:
        if prob < 1.0 and random.random() >= prob:
            yield sample
            continue
        assert 'sample_rate' in sample
        assert 'wav' in sample
        assert 'key' in sample

        sr = sample['sample_rate']
        audio = sample['wav']

        codec_name = random.choice(choices)
        bitrate = random.choice(codec_util.SUPPORTED_CODEC_CONFIG[codec_name])

        sample['wav'] = codec_util.codec_apply(audio, sr, codec_name, bitrate)
        # print(codec_name, bitrate, audio.shape, sample['wav'].shape, type(audio), type(sample['wav']))  # noqa
        yield sample


def event_aug(data,
              plan_path,
              prob=0.3,
              snr_range=(6, 18),
              clip_prob=0.1,
              clip_gain_db=(1, 4),
              max_cache=64):
    """ Add the sparse sound events of one family, and clip, independently
        of each other

        Args:
            data: Iterable[{key, wav, label, sample_rate}], 16 kHz
            plan_path: JSON list of event families, see event_util.load_plan
            prob: probability of adding events
            snr_range: range of the local event SNR in dB
            clip_prob: probability of clipping
            clip_gain_db: range of the gain above full scale in dB
            max_cache: number of decoded event files kept per worker

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    families = event_util.load_plan(plan_path)
    weights = [family['prob'] for family in families]
    loader = functools.lru_cache(maxsize=max_cache)(event_util.load_audio16)
    # seeded when the worker builds the pipeline, before the first sample
    rng = np.random.default_rng(random.getrandbits(32))

    def generate():
        for sample in data:
            event = prob > 0 and random.random() < prob
            clip = clip_prob > 0 and random.random() < clip_prob
            if not event and not clip:
                yield sample
                continue
            x = sample['wav'].detach().cpu().numpy()[0].astype(np.float32,
                                                               copy=False)
            if event:
                family = random.choices(families, weights=weights, k=1)[0]
                x = event_util.add_events(x, dict(family, snr=snr_range), rng,
                                          loader)
            if clip:
                gain = random.uniform(*clip_gain_db)
                peak = float(np.max(np.abs(x)))
                if peak > 0:
                    x = np.clip((x / peak) * 10**(gain / 20), -1, 1)
            sample['wav'] = torch.from_numpy(np.asarray(
                x, dtype=np.float32)).unsqueeze(0)
            yield sample

    return generate()


_AUG_FAILURES = {}


def _aug_failure(name, ex, limit=5):
    """ Log the first failure of an augmentation in this worker; the sample
        is passed through. The worker stops at the `limit`-th failure.
    """
    n = _AUG_FAILURES.get(name, 0) + 1
    _AUG_FAILURES[name] = n
    if n == 1:
        logging.warning(
            '%s: augmentation failed (%s: %s); the sample is passed through',
            name,
            type(ex).__name__,
            str(ex)[:200])
    if n >= limit:
        raise RuntimeError(
            f'{name}: {n} augmentation failures in this worker (last: {type(ex).__name__}: {str(ex)[:200]})'
        )


def _pitch_shift(x, semitones, sr=16000):
    """ Pitch shift by `semitones`, length preserved """
    import librosa
    y = librosa.effects.pitch_shift(x.astype(np.float32),
                                    sr=sr,
                                    n_steps=float(semitones),
                                    bins_per_octave=12)
    n = len(x)
    return (y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))).astype(
        np.float32)


def diverse_aug(data,
                prob=0.5,
                n_ops=(1, 2),
                noise_source=None,
                sample_rate=16000,
                lp_range=(2200, 7200),
                hp_range=(70, 360),
                loss_range=(1, 10),
                pitch_semitones=1.0,
                mask_frac=(0.01, 0.10),
                noise_snr=(5, 30),
                ops=('bandlimit', 'packetloss', 'pitch', 'timemask', 'noise')):
    """ With probability `prob` apply n_ops random operations out of:
        band limit (low-pass 2.2-7.2 kHz | high-pass 70-360 Hz | band-pass
        300-4000 Hz), Opus coding with 1-10 % bursty packet loss concealed by
        the decoder, pitch shift within one semitone, a zeroed span of 1-10 %
        of the samples, noise at 5-30 dB from the noise LMDB

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            prob: probability of augmenting a sample
            n_ops: range of the number of operations
            noise_source: LmdbData of noise files (keys starting with
                'speech' are skipped); without it there is no noise operation
            noise_snr: SNR range of the noise in dB

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    ops = list(ops)
    if noise_source is None:
        ops = [o for o in ops if o != 'noise']
    for sample in data:
        if random.random() >= prob:
            yield sample
            continue
        wav = sample['wav']
        x = wav[0].detach().cpu().numpy().astype(np.float32)
        T = len(x)
        k = random.randint(n_ops[0], n_ops[1])
        chosen = random.sample(ops, min(k, len(ops)))
        try:
            for op in chosen:
                if op == 'bandlimit':
                    kind = random.choice(['lp', 'hp', 'bp'])
                    if kind == 'lp':
                        sos = signal.butter(6,
                                            random.uniform(*lp_range),
                                            'lowpass',
                                            fs=sample_rate,
                                            output='sos')
                    elif kind == 'hp':
                        sos = signal.butter(4,
                                            random.uniform(*hp_range),
                                            'highpass',
                                            fs=sample_rate,
                                            output='sos')
                    else:
                        sos = signal.butter(4, [300, 4000],
                                            'bandpass',
                                            fs=sample_rate,
                                            output='sos')
                    x = signal.sosfilt(sos, x).astype(np.float32)
                elif op == 'packetloss':
                    rng = np.random.default_rng(random.getrandbits(32))
                    # target loss rate
                    L = random.uniform(*loss_range) / 100.0
                    # a lost packet is followed by another one with
                    # probability 0.5: entering a loss with r % gives L
                    r = 100.0 * 0.5 * L / (1.0 - L)
                    x = opus_util.opus_packet_loss(
                        x, int(rng.integers(12000, 32001)), float(r),
                        rng).astype(np.float32)
                elif op == 'pitch':
                    x = _pitch_shift(
                        x, random.uniform(-pitch_semitones, pitch_semitones),
                        sample_rate)
                elif op == 'timemask':
                    L = int(random.uniform(*mask_frac) * T)
                    s = random.randint(0, max(0, T - L))
                    x = x.copy()
                    x[s:s + L] = 0.0
                elif op == 'noise':
                    key, nd = noise_source.random_one()
                    tries = 0
                    while key.startswith('speech') and tries < 20:
                        key, nd = noise_source.random_one()
                        tries += 1
                    if key.startswith('speech'):
                        continue
                    nsr, na = wavfile.read(io.BytesIO(nd))
                    na = na.astype(np.float32) / (1 << 15)
                    if nsr != sample_rate:
                        na = get_random_chunk(na, int(T / sample_rate * nsr))
                        na = signal.resample(na, T)
                    else:
                        na = get_random_chunk(na, T)
                    snr = random.uniform(*noise_snr)
                    pa = np.mean(x**2) + 1e-8
                    pn = np.mean(na**2) + 1e-8
                    x = x + na * np.sqrt(pa / pn / (10**(snr / 10)))
        except Exception as ex:
            _aug_failure('diverse_aug', ex)
            yield sample
            continue
        if len(x) < T:
            x = np.concatenate([x, np.zeros(T - len(x), dtype=np.float32)])
        x = np.clip(x[:T], -1.0, 1.0).astype(np.float32)
        sample['wav'] = torch.from_numpy(x).unsqueeze(0).to(wav.dtype)
        yield sample


# TODO
# codec
# Rawboost
