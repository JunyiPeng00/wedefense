# Copyright (c) 2021 Hongji Wang (jijijiang77@gmail.com)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
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

import copy
import os

import fire
import logging
import kaldiio
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from wedefense.dataset.dataset import Dataset
from wedefense.dataset.dataset_utils import apply_cmvn, spec_aug
from wedefense.frontend import *
from wedefense.models.get_model import get_model
from wedefense.utils.checkpoint import load_checkpoint
from wedefense.utils.utils import parse_config_or_kwargs, validate_path

# Optional (lazy) imports for quantization; we import inside functions to avoid hard dep
try:
    from wedefense.frontend.wav2vec2 import (
        apply_quantization_with_hp_integration,
        get_quantization_config,
    )
except Exception:  # pragma: no cover - optional at runtime
    apply_quantization_with_hp_integration = None
    get_quantization_config = None


def extract(config='conf/config.yaml', **kwargs):
    """Extract embeddings with optional LSQ quantization (and pruning-aware integration).

    Parameters
    ----------
    config : str
        Path to YAML config file.
    **kwargs : Any
        Overrides for config values.

    Notes
    -----
    If configs contain `use_quantization: True`, the model will be wrapped with
    LSQ quantized layers using the same configuration keys as training, e.g.:
      - `quantization_config` (e.g., '8bit_symmetric', '4bit_symmetric')
      - `quantize_weights`, `quantize_activations`, `quantize_bias`
      - `per_channel_weights`, `per_channel_activations`
      - Optional pruning context via `use_pruning_loss` and sparsity knobs to
        correctly integrate HP gating if needed during export/inference.
    """
    # parse configs first
    configs = parse_config_or_kwargs(config, **kwargs)

    model_path = configs['model_path']
    embed_ark = configs['embed_ark']
    batch_size = configs.get('batch_size', 1)
    num_workers = configs.get('num_workers', 1)

    # Since the input length is not fixed, we set the built-in cudnn
    # auto-tuner to False
    torch.backends.cudnn.benchmark = False

    test_conf = copy.deepcopy(configs['dataset_args'])
    # model: frontend (optional) => speaker model
    model = get_model(configs['model'])(**configs['model_args'])
    frontend_type = test_conf.get('frontend', 'fbank')
    if frontend_type != "fbank" and not frontend_type.startswith('lfcc'):
        frontend_args = frontend_type + "_args"
        # frontends besides acoustic features, like s3prl
        print('Initializing frontend model (this could take some time) ...')
        frontend = frontend_class_dict[frontend_type](
            **test_conf[frontend_args], sample_rate=test_conf['resample_rate'])
        model.add_module("frontend", frontend)
    # Apply quantization (optional) BEFORE loading checkpoint so parameter shapes match
    use_quantization = configs.get('use_quantization', False)
    if use_quantization:
        if apply_quantization_with_hp_integration is None or get_quantization_config is None:
            logging.warning("Quantization modules unavailable; continuing without quantization.")
        else:
            try:
                logging.info("<== Quantization (extract) ==> applying LSQ wrappers")
                quant_config = get_quantization_config(configs.get('quantization_config', '8bit_symmetric'))
                if configs.get('quantize_weights', True):
                    quant_config.quantize_weights = True
                if configs.get('quantize_activations', False):
                    quant_config.quantize_activations = True
                # Optional knobs
                quant_config.per_channel_weights = configs.get('per_channel_weights', True)
                quant_config.per_channel_activations = configs.get('per_channel_activations', False)
                if 'quantize_bias' in configs:
                    quant_config.quantize_bias = configs['quantize_bias']

                # Provide pruning context for proper HP gating preservation
                use_pruning = configs.get('use_pruning_loss', False)
                pruning_config = {
                    'target_sparsity': configs.get('target_sparsity', 0.0),
                    'sparsity_warmup_epochs': configs.get('sparsity_warmup_epochs', 0),
                    'sparsity_schedule': configs.get('sparsity_schedule', 'cosine'),
                    'min_sparsity': configs.get('min_sparsity', 0.0),
                }

                model = apply_quantization_with_hp_integration(
                    model,
                    config=quant_config,
                    enable_pruning=use_pruning,
                    pruning_config=pruning_config,
                )
                logging.info("Quantization wrappers applied successfully (extract)")
            except Exception as e:  # pragma: no cover
                logging.error(f"Failed to apply quantization in extract: {e}")
                logging.warning("Continuing without quantization...")

    print('Loading checkpoint ...')
    load_checkpoint(model, model_path)
    print('Finished !!! Start extracting ...')
    device = torch.device("cuda")
    model.to(device).eval()

    # test_configs
    # test_conf = copy.deepcopy(configs['dataset_args'])
    test_conf['speed_perturb'] = False
    if 'fbank_args' in test_conf:
        test_conf['fbank_args']['dither'] = 0.0
    test_conf['spec_aug'] = False
    test_conf['shuffle'] = False
    test_conf['aug_prob'] = configs.get('aug_prob', 0.0)
    test_conf['filter'] = False
    test_conf['codec_aug'] = False
    test_conf['rawboost'] = False

    dataset = Dataset(configs['data_type'],
                      configs['data_list'],
                      test_conf,
                      lab2id_dict={},
                      whole_utt=(batch_size == 1),
                      train_lmdb_file=configs.get('data_lmdb', None),
                      reverb_lmdb_file=configs.get('reverb_data', None),
                      noise_lmdb_file=configs.get('noise_data', None),
                      repeat_dataset=False)
    dataloader = DataLoader(dataset,
                            shuffle=False,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            prefetch_factor=4)

    validate_path(embed_ark)
    embed_ark = os.path.abspath(embed_ark)
    embed_scp = embed_ark[:-3] + "scp"

    with torch.no_grad():
        with kaldiio.WriteHelper('ark,scp:' + embed_ark + "," +
                                 embed_scp) as writer:
            for _, batch in tqdm(enumerate(dataloader)):
                utts = batch['key']
                if frontend_type == 'fbank' or frontend_type.startswith(
                        'lfcc'):
                    features = batch['feat']
                    features = features.float().to(device)  # (B,T,F)
                else:  # 's3prl'
                    wavs = batch['wav']  # (B,1,W)
                    wavs = wavs.squeeze(1).float().to(device)  # (B,W)
                    wavs_len = torch.LongTensor([wavs.shape[1]]).repeat(
                        wavs.shape[0]).to(device)  # (B)
                    features, _ = model.frontend(wavs, wavs_len)

                # apply cmvn
                if test_conf.get('cmvn', True):
                    features = apply_cmvn(features,
                                          **test_conf.get('cmvn_args', {}))
                # spec augmentation
                if test_conf.get('spec_aug', False):
                    features = spec_aug(features, **test_conf['spec_aug_args'])

                # Forward through model
                outputs = model(features)  # embed or (embed_a, embed_b)
                embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
                embeds = embeds.cpu().detach().numpy()  # (B,F)

                for i, utt in enumerate(utts):
                    embed = embeds[i]
                    writer(utt, embed)


if __name__ == '__main__':
    fire.Fire(extract)
