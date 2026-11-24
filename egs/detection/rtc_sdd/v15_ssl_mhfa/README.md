# RTC-SDD: w2v-BERT 2.0 + MHFA

A recipe for the RTC-SDD challenge (spoofed-speech detection under real-time-communication conditions): data
processing, training and testing of a detector made of the w2v-BERT 2.0 encoder (fine-tuned, all 25 hidden states),
an MHFA back-end and a 2-class head, on the challenge's labelled train and dev sets. Data: [DATA.md](DATA.md).

## Requirements
- The toolkit's `requirements.txt`; versions used here (Python 3.9.19): torch 2.8.0 (CUDA 12.8), torchaudio 2.8.0
  (a release that still has `torchaudio.io`), transformers 4.56.2, numpy 1.22.4, scipy 1.10.0, scikit-learn 1.5.1,
  soundfile 0.12.1, kaldiio 2.17.2, lmdb 1.3.0, librosa 0.11.0, opuslib 3.0.1, webrtc-noise-gain 1.3.0, wandb 0.21.1,
  s3prl 0.4.16 (imported by the toolkit for every config).
- Binaries: `ffmpeg` on `PATH` (stage 2); the FFmpeg libraries loaded by torchaudio, with opus, g722 and mu-law
  encoders (codec augmentation); `libopus` (opuslib); `lsof`. Logging: `wandb login` or `export WANDB_MODE=offline`.
- Encoder: three files of [facebook/w2v-bert-2.0](https://huggingface.co/facebook/w2v-bert-2.0) (2.3 GB):
  `huggingface-cli download facebook/w2v-bert-2.0 config.json preprocessor_config.json model.safetensors
  --revision da985ba0987f70aaeb84a80f2851cfac8c697a7b --local-dir pretrained_models/w2v-bert-2.0`
- Optional second encoder (`conf/xlsr1b_mhfa.yaml`): on first use s3prl downloads XLS-R 1B (3.9 GB) into
  `pretrained_models/s3prl_hub` from `https://huggingface.co/s3prl/converted_ckpts/resolve/main/xlsr2_960m_1000k.pt`
  (offline: save it there as `c5e61efc17b34e9f05ec4d5cf15cbf27c44913f9b56429ac732e096df2daa3f4.xlsr2_960m_1000k.pt`).
- Optional third encoder (`conf/omniw2v1b_mhfa.yaml`; 3.9 GB): `huggingface-cli download ylacombe/omniASR_W2V_1B_SSL
  --revision 49be43063f05cfeb664d6ae1ff46e2b6c7d4f822 --local-dir pretrained_models/omniASR_W2V_1B_SSL`
- Data: the challenge's train and dev sets; MUSAN, RIRS_NOISES, ESC-50, FSD50K, RNNoise noise recordings (DATA.md).

## Stages of `run.sh`
| stage | what | output |
|---|---|---|
| 1 | train and dev lists, held-out part of dev | `data/{train,dev}_{offline,online}`, `data/dev_heldout` |
| 2 | noise lists; channel simulation of the offline recordings | `data/noise`, `<sim_dir>/<subset>_<preset>` |
| 3 | training and validation lists, noise LMDB | `data/train`, `data/val`, `data/noise/musan_lmdb` |
| 4 | training (one call per training run) | `<exp_dir>/models/model_<epoch>.pt` |
| 5 | embeddings, posteriors, score file `<utt> <P(spoof)>` | `<exp_dir>/scores/<set>_ep<epoch>.txt` |
| 6 | Macro-F1 at threshold 0.5, overall and per directory | printed |
| 7 | scores with the test-time recipe, Macro-F1 | `<exp_dir>/scores/<set>_ep<epoch>_recipe.txt` |
| 8 | teacher scores of audio without labels, fused score | `data/<unlab_set>/fused_scores.txt` |
| 9 | training list with the teacher-labelled rows | `data/train_student` |

Stages 1-3: `bash run.sh --stage 1 --stop_stage 3 --rtcsdd_dir <dir> --sim_dir <dir> --musan_dir <dir>
--rirs_dir <dir> --esc50_dir <dir> --fsd50k_dir <dir> --rnnoise_dir <dir> [--nj 16]`

## Training
Three runs of stage 4; runs 2 and 3 start from `best_model.pt` of the previous one:
```bash
bash run.sh --stage 4 --stop_stage 4 --config conf/w2vbert_mhfa.yaml --exp_dir exp/w2vbert_mhfa --gpus "[0,1]"
bash run.sh --stage 4 --stop_stage 4 --config conf/w2vbert_mhfa_ft_aug.yaml --exp_dir exp/w2vbert_mhfa_ft_aug \
  --init_exp exp/w2vbert_mhfa --init_epoch best
bash run.sh --stage 4 --stop_stage 4 --config conf/w2vbert_mhfa_ft.yaml --exp_dir exp/w2vbert_mhfa_ft \
  --init_exp exp/w2vbert_mhfa_ft_aug --init_epoch best
```

| config | starts from | initial lr | epochs | in-loader augmentation | GPUs x accum_grad | input features |
|---|---|---|---|---|---|---|
| `w2vbert_mhfa.yaml` | pre-trained encoder | 1e-4 | 8 | events, codec, channel | 2 x 4 | CPU |
| `w2vbert_mhfa_ft_aug.yaml` | run 1 | 5e-5 | 4 | events, codec, channel | 1 x 8 | GPU |
| `w2vbert_mhfa_ft.yaml` | run 2 | 5e-5 | 3 | none | 1 x 8 | GPU |

An epoch is 132,813 random 4 s chunks at batch 4; an update uses 4 x accum_grad x GPUs = 32 utterances (change
`accum_grad` with the number of GPUs); the encoder is trained with 0.05 x the learning rate. With an initial model
the encoder and the back-end are loaded and the 2-class head starts from scratch. Every epoch is saved (2.3 GB);
`best_model.pt` is the epoch with the best group-weighted validation Macro-F1 (`val_metric`, on random 4 s chunks
of `data/val`). A run stops early after 3 validations in a row without a new best (`early_stop_patience`); that
epoch is saved as `final.pt` only. Resource use: peak GPU memory 12.4 GiB per GPU; 1.5-2.8 h per epoch on one 24 GB GPU.

Other configs (6 epochs, from the pre-trained encoder; on 24 GB GPUs: GPUs x accum_grad, peak memory per GPU, time per
epoch, saved epoch; test them with `--stage 5 --stop_stage 6`, stage 7 is for the three configs above):
- `conf/w2vbert_mhfa_u3xi.yaml`, w2v-BERT 2.0 + MHFA with U3-xi pooling: 4 x 2, 12.3 GiB, 0.7 h, 2.3 GB
- `conf/xlsr1b_mhfa.yaml`, XLS-R 1B + MHFA: 2 x 4, 20.0 GiB, 4.1 h, 3.9 GB
- `conf/omniw2v1b_mhfa.yaml`, omniASR-W2V-1B + MHFA: 2 x 4, 19.5 GiB, 4.1 h, 3.9 GB
```bash
bash run.sh --stage 4 --stop_stage 4 --config conf/xlsr1b_mhfa.yaml --exp_dir exp/xlsr1b_mhfa --gpus "[0,1]"
```

## Testing
`bash run.sh --stage 5 --stop_stage 7 --exp_dir exp/w2vbert_mhfa_ft [--epoch N] [--test_sets "dev_heldout"]` scores
the last (or the given) epoch; `--epoch best` scores `best_model.pt`. Stage 7 (`local/score_recipe.py`) first removes
a single-tap echo detected at one of four fixed delays (`local/echo.py`), then averages P(spoof) over the whole file
and its repetition up to 4 s for files shorter than 4 s, and over 4 s windows (hop 1 s) for files longer than 12 s;
other files get one whole-file score.

## Teachers and a student
For audio without labels that you supply (`data/<unlab_set>/wav.scp`; check that you may train on it): stage 8 scores
it with `best_model.pt` of every `--teacher_exps` directory (echo step, then stage 5) and writes the weighted mean of
P(spoof) over these and any `--teacher_scores` files of other models; stage 9 labels a row spoof if the mean is
>= 0.5 and writes `data/train_student` = `data/train` + `--copies` (3) copies of these rows (DATA.md). Any trained
experiment can be a teacher. A student is trained on the list from the pre-trained encoder, or fine-tuned from a
teacher with the two `_ft` configs as above:
```bash
bash run.sh --stage 8 --stop_stage 9 --unlab_set <unlab_set> --teacher_exps "exp/w2vbert_mhfa exp/xlsr1b_mhfa"
bash run.sh --stage 4 --stop_stage 4 --train_set train_student --config conf/w2vbert_mhfa.yaml --exp_dir exp/student \
  --gpus "[0,1]"
bash run.sh --stage 4 --stop_stage 4 --train_set train_student --config conf/w2vbert_mhfa_ft_aug.yaml \
  --exp_dir exp/student_ft --init_exp exp/w2vbert_mhfa --init_epoch best
```

## Checkpoints
https://huggingface.co/JYP2024/rtc-sdd-mhfa-models

## Licence
Code: Apache-2.0, as the toolkit; U3-xi pooling adapted from github.com/mrjunjieli/wespeaker_u_cube, Apache-2.0;
facebook/w2v-bert-2.0: MIT, XLS-R 1B (facebook/wav2vec2-xls-r-1b): Apache-2.0, per their model cards; omniASR-W2V-1B:
Apache-2.0, per the release of its authors. Check the licences of the corpora and the challenge rules on external data
before use (DATA.md).
