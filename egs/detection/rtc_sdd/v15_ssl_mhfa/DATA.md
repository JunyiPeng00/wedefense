# Data processing (stages 1-3 and 8-9 of `run.sh`)

Stages 1-3 use only the labelled train and dev sets: their online recordings plus simulated copies of
the offline recordings (a local RTC channel simulation); a fixed part of dev is held out for
validation. Stages 8-9 add rows of audio without labels that you supply (check that you may train on
it).

## Expected layout

```
<rtcsdd_dir>/train_label.txt, dev_label.txt      <offline|online>/<en|zh>/<name>.wav <bonafide|spoof>
<rtcsdd_dir>/meta/dev_offline_online_pairs.csv   offline_id,online_id (online_id may be empty)
<rtcsdd_dir>/wav/{train,dev}/{offline,online}/{en,zh}/*.wav
```

## Stage 1: lists

`python local/prepare_data.py <rtcsdd_dir> data` writes `wav.scp` and `utt2lab` (key = path in the
label file) for `train_offline` (38,660), `train_online` (37,125), `dev_offline` (7,739),
`dev_online` (7,465) and `dev_heldout` (3,028): a random 20 % of the dev offline recordings (seed
2026; 1,547) with their online copies (1,481). The split is by source recording, not by speaker.

## Stage 2: channel simulation

```
python local/make_noise_lists.py --musan_dir ... --rirs_dir ... --esc50_dir ... --fsd50k_dir ... \
    --rnnoise_dir ... --out_dir data/noise
local/simulate_all.sh <rtcsdd_dir> data <sim_dir> <nj>
```

The first command writes the noise lists (below). The second renders every line of
`conf/sim_variants.txt` (`<subset> <preset> <seed>`) with `local/simulate_rtc.py` into
`<sim_dir>/<subset>_<preset>/`, keeping the relative paths, and logs the parameters drawn for
each file in `sim_params.seed<seed>.jsonl`. Seeds are fixed; finished files are skipped on a restart.
Chain per file: peak normalisation and gain (+-6 dB) -> noise -> impulse response -> echo ->
WebRTC noise suppression and gain control -> codec (Opus 60 %, G.722 20 %, AAC 20 %; Opus with
DTX / FEC and bursty packet loss) -> zero runs -> head / tail trim -> 7 kHz low-pass -> peak level
and limiter. The presets (`PRESETS` in `local/simulate_rtc.py`) set what is switched on:

| preset | subset (seed) | models | key parameters |
|---|---|---|---|
| `noise` | train (11), dev (7) | background noise | one MUSAN noise file on every recording, SNR 5-35 dB; suppression level 1-3; packet loss 5-20 %; earlier sampler (`legacy`): Opus from 8 kbps, 0-0.4 s cut at the head and 0-0.15 s at the tail, peak -2 to -0.3 dBFS, zero runs up to 8 % |
| `noise_hard` | train (12), dev (8) | the same, harsher | SNR 0-20 dB; packet loss up to 30 %; zero runs in 50 % of the files; twice the clipping rate |
| `reverb_noise` | train (51), dev (45) | echo residual with noise | impulse response on 70 %, wet 0.25-0.6; noise on every recording, SNR 15-35 dB; suppression level 1; packet loss 5-10 %; duration floored to 100 ms |
| `reverb` | train (52), dev (46) | echo residual, mostly clean | as `reverb_noise` with wet 0.2-0.5 and noise on 30 %, drawn from `musan_reverb.scp` |
| `event` | train (63), dev (62) | noise families, echo, more codecs | as `reverb` with the family plan instead of MUSAN; echo on 20 % (30-300 ms, -24 to -12 dB); Opus SILK / CELT / hybrid at 6-30 kbps, 25 % narrowband at 8-20 kbps, three concealment modes; 15 % G.711 / G.726 / AAC 16-128 kbps |
| `event_soft` | train (64) | milder `event` | echo on 10 %; rain at 20-35 dB |
| `event_strong` | dev (55) | stronger echo | echo on 30 % at -18 to -6 dB; narrowband at 4-20 kbps |

Family plan (`FAMILIES`): keyboard 8 %, footsteps 13 %, birds 9 %, kitchen 3 % of the files as
short events at a local SNR of 10-25 dB; rain 3 % (15-35 dB) and office 25 % (20-35 dB) as
stationary background; 39 % stay clean; 15 % of the noisy files get a second family.

## Stage 3: training and validation lists

`python local/make_train_list.py data <sim_dir>` writes `data/train` (312,221 rows: train online and
six simulated copies of train offline, then the same for dev without the held-out sources) and
`data/val` (4,575 rows: held-out dev online and the `reverb_noise` and `reverb` copies of held-out
dev offline). A simulated copy has the key `<preset>/<offline key>`; the first key component
(`online` or the preset) is the group of the validation metric. `tools/make_raw_list.py` then
turns both into `raw.list`.

## Stages 8-9: teacher-labelled rows

Input: `data/<unlab_set>/wav.scp` (`<utt> <wav path>`; 16 kHz mono 16-bit wav) of audio without
labels. Stage 8 runs the echo step on it (`local/apply_echo_step.py` writes
`data/<unlab_set>_deecho/wav.scp`, in which a flagged file points to its de-echoed copy), adds a
placeholder `utt2lab` (label `unknown`; the list format needs the column) and scores that list with
`best_model.pt` of every `--teacher_exps` directory through stage 5: one whole-file score per file,
in `<exp>/scores/<unlab_set>_deecho_epbest.txt`. `local/fuse_scores.py` then fuses these files and
the `--teacher_scores` files: fused score = sum(w_i p_i) / sum(w_i) of P(spoof), weights
`--teacher_weights` (default equal), no normalisation, written to `data/<unlab_set>/fused_scores.txt`.
A score file of another model has one line `<utt> <P(spoof)>` per line of `wav.scp`, scored on the
files of `data/<unlab_set>_deecho/wav.scp`. For another epoch of a teacher: after a first run of
stage 8 (it writes that list), run stage 5 with `--epoch N --test_sets <unlab_set>_deecho` and fuse
the files with `local/fuse_scores.py`.

Stage 9 (`local/make_student_list.py`) labels a row `spoof` if its fused score is >= 0.5 and
`bonafide` otherwise (no row is dropped) and writes `data/train_student`: the rows of `data/train`,
then `--copies` (default 3) copies of the teacher-labelled rows with the keys `tl0/<utt>`,
`tl1/<utt>`, ..., pointing to the files of `data/<unlab_set>_deecho/wav.scp` (the files that were
scored). The loader draws rows uniformly and an epoch is a fixed number of chunks, so the copies set
the share of these rows in training.
`data/val` is not changed. Stage 4 trains on the list with `--train_set train_student` (with
`--init_exp <exp> --init_epoch best` it starts from `best_model.pt` of an experiment).

## Public corpora

| corpus | argument | used for |
|---|---|---|
| MUSAN (OpenSLR 17), `noise/` | `--musan_dir` | `musan_noise.scp`: 930 non-speech noise files; `musan_reverb.scp`: the 855 entries of `conf/musan_reverb.txt` (642 files without telephone, music, silent or voice-like recordings; keyboard and footstep files repeated) |
| RIRS_NOISES (OpenSLR 28) | `--rirs_dir` | `rirs.scp`: simulated small and medium rooms (40,000) and 317 measured responses, repeated 23 times so that they make up about 15 % of the draws |
| ESC-50 | `--esc50_dir` | `<family>.scp`: keyboard, footsteps, birds, rain, kitchen, office classes |
| FSD50K (dev audio + ground truth) | `--fsd50k_dir` | `<family>.scp`: the same families, clips with speech or music labels excluded |
| RNNoise noise contributions (`rnnoise_contributions`, raw 48 kHz) | `--rnnoise_dir` | office family: the 1,736 30 s pieces of office and coffee-shop recordings listed in `conf/rnnoise_pieces.txt` (pieces tagged as speech-like left out), written to `data/noise/rnnoise_wav` |

The lists are plain `key path` files in a fixed order (the draws depend on it). Check the licence of
each corpus and the challenge rules on external data before use.

## Requirements and cost

Python: numpy, scipy, soundfile, webrtc-noise-gain, opuslib (needs the system library libopus);
`ffmpeg` with the g722, aac, pcm_alaw, pcm_mulaw and g726 codecs on `PATH` (tested with the git
build N-125504-gc29d710cd5). On 8 CPU cores a train rendering takes roughly 15-25 min and a dev
rendering 3-8 min (about 20 core-hours for all twelve). Output: 16 kHz 16-bit wav, about 7.8 GB per
train and 1.6 GB per dev rendering, 56 GB in total.
