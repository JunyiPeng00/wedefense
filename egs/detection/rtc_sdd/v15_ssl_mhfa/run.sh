#!/bin/bash

# Copyright 2026 Junyi Peng (pengjy@fit.vut.cz)

# RTC-SDD: w2v-BERT 2.0 + MHFA detector on the train and dev sets.
#   bash run.sh --stage 1 --stop_stage 3 ...   # data processing (DATA.md)
#   bash run.sh --stage 4 --stop_stage 4 ...   # training, once per training run (README.md)
#   bash run.sh --stage 5 --stop_stage 7 ...   # scoring
#   bash run.sh --stage 8 --stop_stage 9 ...   # teacher-labelled rows for a student (README.md)

. ./path.sh || exit 1
set -e

stage=1
stop_stage=7

rtcsdd_dir=        # challenge data: train_label.txt, dev_label.txt, meta/, wav/{train,dev}
data=data          # data folder (conf/event_plan.json expects data/noise)
sim_dir=           # output directory of the channel simulation (about 56 GB)
nj=16              # CPU processes of the channel simulation
musan_dir=         # MUSAN
rirs_dir=          # RIRS_NOISES
esc50_dir=         # ESC-50-master
fsd50k_dir=        # FSD50K
rnnoise_dir=       # RNNoise noise contributions (rnnoise_contributions)

config=conf/w2vbert_mhfa.yaml
exp_dir=exp/w2vbert_mhfa
gpus="[0]"         # conf/w2vbert_mhfa.yaml is written for two GPUs: --gpus "[0,1]"
init_exp=          # stage 4: start from a checkpoint of this experiment (empty: from the pre-trained encoder)
init_epoch=        # its epoch (default: the last one; best: best_model.pt)
epoch=             # stages 5-7: epoch of exp_dir to score (default: the last one; best: best_model.pt)
test_sets="dev_heldout"

train_set=train    # stage 4: training list ${data}/${train_set} (stage 9 writes train_student)
unlab_set=         # stages 8-9: ${data}/${unlab_set}/wav.scp lists the audio without labels
teacher_exps=      # stage 8: experiment directories of the teachers (their best_model.pt is used)
teacher_scores=    # stage 8: further score files "<utt> <P(spoof)>" of that audio
teacher_weights=   # stage 8: one weight per teacher, experiments first (default: equal)
copies=3           # stage 9: copies of the teacher-labelled rows in the student list

. tools/parse_options.sh || exit 1

last_epoch() {
  ls $1/models/model_*.pt | sed 's/.*model_\([0-9]*\)\.pt/\1/' | sort -n | tail -1
}

#######################################################################################
# Stage 1. Data folders of the train and dev sets: wav.scp, utt2lab
#######################################################################################
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Prepare datasets ..."
  python local/prepare_data.py ${rtcsdd_dir} ${data}
fi

#######################################################################################
# Stage 2. Noise lists and channel simulation of the offline recordings
#######################################################################################
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Simulate the channel ..."
  python local/make_noise_lists.py --musan_dir ${musan_dir} --rirs_dir ${rirs_dir} \
    --esc50_dir ${esc50_dir} --fsd50k_dir ${fsd50k_dir} \
    --rnnoise_dir ${rnnoise_dir} --out_dir ${data}/noise
  local/simulate_all.sh ${rtcsdd_dir} ${data} ${sim_dir} ${nj}
fi

#######################################################################################
# Stage 3. Training and validation lists; noise LMDB of the in-loader augmentation
#######################################################################################
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Prepare lists ..."
  python local/make_train_list.py ${data} ${sim_dir}
  for dset in train val; do
    python tools/make_raw_list.py ${data}/$dset/wav.scp ${data}/$dset/utt2lab ${data}/$dset/raw.list
  done
  # Note that lmdb does not work on NFS: if ${data} is on NFS, build it on a local disk and copy it.
  python tools/make_lmdb.py ${data}/noise/musan_noise.scp ${data}/noise/musan_lmdb
fi

#######################################################################################
# Stage 4. Training
#######################################################################################
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Start training ..."
  num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
  if [ -n "${init_exp}" ]; then
    model_init=${init_exp}/models/model_${init_epoch:-$(last_epoch ${init_exp})}.pt
    [ "${init_epoch}" = best ] && model_init=${init_exp}/models/best_model.pt
    echo "Initial model: ${model_init}"
  fi
  # To avoid the randomly generated port is occuppied.
  while :
  do
    port=$(( (RANDOM % 100) + 29500 ))
    if ! lsof -i:$port >/dev/null; then
      break
    fi
  done

  torchrun --rdzv_backend=c10d --rdzv_endpoint=$(hostname):$((port)) --nnodes=1 --nproc_per_node=$num_gpus \
    wedefense/bin/train.py --config $config \
      --exp_dir ${exp_dir} \
      --gpus $gpus \
      --data_type raw \
      --train_data ${data}/${train_set}/raw.list \
      --train_label ${data}/${train_set}/utt2lab \
      --val_data ${data}/val/raw.list \
      --val_label ${data}/val/utt2lab \
      --noise_data ${data}/noise/musan_lmdb \
      ${model_init:+--model_init $model_init}
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 5 ]; then
  epoch=${epoch:-$(last_epoch ${exp_dir})}
  model_path=${exp_dir}/models/model_${epoch}.pt
  [ "${epoch}" = best ] && model_path=${exp_dir}/models/best_model.pt
  echo "Model: ${model_path}"
fi

#######################################################################################
# Stage 5. Embeddings, logits and posteriors, score files "<utt> <P(spoof)>"
#######################################################################################
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Extract embeddings and posteriors ..."
  # the input features are computed on the CPU at test time
  sed 's/gpu_fbank: true/gpu_fbank: false/' ${exp_dir}/config.yaml > ${exp_dir}/config_test.yaml
  mkdir -p ${exp_dir}/scores
  for dset in ${test_sets}; do
    name=${dset}_ep${epoch}
    [ -f ${data}/$dset/raw.list ] || \
      python tools/make_raw_list.py ${data}/$dset/wav.scp ${data}/$dset/utt2lab ${data}/$dset/raw.list
    python wedefense/bin/extract.py \
      --config ${exp_dir}/config_test.yaml \
      --model_path $model_path \
      --data_type raw \
      --data_list ${data}/$dset/raw.list \
      --embed_ark ${exp_dir}/embeddings/$name/embedding.ark \
      --batch_size 1 --num_workers 4
    mkdir -p ${exp_dir}/posteriors/$name
    python wedefense/bin/infer.py --model_path $model_path \
      --config ${exp_dir}/config.yaml \
      --num_classes 2 \
      --embedding_scp_path ${exp_dir}/embeddings/$name/embedding.scp \
      --out_path ${exp_dir}/posteriors/$name
    python local/make_scores.py \
      --posteriors_scp ${exp_dir}/posteriors/$name/posteriors.scp \
      --train_label ${data}/train/utt2lab \
      --protocol ${data}/$dset/utt2lab \
      --out ${exp_dir}/scores/$name.txt
  done
fi

#######################################################################################
# Stage 6. Macro-F1 at threshold 0.5
#######################################################################################
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Measuring Performance ..."
  for dset in ${test_sets}; do
    python local/macro_f1.py ${exp_dir}/scores/${dset}_ep${epoch}.txt ${data}/$dset/utt2lab
  done
fi

#######################################################################################
# Stage 7. Scores with the test-time recipe (echo step, short and long files), Macro-F1
#######################################################################################
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Score with the test-time recipe ..."
  mkdir -p ${exp_dir}/scores
  for dset in ${test_sets}; do
    python local/score_recipe.py \
      --config ${exp_dir}/config.yaml \
      --model_path $model_path \
      --wav_scp ${data}/$dset/wav.scp \
      --out ${exp_dir}/scores/${dset}_ep${epoch}_recipe.txt
    python local/macro_f1.py ${exp_dir}/scores/${dset}_ep${epoch}_recipe.txt ${data}/$dset/utt2lab
  done
fi

#######################################################################################
# Stage 8. Teacher scores of the audio without labels (echo step, stage 5), fused score
#######################################################################################
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  echo "Score with the teachers and fuse ..."
  scored=${unlab_set}_deecho
  python local/apply_echo_step.py ${data}/${unlab_set}/wav.scp ${data}/$scored
  # the list format needs a label column; it is not used
  awk '{print $1, "unknown"}' ${data}/$scored/wav.scp > ${data}/$scored/utt2lab
  python tools/make_raw_list.py ${data}/$scored/wav.scp ${data}/$scored/utt2lab ${data}/$scored/raw.list
  score_files=
  for teacher in ${teacher_exps}; do
    bash run.sh --stage 5 --stop_stage 5 --data ${data} --exp_dir $teacher --epoch best --test_sets $scored
    score_files="$score_files $teacher/scores/${scored}_epbest.txt"
  done
  python local/fuse_scores.py --scores $score_files ${teacher_scores} \
    ${teacher_weights:+--weights ${teacher_weights}} \
    --out ${data}/${unlab_set}/fused_scores.txt
fi

#######################################################################################
# Stage 9. Student list: the training list and copies of the teacher-labelled rows
#######################################################################################
if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
  echo "Prepare the student list ..."
  python local/make_student_list.py --train_dir ${data}/train \
    --wav_scp ${data}/${unlab_set}_deecho/wav.scp --scores ${data}/${unlab_set}/fused_scores.txt \
    --copies ${copies} --out_dir ${data}/train_student
  python tools/make_raw_list.py ${data}/train_student/wav.scp ${data}/train_student/utt2lab \
    ${data}/train_student/raw.list
fi
