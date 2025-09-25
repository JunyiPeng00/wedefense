#!/bin/bash
#
#$ -cwd
#$ -V
#$ -N MHFA_PQ_SPL
#$ -o out/out_$JOB_ID_$TASK_ID.out
#$ -e out/error_$JOB_ID_$TASK_ID.err
#$ -q long.q@supergpu12,long.q@supergpu13,long.q@supergpu14,long.q@supergpu16,long.q@supergpu18
#$ -pe smp 1
#$ -l gpu_ram=15G,ram_free=10G,mem_free=10G,gpu=1
#$ -t 4-4

source ~/.bashrc
conda activate /mnt/matylda6/pengjy/python_new/miniconda/envs/wespeaker

cd /mnt/matylda6/pengjy/asv_wedefense/wedefense/egs/detection/asvspoof5/v15_ssl_mhfa_pruning_qua

SSLMODELs=(
        MHFA_wavlmbase_qua_w1_fixed
        MHFA_wavlmbase_qua_w2_fixed
        MHFA_wavlmbase_qua_w4_fixed
        MHFA_wavlmbase_qua_w8_fixed
        )
upstream=${SSLMODELs[$SGE_TASK_ID-1]} 

# echo "[$(date)] Starting Job ID: $SLURM_JOB_ID" > logs/debug_$SLURM_JOB_ID.txt
echo "[$(date)] Starting Job ID: $JOB_ID" > logs/debug_${JOB_ID}_${SGE_TASK_ID}.txt

./run_qua.sh --stage 3 --stop_stage 3 --config conf/qua_fixed/${upstream}.yaml --exp_dir exp/qua_fixed/${upstream}

# echo "[$(date)] Finished Job ID: $SLURM_JOB_ID" >> logs/debug_$SLURM_JOB_ID.txt
echo "[$(date)] Finished Job ID: $JOB_ID" >> logs/debug_${JOB_ID}_${SGE_TASK_ID}.txt
