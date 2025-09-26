#!/bin/bash
#
#$ -cwd
#$ -V
#$ -N MHFA_PQ_SPL
#$ -o out/out_$JOB_ID_$TASK_ID.out
#$ -e out/error_$JOB_ID_$TASK_ID.err
#$ -q long.q@supergpu8,long.q@supergpu9,long.q@supergpu10,long.q@supergpu11,long.q@supergpu12,long.q@supergpu13,long.q@supergpu14,long.q@supergpu16,long.q@supergpu18
#$ -pe smp 1
#$ -l gpu_ram=15G,ram_free=10G,mem_free=10G,gpu=1
#$ -t 1-20

source ~/.bashrc
conda activate /mnt/matylda6/pengjy/python_new/miniconda/envs/wespeaker

cd /mnt/matylda6/pengjy/asv_wedefense/wedefense/egs/detection/asvspoof5/v15_ssl_mhfa_pruning_qua



SSLMODELs=(
        w1/MHFA_wavlmbase_qua_w1_s10
        w1/MHFA_wavlmbase_qua_w1_s20
        w1/MHFA_wavlmbase_qua_w1_s40
        w1/MHFA_wavlmbase_qua_w1_s60
        w1/MHFA_wavlmbase_qua_w1_s80
        w2/MHFA_wavlmbase_qua_w2_s10
        w2/MHFA_wavlmbase_qua_w2_s20
        w2/MHFA_wavlmbase_qua_w2_s40
        w2/MHFA_wavlmbase_qua_w2_s60
        w2/MHFA_wavlmbase_qua_w2_s80
        w4/MHFA_wavlmbase_qua_w4_s10
        w4/MHFA_wavlmbase_qua_w4_s20
        w4/MHFA_wavlmbase_qua_w4_s40
        w4/MHFA_wavlmbase_qua_w4_s60
        w4/MHFA_wavlmbase_qua_w4_s80
        w8/MHFA_wavlmbase_qua_w8_s10
        w8/MHFA_wavlmbase_qua_w8_s20
        w8/MHFA_wavlmbase_qua_w8_s40
        w8/MHFA_wavlmbase_qua_w8_s60
        w8/MHFA_wavlmbase_qua_w8_s80
        )
upstream=${SSLMODELs[$SGE_TASK_ID-1]} 

# echo "[$(date)] Starting Job ID: $SLURM_JOB_ID" > logs/debug_$SLURM_JOB_ID.txt
echo "[$(date)] Starting Job ID: $JOB_ID" > logs/debug_${JOB_ID}_${SGE_TASK_ID}.txt

./run_qua.sh --stage 3 --stop_stage 3 --config conf/qua_pruning/${upstream}.yaml --exp_dir exp/qua_pruning/${upstream}

# echo "[$(date)] Finished Job ID: $SLURM_JOB_ID" >> logs/debug_$SLURM_JOB_ID.txt
echo "[$(date)] Finished Job ID: $JOB_ID" >> logs/debug_${JOB_ID}_${SGE_TASK_ID}.txt
