#!/bin/bash
#SBATCH --job-name=qwen3b_v2_eval
#SBATCH --gpus=6000ada:1
#SBATCH --time=0-04:00:00
#SBATCH --output=/projects/_ssd/xrssd/PCB_structure_layout/logs/eval-%j.out
#SBATCH --error=/projects/_ssd/xrssd/PCB_structure_layout/logs/eval-%j.err

source /etc/profile.d/z00-lmod.sh
module load Miniforge3
source activate omini

cd /projects/_ssd/xrssd/PCB_structure_layout

DATA=/projects/_ssd/xrssd/data/ti_pcb/layout_data/v2_Color_Res_Class_xywh
RUNS=/projects/_ssd/xrssd/PCB_structure_layout/runs
CKPT=$RUNS/Qwen3b_lora_v2_2048_bs16_6000ada/final

python eval_layout.py \
    --ckpt $CKPT \
    --num_samples -1 \
    --max_new_tokens 2048 \
    --test_jsonl $DATA/test.jsonl \
    --gpu 0
