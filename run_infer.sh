#!/bin/bash
#SBATCH --job-name=qwen3b_v2_infer
#SBATCH --gpus=6000ada:1
#SBATCH --time=0-08:00:00
#SBATCH --output=/projects/_ssd/xrssd/PCB_structure_layout/logs/infer-%j.out
#SBATCH --error=/projects/_ssd/xrssd/PCB_structure_layout/logs/infer-%j.err

source /etc/profile.d/z00-lmod.sh
module load Miniforge3
source activate omini

cd /projects/_ssd/xrssd/PCB_structure_layout

DATA=/projects/_ssd/xrssd/data/ti_pcb/layout_data/v2_Color_Res_Class_xywh
RUNS=/projects/_ssd/xrssd/PCB_structure_layout/runs
CKPT=$RUNS/Qwen3b_lora_v2_2048_bs16_6000ada/final
OUT=$RUNS/Qwen3b_lora_v2_2048_bs16_6000ada/infer_output

python infer_layout.py \
    --backbone 3b \
    --ckpt $CKPT \
    --output_dir $OUT \
    --num_samples -1 \
    --max_new_tokens 2048 \
    --test_jsonl $DATA/test.jsonl \
    --board_img_dir $DATA/image/test
