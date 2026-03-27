#!/bin/bash
#SBATCH --job-name=qwen3b_v2
#SBATCH --gpus=6000ada:2
#SBATCH --qos=rose
#SBATCH --time=1-00:00:00
#SBATCH --output=/projects/_ssd/xrssd/PCB_structure_layout/logs/train-%j.out
#SBATCH --error=/projects/_ssd/xrssd/PCB_structure_layout/logs/train-%j.err

source /etc/profile.d/z00-lmod.sh
module load Miniforge3
source activate omini

cd /projects/_ssd/xrssd/PCB_structure_layout

DATA=/projects/_ssd/xrssd/data/ti_pcb/layout_data/v2_Color_Res_Class_xywh
RUNS=/projects/_ssd/xrssd/PCB_structure_layout/runs

accelerate launch --multi_gpu --num_processes 2 \
  train_pcb_layout.py \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --train_data $DATA/train.jsonl \
  --output_dir $RUNS/Qwen3b_lora_v2_2048_bs16_6000ada \
  --max_length 2048 \
  --batch_size 2 \
  --grad_accum 4 \
  --epochs 50 \
  --lr 2e-4 \
  --save_steps 500 \
  --logging_steps 5
