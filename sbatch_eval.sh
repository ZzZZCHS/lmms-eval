#!/bin/bash

#SBATCH --job-name=vidcom2
#SBATCH --output=srun_logs/vidcom2_%j.out
#SBATCH --error=srun_logs/vidcom2_%j.err
#SBATCH --partition=nova
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --account=yangli1-lab

source activate vidcom2

srun accelerate launch --num_processes=1 \
  -m lmms_eval \
  --model llava_onevision \
  --model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen,attn_implementation=flash_attention_2 \
  --tasks mlvu_dev,longvideobench_val_v,mvbench \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix llava_onevision \
  --output_path ./logs/ \
  --limit 2

# --tasks videomme,mlvu_dev,longvideobench_val_v,mvbench