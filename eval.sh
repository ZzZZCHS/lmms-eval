#!/bin/bash
debug=true
if [ $debug = true ]; then
  log_dir="./logs_debug"
  tasks="videomme"
else
  log_dir="./logs"
  tasks="videomme,mlvu_dev,longvideobench_val_v,mvbench"
fi

accelerate launch --num_processes=1 \
  -m lmms_eval \
  --model llava_onevision \
  --model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen,attn_implementation=flash_attention_2 \
  --tasks $tasks \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix llava_onevision_vidcom2 \
  --output_path $log_dir \
  --limit 100



  # --tasks videomme,mlvu_dev,longvideobench_val_v,mvbench \

  # srun --account yangli1-lab --time=8:00:00 --nodes=1 --cpus-per-task=8 --mem=256G --partition=interactive --gres=gpu:a100:1 --pty /bin/bash
  # srun --account bweng-lab --time=8:00:00 --nodes=1 --cpus-per-task=8 --mem=256G --partition=interactive --gres=gpu:a100:1 --pty /bin/bash

  # partition: scavenger h200
  # l40s
  