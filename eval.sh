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

  # srun --account yangli1-lab --time=8:00:00 --nodes=1 --cpus-per-task=8 --mem=256G --partition=nova --gres=gpu:a100:1 --pty /bin/bash
  # srun --account bweng-lab --time=8:00:00 --nodes=1 --cpus-per-task=8 --mem=256G --partition=interactive --gres=gpu:a100:1 --pty /bin/bash

  # partition: scavenger h200
  # l40s
  

  # conda create -n vidcom_cu128 python=3.12 -y
  # conda activate vidcom_cu128
  # pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
  # cd lmms-eval
  # pip install -e ".[all]"
  # cd ..
  # pip install -e ".[train]"
  # pip install httpx==0.23.3 numpy==1.26.4 opencv-python==4.11.0.86 matplotlib==3.10.7
  # wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
  # pip install flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl


  # (transformers-4.57.1)