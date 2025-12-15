#!/bin/bash

# export DECORD_LOG_LEVEL=error

debug=false

account_name="bweng-lab" # yangli1-lab, bweng-lab
partition_name="nova" # nova, interactive, scavenger(h200)
gpu_type="a100" # a100, h200, l40s
gpu_num=1

# if has $1, assign to compression_method, else default to "original"
# before assigning to compression_method, check if $1 is in the allowed list, exit if not
# original, random, interval, vidcom2, fastvid, prunevid, dycoke
allowed_methods=("original" "random" "interval" "vidcom2" "fastvid" "prunevid" "dycoke")
if [ -z "$1" ]; then
  echo "No compression_method argument supplied. Using default compression_method=original"
  compression_method="original"
else
  compression_method="$1"
  if [[ ! " ${allowed_methods[@]} " =~ " ${compression_method} " ]]; then
    echo "Error: compression_method '$compression_method' is not in the allowed list: ${allowed_methods[*]}"
    exit 1
  fi
fi

# if has $2, assign to base_scale, else default to 1.0
if [ -z "$2" ]; then
  echo "No base_scale argument supplied. Using default base_scale=1.0"
  base_scale=1.0
else
  base_scale=$2
fi

base_scale_p=$(awk -v scale="$base_scale" 'BEGIN { print scale * 100 }')
exp_name="${compression_method}_${base_scale_p}"

if [ $debug = true ]; then
  log_dir="./logs_debug/${exp_name}"
  tasks="videomme"
  limit=100
  cpu_memory="64G"
else
  log_dir="./logs/${exp_name}"
  # log_dir="./logs/random_25"
  # log_dir="./logs/density_1"
  tasks="videomme,mlvu_dev,longvideobench_val_v,mvbench"
  # tasks="videomme"
  # tasks=$1
  limit=1000000000
  cpu_memory="64G"
fi
echo "Logging to $log_dir"

start_time=$(date +%s)
echo "Start time: $(date)"

srun --account="$account_name" --time=24:00:00 --nodes=1 --cpus-per-task=8 --mem=${cpu_memory} --partition="$partition_name" --gres=gpu:"$gpu_type":"$gpu_num" \
  accelerate launch --num_processes=1 \
  -m lmms_eval \
  --model llava_vid \
  --model_args pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=64,mm_spatial_pool_mode=average,attn_implementation=flash_attention_2 \
  --gen_kwargs max_new_tokens=16,temperature=0,top_p=1.0,num_beams=1,do_sample=False,base_scale=${base_scale},compression_method=${compression_method} \
  --tasks $tasks \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix llava_vid \
  --output_path $log_dir \
  --limit $limit

end_time=$(date +%s)
echo "End time: $(date)"

elapsed=$((end_time - start_time))

# convert seconds → HH:MM:SS
hours=$((elapsed / 3600))
minutes=$(((elapsed % 3600) / 60))
seconds=$((elapsed % 60))

printf "Total runtime: %02d:%02d:%02d (HH:MM:SS)\n" $hours $minutes $seconds

echo "llava-ov-7b evaluation completed. Logs are saved in $log_dir."