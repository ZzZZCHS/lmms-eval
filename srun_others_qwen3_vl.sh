#!/bin/bash

# export DECORD_LOG_LEVEL=error

MAX_PIXELS=262144 #$((768 * 32 * 32))
MIN_PIXELS=8192 #$((8 * 32 * 32))

# account_name="yangli1-lab" # yangli1-lab, bweng-lab
account_name=$(./select_account.sh)
echo "Selected account: $account_name"

partition_name="nova" # nova, interactive, scavenger(h200)
gpu_type="a100" # a100, h200, l40s
gpu_num=1

allowed_methods=("original" "random" "interval" "vidcom2" "fastvid" "prunevid" "dycoke" "visionzip" "fastv")
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

if [ -z "$2" ]; then
  echo "No base_scale argument supplied. Using default base_scale=1.0"
  base_scale=1.0
else
  base_scale=$2
fi


debug=false
fastv_R=1.0
for arg in "$@"
do
  if [ "$arg" == "--debug" ]; then
    debug=true
  elif [[ "$arg" == --fastv_R=* ]]; then
    fastv_R="${arg#--fastv_R=}"
  fi
done

base_scale_p=$(awk -v scale="$base_scale" 'BEGIN { print scale * 100 }')
fastv_R_p=$(awk -v scale="$fastv_R" 'BEGIN { print scale * 100 }')
if [ "$fastv_R_p" == "100" ]; then
  exp_name="${compression_method}_${base_scale_p}"
else
  exp_name="${compression_method}_${base_scale_p}_R${fastv_R_p}"
fi

if [ $debug = true ]; then
  log_dir="./logs_debug/${exp_name}"
  tasks="videomme"
  limit=100
  cpu_memory="64G"
else
  log_dir="./logs/${exp_name}"
  # log_dir="./logs/random_25"
  # log_dir="./logs/density_1"
  # tasks="videomme,mlvu_dev,longvideobench_val_v,mvbench"
  tasks="videomme,mlvu_dev,longvideobench_val_v,mvbench"
  limit=1000000000
  cpu_memory="64G"
fi
echo "Logging to $log_dir"

start_time=$(date +%s)
echo "Start time: $(date)"

srun --account="$account_name" --time=2-00:00:00 --nodes=1 --cpus-per-task=8 --mem=${cpu_memory} --partition="$partition_name" --gres=gpu:"$gpu_type":"$gpu_num" \
  accelerate launch --num_processes="$gpu_num" \
  -m lmms_eval \
  --model qwen3_vl \
  --model_args pretrained=Qwen/Qwen3-VL-8B-Instruct,min_pixels=${MIN_PIXELS},max_pixels=${MAX_PIXELS},fps=2,max_num_frames=128,attn_implementation=flash_attention_2,base_scale=${base_scale},compression_method=${compression_method},fastv_R=${fastv_R} \
  --gen_kwargs max_new_tokens=16,temperature=0,top_p=1.0,num_beams=1,do_sample=False \
  --tasks $tasks \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix qwen3_vl \
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

echo "qwen3-vl-8b evaluation completed. Logs are saved in $log_dir."