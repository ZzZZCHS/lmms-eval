#!/bin/bash

# export DECORD_LOG_LEVEL=error
export DECORD_EOF_RETRY_MAX=100000
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

debug=false
for arg in "$@"
do
  if [ "$arg" == "--debug" ]; then
    debug=true
  fi
done

MAX_PIXELS=262144 #$((768 * 32 * 32) # 530000
MIN_PIXELS=8192 #$((8 * 32 * 32))
max_num_frames=128
max_new_tokens=16

read -r account_name best_pressure partition_name < <(./select_account.sh)
# partition_name="scavenger" # nova, interactive, scavenger(h200), instruction

# allow manual override
for arg in "$@"; do
  if [[ "$arg" == --account=* ]]; then
    account_name="${arg#*=}"
  fi
  if [[ "$arg" == --partition=* ]]; then
    partition_name="${arg#*=}"
  fi
done

echo "Selected account: $account_name"
echo "Best pressure: $best_pressure"
echo "Selected partition: $partition_name"

gpu_type="a100" # a100, h200, l40s
gpu_num=1

compression_method="interval"
base_scale=0.01
importance_a=800 # -1 for auto
importance_distance_type="l2" # l2, cosine
base_scale_p=$(awk -v scale="$base_scale" 'BEGIN { print scale * 100 }')
interval_separate_method="consecutive_difference_change" # consecutive_difference_change, single_interval
token_merge_alpha=2 # -1 means no token merge, 2 means importance-based weighted token merge
random_sampling_method="pivotal" # pivotal, multinomial
random_sampling_seed=2718281828459045 # 3141592653589793, 2718281828459045, 1644934089375537, 9182736455463721, 1357913579135791, 8112963841460663, 4876659872345019, 7568372919931127, 9923457712349835, 5521810983345569, 6748391029384751, 314159, 271821, 918273, 135791, 547921, 889331, 42, 314159265358979[1,5,7,9]
temporal_sigma=0 # 16
diff_threshold=110
diff_change_threshold=70 # 70
diff_change_percent_threshold=0.4 # 0.35

for arg in "$@"
do
  if [[ "$arg" == --base_scale=* ]]; then
    base_scale="${arg#*=}"
    base_scale_p=$(awk -v scale="$base_scale" 'BEGIN { print scale * 100 }')
  fi
  if [[ "$arg" == --interval_separate_method=* ]]; then
    interval_separate_method="${arg#*=}"
  fi
  if [[ "$arg" == --importance_a=* ]]; then
    importance_a="${arg#*=}"
  fi
  if [[ "$arg" == --max_pixels=* ]]; then
    MAX_PIXELS="${arg#*=}"
  fi
  if [[ "$arg" == --max_num_frames=* ]]; then
    max_num_frames="${arg#*=}"
  fi
  if [[ "$arg" == --compression_method=* ]]; then
    compression_method="${arg#*=}"
  fi
  if [[ "$arg" == --max_new_tokens=* ]]; then
    max_new_tokens="${arg#*=}"
  fi
done

# assert compression_method is in [interval, original]
if [[ "$compression_method" != "interval" && "$compression_method" != "original" ]]; then
  echo "Error: compression_method must be either 'interval' or 'original'."
  exit 1
fi


exp_name="${compression_method}_${interval_separate_method}_${importance_distance_type}_a${importance_a}_merge${token_merge_alpha}_${random_sampling_method}_seed${random_sampling_seed}_Tsigma${temporal_sigma}_diff-${diff_threshold}-${diff_change_threshold}-${diff_change_percent_threshold}_${base_scale_p}"
# exp_name="original"

if [ $debug = true ]; then
  log_dir="./logs_debug/${exp_name}"
  tasks="videomme"
  limit=35
  cpu_memory="64G"
else
  log_dir="./logs/${exp_name}"
  # log_dir="./logs/random_25"
  # log_dir="./logs/density_1"
  # tasks="videomme,mlvu_dev,longvideobench_val_v,mvbench"
  # tasks="videomme,mlvu_dev,longvideobench_val_v,mvbench"
  tasks="videomme"
  limit=1000000000
  cpu_memory="64G"
fi
echo "Logging to $log_dir"

# if has --tasks argument, override tasks variable
for arg in "$@"
do
  if [[ "$arg" == --tasks=* ]]; then
    tasks="${arg#*=}"
  fi
done

start_time=$(date +%s)
echo "Start time: $(date)"

# srun --account="$account_name" --time=1-00:00:00 --nodes=1 --cpus-per-task=8 --mem=${cpu_memory} --partition="$partition_name" --gres=gpu:"$gpu_type":"$gpu_num" \
  accelerate launch --num_processes="$gpu_num" \
  -m lmms_eval \
  --model qwen3_vl \
  --model_args pretrained=Qwen/Qwen3-VL-8B-Instruct,min_pixels=${MIN_PIXELS},max_pixels=${MAX_PIXELS},fps=2,max_num_frames=${max_num_frames},attn_implementation=flash_attention_2,base_scale=${base_scale},importance_a=${importance_a},importance_distance_type=${importance_distance_type},interval_separate_method=${interval_separate_method},token_merge_alpha=${token_merge_alpha},random_sampling_method=${random_sampling_method},random_sampling_seed=${random_sampling_seed},temporal_sigma=${temporal_sigma},diff_threshold=${diff_threshold},diff_change_threshold=${diff_change_threshold},diff_change_percent_threshold=${diff_change_percent_threshold},compression_method=${compression_method} \
  --gen_kwargs max_new_tokens=${max_new_tokens},temperature=0,top_p=1.0,num_beams=1,do_sample=False \
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