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

read -r account_name best_pressure partition_name < <(./select_account.sh)

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
base_scale=0.1
importance_a=800 # -1 for auto
importance_distance_type="l2" # l2, cosine
base_scale_p=$(awk -v scale="$base_scale" 'BEGIN { print scale * 100 }')
interval_separate_method="consecutive_difference_change" # consecutive_difference_change, single_interval
token_merge_alpha=2 # -1 means no token merge, 2 means importance-based weighted token merge
token_merge_type="importance"
random_sampling_method="pivotal" # pivotal, multinomial
random_sampling_seed=2718281828459045

consolidation_sim_threshold=-1
attn_gamma=0.0
aug_gamma=1.0
do_whitening=false
keep_position_ids=false

temporal_sigma=0
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
done

exp_name="${compression_method}_${interval_separate_method}_${importance_distance_type}_a${importance_a}_sim_thres${consolidation_sim_threshold}_whiten-${do_whitening}_attn_gamma${attn_gamma}_aug_gamma${aug_gamma}_merge-${token_merge_type}-${token_merge_alpha}_${random_sampling_method}_seed${random_sampling_seed}_Tsigma${temporal_sigma}_diff-${diff_threshold}-${diff_change_threshold}-${diff_change_percent_threshold}_keeppos${keep_position_ids}_${base_scale_p}"
# exp_name="original"

if [ $debug = true ]; then
  log_dir="./logs_debug/${exp_name}"
  tasks="videomme"
  limit=10
  cpu_memory="48G"
else
  log_dir="./logs/${exp_name}"
  # log_dir="./logs/random_25"
  # log_dir="./logs/density_1"
  tasks="videomme,mlvu_dev,longvideobench_val_v,mvbench"
  # tasks="videomme"
  limit=1000000000
  cpu_memory="64G"
fi
echo "Logging to $log_dir"

# if --tasks is passed in args, set tasks to that
for arg in "$@"
do
  if [[ "$arg" == --tasks=* ]]; then
    tasks="${arg#*=}"
  fi
done

start_time=$(date +%s)
echo "Start time: $(date)"

srun --account="$account_name" --time=48:00:00 --nodes=1 --cpus-per-task=8 --mem=${cpu_memory} --partition="$partition_name" --gres=gpu:"$gpu_type":"$gpu_num" --exclude="nova21-gpu-2" \
  accelerate launch --num_processes="$gpu_num" \
  -m lmms_eval \
  --model llava_vid \
  --model_args pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=64,mm_spatial_pool_mode=average,attn_implementation=flash_attention_2 \
  --gen_kwargs max_new_tokens=16,temperature=0,top_p=1.0,num_beams=1,do_sample=False,base_scale=${base_scale},importance_a=${importance_a},importance_distance_type=${importance_distance_type},interval_separate_method=${interval_separate_method},token_merge_alpha=${token_merge_alpha},random_sampling_method=${random_sampling_method},random_sampling_seed=${random_sampling_seed},temporal_sigma=${temporal_sigma},compression_method=${compression_method} \
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

echo "llava-video-7b evaluation completed. Logs are saved in $log_dir/lmms-lab__LLaVA-Video-7B-Qwen2"