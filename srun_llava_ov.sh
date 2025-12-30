#!/bin/bash

# export DECORD_LOG_LEVEL=error

debug=false

# account_name="yangli1-lab" # yangli1-lab, bweng-lab
account_name=$(./select_account.sh)
echo "Selected account: $account_name"

partition_name="nova" # nova, interactive, scavenger(h200)
gpu_type="a100" # a100, h200, l40s
gpu_num=1

compression_method="interval"
base_scale=$1
importance_a=800
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
exp_name="${compression_method}_${interval_separate_method}_${importance_distance_type}_a${importance_a}_merge${token_merge_alpha}_${random_sampling_method}_seed${random_sampling_seed}_Tsigma${temporal_sigma}_diff-${diff_threshold}-${diff_change_threshold}-${diff_change_percent_threshold}_${base_scale_p}"
# exp_name="original"

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
  limit=1000000000
  cpu_memory="64G"
fi
echo "Logging to $log_dir"

start_time=$(date +%s)
echo "Start time: $(date)"

srun --account="$account_name" --time=24:00:00 --nodes=1 --cpus-per-task=8 --mem=${cpu_memory} --partition="$partition_name" --gres=gpu:"$gpu_type":"$gpu_num" \
  accelerate launch --num_processes=1 \
  -m lmms_eval \
  --model llava_onevision \
  --model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen,attn_implementation=flash_attention_2 \
  --gen_kwargs max_new_tokens=16,temperature=0,top_p=1.0,num_beams=1,do_sample=False,base_scale=${base_scale},importance_a=${importance_a},importance_distance_type=${importance_distance_type},interval_separate_method=${interval_separate_method},token_merge_alpha=${token_merge_alpha},random_sampling_method=${random_sampling_method},random_sampling_seed=${random_sampling_seed},temporal_sigma=${temporal_sigma},diff_threshold=${diff_threshold},diff_change_threshold=${diff_change_threshold},diff_change_percent_threshold=${diff_change_percent_threshold},compression_method=${compression_method} \
  --tasks $tasks \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix llava_onevision \
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