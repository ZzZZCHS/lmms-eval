#!/bin/bash
set -e

export HOME=/home/haifengh

# Conda
export CONDA_ROOT=/home/haifengh/miniconda3
source $CONDA_ROOT/etc/profile.d/conda.sh
conda activate vidcom_cu128

# CUDA
module load cuda
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Caches
export HF_HOME=/home/haifengh/.cache/huggingface
export HF_HUB_CACHE=/home/haifengh/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=/home/haifengh/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/home/haifengh/.cache/huggingface/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TORCH_HOME=/home/haifengh/.cache/torch
export ACCELERATE_CONFIG_FILE=$HF_HOME/accelerate/default_config.yaml
export TRITON_CACHE_DIR=/home/$USER/triton_cache

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

# account_name="bweng-lab" # yangli1-lab, bweng-lab, class-faculty
account_name=$(./select_account.sh)
echo "Selected account: $account_name"

partition_name="nova" # nova, interactive, scavenger(h200), instruction
gpu_type="a100" # a100, h200, l40s
gpu_num=1 # 4 for 72b

compression_method="interval"
base_scale=$1
importance_a=800 # -1 for auto
importance_distance_type="l2" # l2, cosine
base_scale_p=$(awk -v scale="$base_scale" 'BEGIN { print scale * 100 }')
interval_separate_method="consecutive_difference_change" # consecutive_difference_change, single_interval
# if args has --interval_separate_method, set to that
for arg in "$@"
do
  if [[ "$arg" == --interval_separate_method=* ]]; then
    interval_separate_method="${arg#*=}"
  fi
done

token_merge_alpha=2 # -1 means no token merge, 2 means importance-based weighted token merge
# if args has --token_merge_alpha, set to that
for arg in "$@"
do
  if [[ "$arg" == --token_merge_alpha=* ]]; then
    token_merge_alpha="${arg#*=}"
  fi
done

token_merge_type="importance"
# if args has --token_merge_type, set to that
for arg in "$@"
do
  if [[ "$arg" == --token_merge_type=* ]]; then
    token_merge_type="${arg#*=}"
  fi
done

random_sampling_method="pivotal" # pivotal, multinomial, topk
# if args has --random_sampling_method, set to that
for arg in "$@"
do
  if [[ "$arg" == --random_sampling_method=* ]]; then
    random_sampling_method="${arg#*=}"
  fi
done

max_frames_num=32
# if args has --max_frames_num, set to that
for arg in "$@"
do
  if [[ "$arg" == --max_frames_num=* ]]; then
    max_frames_num="${arg#*=}"
  fi
done

random_sampling_seed=2718281828459045 # 3141592653589793, 2718281828459045, 1644934089375537, 9182736455463721, 1357913579135791, 8112963841460663, 4876659872345019, 7568372919931127, 9923457712349835, 5521810983345569, 6748391029384751, 314159, 271821, 918273, 135791, 547921, 889331, 42, 314159265358979[1,5,7,9]
temporal_sigma=0 # 16
diff_threshold=110
diff_change_threshold=70 # 70
diff_change_percent_threshold=0.4 # 0.35
exp_name="${compression_method}_${interval_separate_method}_${importance_distance_type}_a${importance_a}_merge-${token_merge_type}-${token_merge_alpha}_${random_sampling_method}_seed${random_sampling_seed}_Tsigma${temporal_sigma}_diff-${diff_threshold}-${diff_change_threshold}-${diff_change_percent_threshold}_${base_scale_p}"
model_size="7b" # for 72b
# exp_name="original"

if [ $debug = true ]; then
  log_dir="./logs_debug/${exp_name}"
  tasks="videomme"
  limit=1000
  cpu_memory="64G" # 384G for 72b
  srun_time="1:00:00"
else
  log_dir="./logs/${exp_name}"
  # log_dir="./logs/random_25"
  # log_dir="./logs/density_1"
  # tasks="videomme,mlvu_dev,longvideobench_val_v,mvbench"
  tasks="videomme"
  # tasks=$3
  limit=1000000000
  cpu_memory="64G" # 384G for 72b
  srun_time="20:00:00" # for 72b
fi
echo "Logging to $log_dir"

start_time=$(date +%s)
echo "Start time: $(date)"

srun --account="$account_name" --time=${srun_time} --nodes=1 --cpus-per-task=8 --mem=${cpu_memory} --partition="$partition_name" --gres=gpu:"$gpu_type":"$gpu_num" \
  accelerate launch --num_processes=${gpu_num} \
  -m lmms_eval \
  --model llava_onevision \
  --model_args pretrained=lmms-lab/llava-onevision-qwen2-${model_size}-ov,conv_template=qwen_1_5,max_frames_num=${max_frames_num},model_name=llava_qwen,attn_implementation=flash_attention_2 \
  --gen_kwargs max_new_tokens=16,temperature=0,top_p=1.0,num_beams=1,do_sample=False,base_scale=${base_scale},importance_a=${importance_a},importance_distance_type=${importance_distance_type},interval_separate_method=${interval_separate_method},token_merge_alpha=${token_merge_alpha},token_merge_type=${token_merge_type},random_sampling_method=${random_sampling_method},random_sampling_seed=${random_sampling_seed},temporal_sigma=${temporal_sigma},diff_threshold=${diff_threshold},diff_change_threshold=${diff_change_threshold},diff_change_percent_threshold=${diff_change_percent_threshold},compression_method=${compression_method} \
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

echo "llava-ov-$model_size evaluation completed. Logs are saved in $log_dir."
