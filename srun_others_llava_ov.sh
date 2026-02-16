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

# account_name="class-faculty" # yangli1-lab, bweng-lab
account_name=$(./select_account.sh)
echo "Selected account: $account_name"

partition_name="nova" # nova, interactive, scavenger(h200)
gpu_type="a100" # a100, h200, l40s
gpu_num=1 # 4 for 72b

# if has $1, assign to compression_method, else default to "original"
# before assigning to compression_method, check if $1 is in the allowed list, exit if not
# original, random, interval, vidcom2, fastvid, prunevid, dycoke
allowed_methods=("original" "random" "interval" "vidcom2" "fastvid" "prunevid" "dycoke" "fastv" "visionzip")
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

model_size="7b"
# if has $3, assign to model_size, else default to 7b
if [ -n "$3" ]; then
  model_size="$3"
fi

# if has --debug flag, assign to debug, else default to false
# if has --fastv_R=xxx flag, assign to fastv_R
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
  limit=1000
  cpu_memory="64G" # 384G for 72b
else
  log_dir="./logs/${exp_name}"
  # log_dir="./logs/random_25"
  # log_dir="./logs/density_1"
  # tasks="videomme,mlvu_dev,longvideobench_val_v,mvbench"
  # tasks="videomme"
  tasks="videomme"
  limit=1000000000
  cpu_memory="64G" # 384G for 72b
fi
echo "Logging to $log_dir"

start_time=$(date +%s)
echo "Start time: $(date)"

srun --account="$account_name" --time=24:00:00 --nodes=1 --cpus-per-task=8 --mem=${cpu_memory} --partition="$partition_name" --gres=gpu:"$gpu_type":"$gpu_num" \
  accelerate launch --num_processes="$gpu_num" \
  -m lmms_eval \
  --model llava_onevision \
  --model_args pretrained=lmms-lab/llava-onevision-qwen2-${model_size}-ov,conv_template=qwen_1_5,model_name=llava_qwen,attn_implementation=flash_attention_2 \
  --gen_kwargs max_new_tokens=16,temperature=0,top_p=1.0,num_beams=1,do_sample=False,base_scale=${base_scale},compression_method=${compression_method},fastv_R=${fastv_R} \
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