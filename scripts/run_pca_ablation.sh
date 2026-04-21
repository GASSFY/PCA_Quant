#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG_PATH="${CONFIG_PATH:-configs/default.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-ablation_runs}"
LOW_BIT="${LOW_BIT:-4}"
HIGH_BIT="${HIGH_BIT:-16}"
RATIO_ZERO="${RATIO_ZERO:-0}"
RATIO_MAIN="${RATIO_MAIN:-0.01}"
DELETE_INTERMEDIATE_PT="${DELETE_INTERMEDIATE_PT:-1}"
N_TRIALS_GLOBAL="${N_TRIALS_GLOBAL:-25}"
N_TRIALS_LAYERWISE="${N_TRIALS_LAYERWISE:-5}"
OPTUNA_SEED="${OPTUNA_SEED:-0}"

TASKS=("mmmu_val" "realworldqa" "ocrbench" "ai2d")


# 预留多模型实验：保持 MODEL_TYPES 与 MODEL_ARGSS 长度一致
MODEL_TYPES=(
  "internvl2"
)
MODEL_ARGSS=(
  "pretrained=OpenGVLab/InternVL2-8B"
)

if [ "${#MODEL_TYPES[@]}" -ne "${#MODEL_ARGSS[@]}" ]; then
  echo "[ERR] MODEL_TYPES 与 MODEL_ARGSS 长度不一致"
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"

# 从 model_args 中提取模型名称，用于唯一标识模型
derive_model_dir() {
  python3 -c "import re,sys
s=sys.argv[1]
m=re.search(r'pretrained=([^,]+)', s)
v=(m.group(1).strip() if m else 'model')
for c in ['/', '\\\\', ':', '*', '?', '\"', '<', '>', '|', ' ']:
    v=v.replace(c, '_')
print(v or 'model')
" "$1"
}

# 如果模型名称重复，则添加后缀
resolve_model_dir() {
  local idx="$1"
  local base
  base="$(derive_model_dir "${MODEL_ARGSS[$idx]}")"
  local j
  for ((j=0; j<idx; j++)); do
    if [ "$(derive_model_dir "${MODEL_ARGSS[$j]}")" = "$base" ]; then
      echo "${base}_${idx}"
      return
    fi
  done
  echo "$base"
}

# 根据任务选择评估指标
metric_for_task() {
  local task="$1"
  case "$task" in
    mmmu_val) echo "mmmu_acc" ;;
    realworldqa) echo "exact_match" ;;
    ocrbench) echo "ocrbench" ;;
    ai2d) echo "exact_match" ;;
    *) echo "acc" ;;
  esac
}

log_run() {
  local logfile="$1"
  shift
  "$@" 2>&1 | tee "$logfile"
}

write_log_header() {
  local logfile="$1"
  shift
  {
    echo "===== RUN HEADER ====="
    for line in "$@"; do
      echo "$line"
    done
    echo "======================"
  } > "$logfile"
}

append_run_log() {
  local logfile="$1"
  shift
  "$@" 2>&1 | tee -a "$logfile"
}

write_overridden_config() {
  local out_cfg="$1"
  local model="$2"
  local model_args="$3"
  local task="$4"
  local method="$5"
  local ratio="$6"
  local low_bit="$7"
  local high_bit="$8"

  python3 -c "import sys,yaml
src=sys.argv[1]
dst=sys.argv[2]
model=sys.argv[3]
model_args=sys.argv[4]
task=sys.argv[5]
method=sys.argv[6]
ratio=float(sys.argv[7])
low_bit=int(sys.argv[8])
high_bit=int(sys.argv[9])
with open(src,'r',encoding='utf-8') as f:
    cfg=yaml.safe_load(f) or {}
cfg['model']=model
cfg['model_args']=model_args
cfg['tasks']=task
cfg['method']=method
cfg['high_precision_ratio']=ratio
cfg['low_bit']=low_bit
cfg['high_bit']=high_bit
cfg['run_process']=True
with open(dst,'w',encoding='utf-8') as f:
    yaml.safe_dump(cfg,f,allow_unicode=True,sort_keys=False)
" "$CONFIG_PATH" "$out_cfg" "$model" "$model_args" "$task" "$method" "$ratio" "$low_bit" "$high_bit"
}

write_eval_config() {
  local out_cfg="$1"
  local model="$2"
  local model_args="$3"
  local task="$4"
  local scale_path="$5"
  local output_path="$6"

  python3 -c "import sys,yaml
src=sys.argv[1]
dst=sys.argv[2]
model=sys.argv[3]
model_args=sys.argv[4]
task=sys.argv[5]
scale_path=sys.argv[6]
output_path=sys.argv[7]
with open(src,'r',encoding='utf-8') as f:
    cfg=yaml.safe_load(f) or {}
cfg['model']=model
cfg['model_args']=model_args
cfg['tasks']=task
cfg['scale_path']=scale_path
cfg['output_path']=output_path
with open(dst,'w',encoding='utf-8') as f:
    yaml.safe_dump(cfg,f,allow_unicode=True,sort_keys=False)
" "$CONFIG_PATH" "$out_cfg" "$model" "$model_args" "$task" "$scale_path" "$output_path"
}

write_quant_config() {
  local out_cfg="$1"
  local model="$2"
  local model_args="$3"
  local method="$4"
  local ratio="$5"
  local beta="$6"
  local pca_k="$7"
  local low_bit="$8"
  local high_bit="$9"
  local scale_path="${10}"
  local results_path="${11}"
  local layerwise_json="${12}"

  python3 -c "import sys,yaml
src=sys.argv[1]
dst=sys.argv[2]
model=sys.argv[3]
model_args=sys.argv[4]
method=sys.argv[5]
ratio=float(sys.argv[6])
beta=float(sys.argv[7])
pca_k=int(sys.argv[8])
low_bit=int(sys.argv[9])
high_bit=int(sys.argv[10])
scale_path=sys.argv[11]
results_path=sys.argv[12]
layerwise_json=sys.argv[13]
with open(src,'r',encoding='utf-8') as f:
    cfg=yaml.safe_load(f) or {}
cfg['model']=model
cfg['model_args']=model_args
cfg['run_process']=True
cfg['method']=method
cfg['high_precision_ratio']=ratio
cfg['beta']=beta
cfg['pca_k']=pca_k
cfg['low_bit']=low_bit
cfg['high_bit']=high_bit
cfg['scale_path']=scale_path
cfg['results_path']=results_path
if layerwise_json:
    cfg['layerwise_params']=layerwise_json
with open(dst,'w',encoding='utf-8') as f:
    yaml.safe_dump(cfg,f,allow_unicode=True,sort_keys=False)
" "$CONFIG_PATH" "$out_cfg" "$model" "$model_args" "$method" "$ratio" "$beta" "$pca_k" "$low_bit" "$high_bit" "$scale_path" "$results_path" "$layerwise_json"
}

delete_if_needed() {
  local path="$1"
  if [ "$DELETE_INTERMEDIATE_PT" = "1" ] && [ -f "$path" ]; then
    rm -f "$path"
    echo "[CLEANUP] deleted ${path}"
  fi
}

run_eval() {
  local model="$1"
  local model_args="$2"
  local task="$3"
  local scale_path="$4"
  local out_dir="$5"
  local log_file="$6"
  local eval_cfg="${out_dir}/eval_config.yaml"

  mkdir -p "$out_dir"
  write_eval_config "$eval_cfg" "$model" "$model_args" "$task" "$scale_path" "$out_dir"
  write_log_header "$log_file" \
    "stage=eval" \
    "task=${task}" \
    "model=${model}" \
    "model_args=${model_args}" \
    "config=${eval_cfg}" \
    "effective_scale_path=${scale_path:-<empty>}" \
    "effective_output_path=${out_dir}"
  append_run_log "$log_file" python3 main_eval.py \
    --config "$eval_cfg"
}

run_quant() {
  local model="$1"
  local model_args="$2"
  local method="$3"
  local ratio="$4"
  local beta="$5"
  local pca_k="$6"
  local layerwise_json="$7"
  local scale_path="$8"
  local summary_path="$9"
  local log_file="${10}"
  local quant_cfg
  quant_cfg="$(dirname "$scale_path")/quant_config.yaml"

  write_quant_config "$quant_cfg" "$model" "$model_args" "$method" "$ratio" "$beta" "$pca_k" "$LOW_BIT" "$HIGH_BIT" "$scale_path" "$summary_path" "$layerwise_json"
  write_log_header "$log_file" \
    "stage=quant" \
    "model=${model}" \
    "model_args=${model_args}" \
    "method=${method}" \
    "ratio=${ratio}" \
    "beta=${beta}" \
    "pca_k=${pca_k}" \
    "config=${quant_cfg}" \
    "effective_scale_path=${scale_path}" \
    "effective_results_path=${summary_path}"
  append_run_log "$log_file" python3 main_quant.py --config "$quant_cfg"
}

run_case_quant_eval() {
  local model="$1"
  local model_args="$2"
  local task="$3"
  local case_dir="$4"
  local method="$5"
  local ratio="$6"
  local beta="$7"
  local pca_k="$8"
  local layerwise_json="${9:-}"

  local logs_dir="${case_dir}/logs"
  local scale_dir="${case_dir}/scale_cache"
  local eval_dir="${case_dir}/eval_out"
  mkdir -p "$logs_dir" "$scale_dir" "$eval_dir"

  local scale_path="${scale_dir}/quant.pt"
  local summary_path="${scale_dir}/quant.summary.json"
  local quant_log="${logs_dir}/quant.log"
  local eval_log="${logs_dir}/eval.log"

  run_quant "$model" "$model_args" "$method" "$ratio" "$beta" "$pca_k" "$layerwise_json" "$scale_path" "$summary_path" "$quant_log"
  run_eval "$model" "$model_args" "$task" "$scale_path" "$eval_dir" "$eval_log"
  delete_if_needed "$scale_path"
}

echo "============================================================"
echo "PCA_Quant ablation started"
echo "CONFIG_PATH=$CONFIG_PATH"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "LOW_BIT=$LOW_BIT, HIGH_BIT=$HIGH_BIT"
echo "RATIO_ZERO=$RATIO_ZERO, RATIO_MAIN=$RATIO_MAIN"
echo "DELETE_INTERMEDIATE_PT=$DELETE_INTERMEDIATE_PT"
echo "GLOBAL_OPTUNA_TRIALS=$N_TRIALS_GLOBAL, LAYERWISE_TRIALS=$N_TRIALS_LAYERWISE"
echo "============================================================"

for mi in "${!MODEL_TYPES[@]}"; do
  model="${MODEL_TYPES[$mi]}"
  model_args="${MODEL_ARGSS[$mi]}"
  model_dir="$(resolve_model_dir "$mi")"
  model_root="${OUTPUT_ROOT}/${model_dir}"
  mkdir -p "$model_root"

  echo
  echo "############################################################"
  echo "# Model: ${model}"
  echo "# model_args: ${model_args}"
  echo "# output: ${model_root}"
  echo "############################################################"

  for task in "${TASKS[@]}"; do
    task_root="${model_root}/${task}"
    mkdir -p "$task_root"
    echo
    echo "---------------- task=${task} ----------------"

    # 0) 原始模型基线（不加载量化权重）
    baseline_dir="${task_root}/baseline_fp"
    mkdir -p "${baseline_dir}/logs" "${baseline_dir}/eval_out"
    run_eval "$model" "$model_args" "$task" "" "${baseline_dir}/eval_out" "${baseline_dir}/logs/eval.log"

    # 1) ratio=0 + gate_only
    run_case_quant_eval \
      "$model" "$model_args" "$task" \
      "${task_root}/ratio0_gate_only" \
      "gate_only" "$RATIO_ZERO" "1.0" "32" ""

    # 2) ratio=0.01 + gate_only
    run_case_quant_eval \
      "$model" "$model_args" "$task" \
      "${task_root}/ratio001_gate_only" \
      "gate_only" "$RATIO_MAIN" "1.0" "32" ""

    # 3) ratio=0.01 + abs_only
    run_case_quant_eval \
      "$model" "$model_args" "$task" \
      "${task_root}/ratio001_abs_only" \
      "abs_only" "$RATIO_MAIN" "1.0" "32" ""

    # 4) ratio=0.01 + proj_log + 全局 beta Optuna（按任务单独）
    global_dir="${task_root}/ratio001_projlog_global_optuna"
    mkdir -p "${global_dir}/logs" "${global_dir}/tune_out"
    metric_sub="$(metric_for_task "$task")"
    global_cfg="${global_dir}/tune_out/tune_config.yaml"
    write_overridden_config "$global_cfg" "$model" "$model_args" "$task" "proj_log" "$RATIO_MAIN" "$LOW_BIT" "$HIGH_BIT"
    write_log_header "${global_dir}/logs/optuna.log" \
      "stage=optuna_global" \
      "task=${task}" \
      "model=${model}" \
      "config=${global_cfg}" \
      "output_dir=${global_dir}/tune_out" \
      "metric_substring=${metric_sub}"
    append_run_log "${global_dir}/logs/optuna.log" python3 scripts/optuna_tune.py \
      --config "$global_cfg" \
      --tasks "$task" \
      --metric-substring "$metric_sub" \
      --n-trials "$N_TRIALS_GLOBAL" \
      --seed "$OPTUNA_SEED" \
      --output-dir "${global_dir}/tune_out" \
      --discard-trial-artifacts

    best_json="${global_dir}/tune_out/best_params.json"
    best_beta="$(python3 -c "import json,sys; d=json.load(open(sys.argv[1],'r',encoding='utf-8')); print(d['best_params']['beta'])" "$best_json")"
    best_pca_k="$(python3 -c "import json,sys; d=json.load(open(sys.argv[1],'r',encoding='utf-8')); print(int(d['best_params']['pca_k']))" "$best_json")"
    run_case_quant_eval \
      "$model" "$model_args" "$task" \
      "$global_dir" \
      "proj_log" "$RATIO_MAIN" "$best_beta" "$best_pca_k" ""

    # 5) ratio=0.01 + proj_log + 分层 beta/pca_k Optuna
    layer_dir="${task_root}/ratio001_projlog_layerwise_optuna"
    mkdir -p "${layer_dir}/logs" "${layer_dir}/tune_out" "${layer_dir}/eval_out"
    layer_cfg="${layer_dir}/tune_out/tune_config.yaml"
    write_overridden_config "$layer_cfg" "$model" "$model_args" "$task" "proj_log" "$RATIO_MAIN" "$LOW_BIT" "$HIGH_BIT"
    write_log_header "${layer_dir}/logs/layerwise_tune.log" \
      "stage=optuna_layerwise" \
      "task=${task}" \
      "model=${model}" \
      "config=${layer_cfg}" \
      "output_dir=${layer_dir}/tune_out" \
      "n_trials_layer=${N_TRIALS_LAYERWISE}"
    append_run_log "${layer_dir}/logs/layerwise_tune.log" python3 scripts/optuna_tune_layerwise.py \
      --config "$layer_cfg" \
      --output-dir "${layer_dir}/tune_out" \
      --n-trials-layer "$N_TRIALS_LAYERWISE" \
      --seed "$OPTUNA_SEED"

    layer_pt="${layer_dir}/tune_out/layerwise_quant.pt"
    run_eval "$model" "$model_args" "$task" "$layer_pt" "${layer_dir}/eval_out" "${layer_dir}/logs/eval.log"
    delete_if_needed "$layer_pt"

    echo "[DONE] model=${model_dir} task=${task}"
  done
done

echo
echo "============================================================"
echo "All experiments finished."
echo "Results root: ${OUTPUT_ROOT}"
echo "Tips:"
echo "  1) 先做 smoke test：PCA_TASKS='mmmu_val' N_TRIALS_GLOBAL=3 N_TRIALS_LAYERWISE=2 bash scripts/run_pca_ablation.sh"
echo "  2) 如需保留量化权重：DELETE_INTERMEDIATE_PT=0 bash scripts/run_pca_ablation.sh"
echo "============================================================"
