#!/usr/bin/env bash
# =============================================================================
# PCA_Quant 自动化对比实验脚本（简洁版）
# - 目标：按模型 / low_bit / ratio / method 组合自动量化并评估
# - 原则：尽量少的 bash 语法；通过 Python 小片段修改 YAML
# - 输出：ablation_results/<model>/<low_bit>/<ratio>/<method>/
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# 0) 基础路径与公共参数
# -----------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BASE_CONFIG="${BASE_CONFIG:-configs/default.yaml}"
OUT_ROOT="${OUT_ROOT:-ablation_results}"

# low_bit 列表
LOW_BITS=(2 3 4 6 8)

# 评估任务列表
TASKS=("mmmu_val" "realworldqa" "ocrbench" "ai2d")

# 是否删除中间量化权重（1=删除，0=保留）
DELETE_PT="${DELETE_PT:-1}"

# layerwise 调参 trial 数
N_TRIALS_LAYERWISE="${N_TRIALS_LAYERWISE:-5}"

# -----------------------------------------------------------------------------
# 1) 模型配置（按你的说明原样保留）
# -----------------------------------------------------------------------------
MODEL_TYPES=(
  "internvl2"
  "internvl2"
  "llava_onevision"
  "llava_v1.5"
)

MODEL_ARGSS=(
  "pretrained=OpenGVLab/InternVL2-8B"
  "pretrained=OpenGVLab/InternVL2-2B"
  "pretrained=lmms-lab/llava-onevision-qwen2-7b-ov"
  "pretrained=liuhaotian/llava-v1.5-7b"
)

if [ "${#MODEL_TYPES[@]}" -ne "${#MODEL_ARGSS[@]}" ]; then
  echo "[ERR] MODEL_TYPES 与 MODEL_ARGSS 长度不一致"
  exit 1
fi

# -----------------------------------------------------------------------------
# 2) 普通量化实验矩阵（gate_only / abs_only）
# -----------------------------------------------------------------------------
# 格式：ratio method
COMMON_CASES=(
  "0 gate_only"
  "0.01 gate_only"
  "0.01 abs_only"
  "0.1 gate_only"
  "0.1 abs_only"
)

# proj_log 走 layerwise 调参流程
PROJ_LOG_RATIOS=("0.01" "0.1")

# -----------------------------------------------------------------------------
# 3) 小工具：从 model_args 提取目录名
# -----------------------------------------------------------------------------
derive_model_dir() {
  python - "$1" <<'PY'
import re, sys
s = sys.argv[1]
m = re.search(r'pretrained=([^,]+)', s)
v = (m.group(1).strip() if m else "model")
for c in ["/", "\\", ":", "*", "?", "\"", "<", ">", "|", " "]:
    v = v.replace(c, "_")
print(v or "model")
PY
}

# -----------------------------------------------------------------------------
# 4) 小工具：更新 YAML 指定字段（简单可靠）
# -----------------------------------------------------------------------------
update_yaml() {
  local yaml_path="$1"
  local key="$2"
  local value="$3"
  local value_type="${4:-str}"  # str / int / float / bool / null

  python - "$yaml_path" "$key" "$value" "$value_type" <<'PY'
import sys, yaml
path, key, value, value_type = sys.argv[1:5]
with open(path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

if value_type == "int":
    cfg[key] = int(value)
elif value_type == "float":
    cfg[key] = float(value)
elif value_type == "bool":
    cfg[key] = value.lower() in ("1", "true", "yes")
elif value_type == "null":
    cfg[key] = None
else:
    cfg[key] = value

with open(path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
PY
}

# -----------------------------------------------------------------------------
# 5) 小工具：每次评测后把 results.json 改名为 results_<task>.json
# -----------------------------------------------------------------------------
rename_eval_result() {
  local eval_dir="$1"
  local task="$2"
  local src="${eval_dir}/results.json"
  local dst="${eval_dir}/results_${task}.json"
  if [ -f "$src" ]; then
    mv -f "$src" "$dst"
  fi
}

# -----------------------------------------------------------------------------
# 6) 普通量化流程：main_quant 一次 + main_eval 四次
# -----------------------------------------------------------------------------
run_common_case() {
  local model="$1"
  local model_args="$2"
  local low_bit="$3"
  local ratio="$4"
  local method="$5"
  local case_dir="$6"

  mkdir -p "${case_dir}/logs" "${case_dir}/scale_cache" "${case_dir}/eval_results"
  cp "$BASE_CONFIG" "${case_dir}/default.yaml"
  local cfg="${case_dir}/default.yaml"

  local scale_path="${case_dir}/scale_cache/pca_quant_model.pt"
  local summary_path="${case_dir}/scale_cache/summary.json"

  # --- 更新量化配置 ---
  update_yaml "$cfg" "model" "$model" "str"
  update_yaml "$cfg" "model_args" "$model_args" "str"
  update_yaml "$cfg" "method" "$method" "str"
  update_yaml "$cfg" "high_precision_ratio" "$ratio" "float"
  update_yaml "$cfg" "low_bit" "$low_bit" "int"
  update_yaml "$cfg" "scale_path" "$scale_path" "str"
  update_yaml "$cfg" "results_path" "$summary_path" "str"
  update_yaml "$cfg" "output_path" "${case_dir}/eval_results" "str"
  update_yaml "$cfg" "run_process" "true" "bool"

  echo "[RUN-QUANT] ${case_dir}"
  python main_quant.py --config "$cfg" 2>&1 | tee "${case_dir}/logs/quant.log"

  # --- 量化完成后，循环四个任务评测 ---
  update_yaml "$cfg" "run_process" "false" "bool"
  for task in "${TASKS[@]}"; do
    update_yaml "$cfg" "tasks" "$task" "str"
    echo "[RUN-EVAL] ${case_dir} task=${task}"
    python main_eval.py --config "$cfg" 2>&1 | tee -a "${case_dir}/logs/eval.log"
    rename_eval_result "${case_dir}/eval_results" "$task"
  done

  # --- 清理中间 .pt ---
  if [ "$DELETE_PT" = "1" ] && [ -f "$scale_path" ]; then
    rm -f "$scale_path"
    echo "[CLEANUP] Deleted ${scale_path}"
  fi
}

# -----------------------------------------------------------------------------
# 7) proj_log 流程：optuna_tune_layerwise + main_eval 四次
# -----------------------------------------------------------------------------
run_projlog_layerwise_case() {
  local model="$1"
  local model_args="$2"
  local low_bit="$3"
  local ratio="$4"
  local case_dir="$5"

  mkdir -p "${case_dir}/logs" "${case_dir}/layerwise_mse" "${case_dir}/eval_results"
  cp "$BASE_CONFIG" "${case_dir}/default.yaml"
  local cfg="${case_dir}/default.yaml"

  local layerwise_dir="${case_dir}/layerwise_mse"
  local scale_path="${layerwise_dir}/layerwise_quant.pt"
  local summary_path="${layerwise_dir}/summary.json"

  # --- 更新 layerwise 调参配置 ---
  update_yaml "$cfg" "model" "$model" "str"
  update_yaml "$cfg" "model_args" "$model_args" "str"
  update_yaml "$cfg" "method" "proj_log" "str"
  update_yaml "$cfg" "high_precision_ratio" "$ratio" "float"
  update_yaml "$cfg" "low_bit" "$low_bit" "int"
  update_yaml "$cfg" "scale_path" "$scale_path" "str"
  update_yaml "$cfg" "results_path" "" "str"
  update_yaml "$cfg" "output_path" "${case_dir}/eval_results" "str"
  update_yaml "$cfg" "run_process" "true" "bool"

  echo "[RUN-LAYERWISE] ${case_dir}"
  python scripts/optuna_tune_layerwise.py \
    --config "$cfg" \
    --n-trials-layer "$N_TRIALS_LAYERWISE" \
    --output-dir "$layerwise_dir" \
    --final-summary-name "summary.json" \
    2>&1 | tee "${case_dir}/logs/layerwise_tune.log"

  # --- 调参输出后，循环四个任务评测 ---
  update_yaml "$cfg" "run_process" "false" "bool"
  update_yaml "$cfg" "scale_path" "$scale_path" "str"
  update_yaml "$cfg" "results_path" "" "str"
  for task in "${TASKS[@]}"; do
    update_yaml "$cfg" "tasks" "$task" "str"
    echo "[RUN-EVAL] ${case_dir} task=${task}"
    python main_eval.py --config "$cfg" 2>&1 | tee -a "${case_dir}/logs/eval.log"
    rename_eval_result "${case_dir}/eval_results" "$task"
  done

  # --- 清理中间 .pt ---
  if [ "$DELETE_PT" = "1" ] && [ -f "$scale_path" ]; then
    rm -f "$scale_path"
    echo "[CLEANUP] Deleted ${scale_path}"
  fi
}

# -----------------------------------------------------------------------------
# 8) 主循环
# -----------------------------------------------------------------------------
mkdir -p "$OUT_ROOT"
echo "========================================================"
echo "[START] 自动化对比实验"
echo "BASE_CONFIG=${BASE_CONFIG}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "DELETE_PT=${DELETE_PT}"
echo "N_TRIALS_LAYERWISE=${N_TRIALS_LAYERWISE}"
echo "========================================================"

for i in "${!MODEL_TYPES[@]}"; do
  model="${MODEL_TYPES[$i]}"
  model_args="${MODEL_ARGSS[$i]}"
  model_dir="$(derive_model_dir "$model_args")"

  echo ""
  echo "########################################################"
  echo "[MODEL] ${model} | ${model_args}"
  echo "[DIR]   ${model_dir}"
  echo "########################################################"

  for low_bit in "${LOW_BITS[@]}"; do
    # 普通量化方法
    for case_item in "${COMMON_CASES[@]}"; do
      ratio="$(echo "$case_item" | awk '{print $1}')"
      method="$(echo "$case_item" | awk '{print $2}')"
      case_dir="${OUT_ROOT}/${model_dir}/low_bit_${low_bit}/ratio_${ratio}/${method}"
      run_common_case "$model" "$model_args" "$low_bit" "$ratio" "$method" "$case_dir"
    done

    # proj_log（layerwise）两组 ratio
    for ratio in "${PROJ_LOG_RATIOS[@]}"; do
      case_dir="${OUT_ROOT}/${model_dir}/low_bit_${low_bit}/ratio_${ratio}/proj_log"
      run_projlog_layerwise_case "$model" "$model_args" "$low_bit" "$ratio" "$case_dir"
    done
  done
done

echo ""
echo "========================================================"
echo "[DONE] 全部实验完成"
echo "结果目录: ${OUT_ROOT}"
echo "========================================================"
