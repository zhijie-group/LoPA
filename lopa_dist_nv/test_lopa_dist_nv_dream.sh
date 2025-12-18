#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

TARGET_SCRIPT="${ROOT_DIR}/generate_branch_parallel.py"
if [[ ! -f "${TARGET_SCRIPT}" ]]; then
  echo "Missing target script: ${TARGET_SCRIPT}" >&2
  exit 1
fi

export HF_ALLOW_CODE_EVAL="1"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONUNBUFFERED="1"
export DIST_BACKEND="nccl"
export HF_HOME="${HF_HOME:}"
# export HF_HOME="/root/autodl-tmp/D2F-Plus-Plus/.cache"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export D2F_CUDA_CLEANUP_FREE_RATIO="${D2F_CUDA_CLEANUP_FREE_RATIO:-0.20}"
export D2F_GC_CLEANUP_MARGIN="${D2F_GC_CLEANUP_MARGIN:-2.5}"
ENABLE_PROFILING="${ENABLE_PROFILING:-false}"

if [[ "${ENABLE_PROFILING}" == "true" ]]; then
  export D2F_PROFILE="1"
else
  export D2F_PROFILE="0"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_BRANCHES=${NUM_BRANCHES:-4}
MASTER_PORT=${MASTER_PORT:-29540}
RUN_NAME=${RUN_NAME:-$(date +"%Y%m%d-%H%M%S")}
RESULT_ROOT="${RESULT_ROOT:-${ROOT_DIR}/result/lopa_dist_nv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/result/lopa_dist_nv/output}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/log}"
ENABLE_RUN_LOG="${ENABLE_RUN_LOG:-true}"
ENABLE_SAMPLE_LOG="${ENABLE_SAMPLE_LOG:-false}"
DRY_RUN="${DRY_RUN:-false}"
RUN_TASK_SWEEP="${RUN_TASK_SWEEP:-true}"
RUN_HUMANEVAL_SWEEP="${RUN_HUMANEVAL_SWEEP:-false}"
case "$RUN_MODE" in
  "base")
    APPLY_CHAT_TEMPLATE_TASKS="${APPLY_CHAT_TEMPLATE_TASKS:-false}"
    FEWSHOT_AS_MULTITURN_TASKS="${FEWSHOT_AS_MULTITURN_TASKS:-false}"
    ;;
  "instruct")
    APPLY_CHAT_TEMPLATE_TASKS="${APPLY_CHAT_TEMPLATE_TASKS:-true}"
    FEWSHOT_AS_MULTITURN_TASKS="${FEWSHOT_AS_MULTITURN_TASKS:-true}"
    ;;
  *)
    APPLY_CHAT_TEMPLATE_TASKS="${APPLY_CHAT_TEMPLATE_TASKS:-true}"
    FEWSHOT_AS_MULTITURN_TASKS="${FEWSHOT_AS_MULTITURN_TASKS:-true}"
    ;;
esac
# Ensure consistency: apply_chat_template and fewshot_as_multiturn must be both true or both false
if [[ "${FEWSHOT_AS_MULTITURN_TASKS}" == "true" && "${APPLY_CHAT_TEMPLATE_TASKS}" != "true" ]]; then
  echo "WARNING: FEWSHOT_AS_MULTITURN_TASKS is true but APPLY_CHAT_TEMPLATE_TASKS is not. Forcing APPLY_CHAT_TEMPLATE_TASKS=true" >&2
  APPLY_CHAT_TEMPLATE_TASKS="true"
elif [[ "${FEWSHOT_AS_MULTITURN_TASKS}" != "true" && "${APPLY_CHAT_TEMPLATE_TASKS}" == "true" ]]; then
  echo "WARNING: APPLY_CHAT_TEMPLATE_TASKS is true but FEWSHOT_AS_MULTITURN_TASKS is not. Forcing FEWSHOT_AS_MULTITURN_TASKS=true" >&2
  FEWSHOT_AS_MULTITURN_TASKS="true"
fi
APPLY_CHAT_TEMPLATE_HUMANEVAL="${APPLY_CHAT_TEMPLATE_HUMANEVAL:-false}"
FEWSHOT_AS_MULTITURN_HUMANEVAL="${FEWSHOT_AS_MULTITURN_HUMANEVAL:-false}"
# Ensure consistency: apply_chat_template and fewshot_as_multiturn must be both true or both false
if [[ "${FEWSHOT_AS_MULTITURN_HUMANEVAL}" == "true" && "${APPLY_CHAT_TEMPLATE_HUMANEVAL}" != "true" ]]; then
  echo "WARNING: FEWSHOT_AS_MULTITURN_HUMANEVAL is true but APPLY_CHAT_TEMPLATE_HUMANEVAL is not. Forcing APPLY_CHAT_TEMPLATE_HUMANEVAL=true" >&2
  APPLY_CHAT_TEMPLATE_HUMANEVAL="true"
elif [[ "${FEWSHOT_AS_MULTITURN_HUMANEVAL}" != "true" && "${APPLY_CHAT_TEMPLATE_HUMANEVAL}" == "true" ]]; then
  echo "WARNING: APPLY_CHAT_TEMPLATE_HUMANEVAL is true but FEWSHOT_AS_MULTITURN_HUMANEVAL is not. Forcing FEWSHOT_AS_MULTITURN_HUMANEVAL=true" >&2
  FEWSHOT_AS_MULTITURN_HUMANEVAL="true"
fi
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}
EVAL_EXTRA_ARGS="${EVAL_EXTRA_ARGS:-}"  # space-separated extra CLI switches appended to every eval command
MODEL_ARGS_APPEND="${MODEL_ARGS_APPEND:-}"  # optional comma-separated extras appended to every model_args string
LOG_NAME="${LOG_NAME:-}"
GLOBAL_LOG_FILE="${LOG_FILE:-${LOG_NAME:-}}"

mkdir -p "${RESULT_ROOT}" "${OUTPUT_ROOT}" "${LOG_DIR}"

# --------------------------------------------------------------------------------------
# Base model / LoRA configuration
# --------------------------------------------------------------------------------------

MODEL_PRETRAINED=${MODEL_PRETRAINED:}
MODEL_LORA_PATH_DEFAULT=${MODEL_LORA_PATH_DEFAULT:}
LORA_MODELS_INPUT="${LORA_MODELS:-${MODEL_LORA_PATH_DEFAULT}}"
read -r -a LORA_MODELS <<< "${LORA_MODELS_INPUT}"
if [[ ${#LORA_MODELS[@]} -eq 0 ]]; then
  echo "At least one LoRA model path must be provided via LORA_MODELS" >&2
  exit 1
fi

MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-4096}
MODEL_TOP_K=${MODEL_TOP_K:-}
case "$RUN_MODE" in
  "base")
    MODEL_ADD_BOS_TOKEN=${MODEL_ADD_BOS_TOKEN:-true}
    ;;
  "instruct")
    MODEL_ADD_BOS_TOKEN=${MODEL_ADD_BOS_TOKEN:-false}
    ;;
  *)
    MODEL_ADD_BOS_TOKEN=${MODEL_ADD_BOS_TOKEN:-true}
    ;;
esac

MODEL_ESCAPE_UNTIL=${MODEL_ESCAPE_UNTIL:-true}
MODEL_MASK_TOKEN_ID=${MODEL_MASK_TOKEN_ID:-151666}
MODEL_ATTENTION_IMPL=${MODEL_ATTENTION_IMPL:-flash_attention_2}
MODEL_TORCH_COMPILE=${MODEL_TORCH_COMPILE:-false}
MODEL_TORCH_COMPILE_MODE=${MODEL_TORCH_COMPILE_MODE:-reduce-overhead}
MODEL_USE_UNCERTAINTY_LOGIC=${MODEL_USE_UNCERTAINTY_LOGIC:-true}
MODEL_USE_FULL_ATTENTION=${MODEL_USE_FULL_ATTENTION:-true}
MODEL_USE_SAGE_ATTENTION=${MODEL_USE_SAGE_ATTENTION:-false}
MODEL_BRANCH_CONFIDENCE_DECAY=${MODEL_BRANCH_CONFIDENCE_DECAY:-0.8}
MODEL_SHOW_SPEED=${MODEL_SHOW_SPEED:-true}
MODEL_SHOW_BRANCH_DETAILS=${MODEL_SHOW_BRANCH_DETAILS:-false}
MODEL_PROFILE_LOGGING=${MODEL_PROFILE_LOGGING:-}
MODEL_ALG=${MODEL_ALG:-}
MODEL_ALG_TEMP=${MODEL_ALG_TEMP:-}
MODEL_NLL_TYPE=${MODEL_NLL_TYPE:-}
MODEL_LOG_TYPE=${MODEL_LOG_TYPE:-}
MODEL_MC_NUM=${MODEL_MC_NUM:-}
MODEL_CLASSIFIER_FREE_GUIDANCE=${MODEL_CLASSIFIER_FREE_GUIDANCE:-}
MODEL_SAMPLING_EPS=${MODEL_SAMPLING_EPS:-}
MODEL_PARALLELIZE=${MODEL_PARALLELIZE:-}
MODEL_AUTOGPTQ=${MODEL_AUTOGPTQ:-}
MODEL_TRUST_REMOTE_CODE=${MODEL_TRUST_REMOTE_CODE:-true}
MODEL_DEBUG_PRINT=${MODEL_DEBUG_PRINT:-}
MODEL_FORCE_BASE_BRANCH=${MODEL_FORCE_BASE_BRANCH:-false}

# Optimization flags
MODEL_MERGE_LORA_WEIGHTS=${MODEL_MERGE_LORA_WEIGHTS:-true}
MODEL_ENABLE_PREFIX_CACHE=${MODEL_ENABLE_PREFIX_CACHE:-true}
MODEL_MERGE_QKV_PROJECTIONS=${MODEL_MERGE_QKV_PROJECTIONS:-false}

if [[ -n "${MODEL_ARGS:-}" ]]; then
  echo "MODEL_ARGS override is not supported in test_lopa_dist_nv_dream.sh. Use MODEL_ARGS_APPEND for extra key/value pairs." >&2
  exit 1
fi

# --------------------------------------------------------------------------------------
# Task sweep defaults (mirrors the active block in eval_dream7.sh)
# --------------------------------------------------------------------------------------
# ==============================================================================
# Expansion Configuration for 4 Threshold Groups
# Each group runs: mbpp, gsm8k
# Total runs: 8
# ==============================================================================

# ==============================================================================
# 1. Expand Tasks & Static Params (Repeat 6 times -> Total 12 entries)
# ==============================================================================
# 6 Groups * 2 Tasks (MBPP + GSM8K) = 12 Entries
TASKS="${TASKS:-mbpp gsm8k mbpp gsm8k mbpp gsm8k mbpp gsm8k mbpp gsm8k mbpp gsm8k}"
# case "$RUN_MODE" in
#   "base")
#     NSHOTS="${NSHOTS:-3 4 3 4 3 4 3 4 3 4 3 4}"
#     ;;
#   "instruct")
#     NSHOTS="${NSHOTS:-0 0 0 0 0 0 0 0 0 0 0 0}"
#     ;;
# esac
# NSHOTS="${NSHOTS:-0 0 0 0 0 0 0 0 0 0 0 0}"
NSHOTS="${NSHOTS:-3 4 3 4 3 4 3 4 3 4 3 4}"
LENGTHS="${LENGTHS:-512 512 512 512 512 512 512 512 512 512 512 512}"
TEMPERATURES="${TEMPERATURES:-0 0 0 0 0 0 0 0 0 0 0 0}"
LIMITS="${LIMITS:-10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-512 512 512 512 512 512 512 512 512 512 512 512}"
BLOCK_SIZES="${BLOCK_SIZES:-32 32 32 32 32 32 32 32 32 32 32 32}"

# ==============================================================================
# 2. Variable Thresholds (6 Groups)
# ==============================================================================
# Mapping based on your Table:
# Group 1: Add=0.1, Act=0.95, Conf=0.95
# Group 2: Add=0.1, Act=0.90, Conf=0.90
# Group 3: Add=0.1, Act=0.85, Conf=0.85
# Group 4: Add=0.1, Act=0.80, Conf=0.80
# Group 5: Add=0.1, Act=0.75, Conf=0.75
# Group 6: Add=0.1, Act=0.70, Conf=0.70

# tau_add (Fixed at 0.1 for all)
BLOCK_ADD_THRESHOLDS="${BLOCK_ADD_THRESHOLDS:-0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1}"

# tau_act (Decoded Token Threshold)
# Values: 0.95, 0.90, 0.85, 0.80, 0.75, 0.70 (Repeated twice for mbpp/gsm8k)
DECODED_TOKEN_THRESHOLDS="${DECODED_TOKEN_THRESHOLDS:-0.95 0.95 0.90 0.90 0.85 0.85 0.80 0.80 0.75 0.75 0.70 0.70}"

# tau_conf (Skip Threshold) - Assuming same as tau_act based on table
SKIP_THRESHOLDS="${SKIP_THRESHOLDS:-0.95 0.95 0.90 0.90 0.85 0.85 0.80 0.80 0.75 0.75 0.70 0.70}"

# ==============================================================================
# 3. Expand Other Params (Repeat 6 times -> Total 12 entries)
# ==============================================================================
TOP_PS="${TOP_PS:-none none none none none none none none none none none none}"
DTYPES="${DTYPES:-bfloat16 bfloat16 bfloat16 bfloat16 bfloat16 bfloat16 bfloat16 bfloat16 bfloat16 bfloat16 bfloat16 bfloat16}"
SAMPLING_STRATEGIES="${SAMPLING_STRATEGIES:-neg_entropy neg_entropy neg_entropy neg_entropy neg_entropy neg_entropy neg_entropy neg_entropy neg_entropy neg_entropy neg_entropy neg_entropy}"
MAX_BRANCHES_KEPTS="${MAX_BRANCHES_KEPTS:-1 1 1 1 1 1 1 1 1 1 1 1}"

# Keeping ${NUM_BRANCHES} dynamic as requested, just expanding the count to 12
BRANCHING_FACTORS="${BRANCHING_FACTORS:-${NUM_BRANCHES} ${NUM_BRANCHES} ${NUM_BRANCHES} ${NUM_BRANCHES} ${NUM_BRANCHES} ${NUM_BRANCHES} ${NUM_BRANCHES} ${NUM_BRANCHES} ${NUM_BRANCHES} ${NUM_BRANCHES} ${NUM_BRANCHES} ${NUM_BRANCHES}}"

BRANCH_TOPPS="${BRANCH_TOPPS:-1 1 1 1 1 1 1 1 1 1 1 1}"
SELECTION_CONF_ALPHAS="${SELECTION_CONF_ALPHAS:-0 0 0 0 0 0 0 0 0 0 0 0}"
BRANCH_VERIFICATION_MODES="${BRANCH_VERIFICATION_MODES:-true true true true true true true true true true true true}"
BASE_BRANCH_COMPETITIONS="${BASE_BRANCH_COMPETITIONS:-true true true true true true true true true true true true}"
VERIFICATION_FORCE_BASE_WINNERS="${VERIFICATION_FORCE_BASE_WINNERS:-false false false false false false false false false false false false}"

# TASKS="${TASKS:-mbpp gsm8k}"
# NSHOTS="${NSHOTS:-3 4}"
# LENGTHS="${LENGTHS:-512 512}"
# TEMPERATURES="${TEMPERATURES:-0 0}"
# LIMITS="${LIMITS:-10000 10000}"
# DIFFUSION_STEPS="${DIFFUSION_STEPS:-512 512}"
# BLOCK_SIZES="${BLOCK_SIZES:-32 32}"
# BLOCK_ADD_THRESHOLDS="${BLOCK_ADD_THRESHOLDS:-0.1 0.1}"
# DECODED_TOKEN_THRESHOLDS="${DECODED_TOKEN_THRESHOLDS:-0.70 0.70}"
# SKIP_THRESHOLDS="${SKIP_THRESHOLDS:-0.70 0.70}"
# TOP_PS="${TOP_PS:-none none}"
# DTYPES="${DTYPES:-bfloat16 bfloat16}"
# SAMPLING_STRATEGIES="${SAMPLING_STRATEGIES:-neg_entropy neg_entropy}"
# MAX_BRANCHES_KEPTS="${MAX_BRANCHES_KEPTS:-1 1}"
# BRANCHING_FACTORS="${BRANCHING_FACTORS:-${NUM_BRANCHES} ${NUM_BRANCHES}}"
# BRANCH_TOPPS="${BRANCH_TOPPS:-1 1}"
# SELECTION_CONF_ALPHAS="${SELECTION_CONF_ALPHAS:-0 0}"
# BRANCH_VERIFICATION_MODES="${BRANCH_VERIFICATION_MODES:-false false}"
# BASE_BRANCH_COMPETITIONS="${BASE_BRANCH_COMPETITIONS:-true true}"
# VERIFICATION_FORCE_BASE_WINNERS="${VERIFICATION_FORCE_BASE_WINNERS:-false false}"

# HumanEval sweep defaults (disabled unless RUN_HUMANEVAL_SWEEP=true)
HUMANEVAL_NSHOTS="${HUMANEVAL_NSHOTS:-0}"
HUMANEVAL_LENGTHS="${HUMANEVAL_LENGTHS:-512}"
HUMANEVAL_TEMPERATURES="${HUMANEVAL_TEMPERATURES:-0}"
HUMANEVAL_LIMITS="${HUMANEVAL_LIMITS:-10000}"
HUMANEVAL_DIFFUSION_STEPS="${HUMANEVAL_DIFFUSION_STEPS:-512}"
HUMANEVAL_BLOCK_SIZES="${HUMANEVAL_BLOCK_SIZES:-32}"
HUMANEVAL_BLOCK_ADD_THRESHOLDS="${HUMANEVAL_BLOCK_ADD_THRESHOLDS:-0.1}"
HUMANEVAL_DECODED_TOKEN_THRESHOLDS="${HUMANEVAL_DECODED_TOKEN_THRESHOLDS:-0.7}"
HUMANEVAL_SKIP_THRESHOLDS="${HUMANEVAL_SKIP_THRESHOLDS:-0.7}"
HUMANEVAL_TOP_PS="${HUMANEVAL_TOP_PS:-none}"
HUMANEVAL_DTYPES="${HUMANEVAL_DTYPES:-bfloat16}"
HUMANEVAL_SAMPLING_STRATEGIES="${HUMANEVAL_SAMPLING_STRATEGIES:-neg_entropy}"
HUMANEVAL_MAX_BRANCHES_KEPTS="${HUMANEVAL_MAX_BRANCHES_KEPTS:-1}"
HUMANEVAL_BRANCHING_FACTORS="${HUMANEVAL_BRANCHING_FACTORS:-${NUM_BRANCHES}}"
HUMANEVAL_BRANCH_TOPPS="${HUMANEVAL_BRANCH_TOPPS:-1}"
HUMANEVAL_SELECTION_CONF_ALPHAS="${HUMANEVAL_SELECTION_CONF_ALPHAS:-0}"
HUMANEVAL_BRANCH_VERIFICATION_MODES="${HUMANEVAL_BRANCH_VERIFICATION_MODES:-true}"
HUMANEVAL_BASE_BRANCH_COMPETITIONS="${HUMANEVAL_BASE_BRANCH_COMPETITIONS:-true}"
HUMANEVAL_VERIFICATION_FORCE_BASE_WINNERS="${HUMANEVAL_VERIFICATION_FORCE_BASE_WINNERS:-false}"

# --------------------------------------------------------------------------------------
# Helper utilities
# --------------------------------------------------------------------------------------
slugify() {
  local input="$1"
  input="${input// /_}"
  input="${input//[^a-zA-Z0-9_.-]/_}"
  printf '%s' "${input}"
}

append_arg() {
  local arr_name="$1"
  local key="$2"
  local value="$3"
  if [[ -z "${value}" ]]; then
    return
  fi
  declare -n arr_ref="${arr_name}"
  arr_ref+=("${key}=${value}")
}

build_model_args() {
  local -a args=()
  append_arg args "pretrained" "${MODEL_PRETRAINED}"
  append_arg args "lora_path" "${MODEL_LORA_PATH}"
  append_arg args "max_new_tokens" "${MODEL_MAX_NEW_TOKENS}"
  append_arg args "max_length" "${MODEL_MAX_LENGTH}"
  append_arg args "diffusion_steps" "${MODEL_DIFFUSION_STEPS}"
  append_arg args "temperature" "${MODEL_TEMPERATURE}"
  append_arg args "top_p" "${MODEL_TOP_P}"
  append_arg args "top_k" "${MODEL_TOP_K}"
  append_arg args "add_bos_token" "${MODEL_ADD_BOS_TOKEN}"
  append_arg args "escape_until" "${MODEL_ESCAPE_UNTIL}"
  append_arg args "block_size" "${MODEL_BLOCK_SIZE}"
  append_arg args "block_add_threshold" "${MODEL_BLOCK_ADD_THRESHOLD}"
  append_arg args "decoded_token_threshold" "${MODEL_DECODED_TOKEN_THRESHOLD}"
  append_arg args "skip_threshold" "${MODEL_SKIP_THRESHOLD}"
  append_arg args "mask_token_id" "${MODEL_MASK_TOKEN_ID}"
  append_arg args "dtype" "${MODEL_DTYPE}"
  append_arg args "sampling_strategy" "${MODEL_SAMPLING_STRATEGY}"
  append_arg args "max_branches_kept" "${MODEL_MAX_BRANCHES_KEPT}"
  append_arg args "branching_factor" "${MODEL_BRANCHING_FACTOR}"
  append_arg args "branch_topp" "${MODEL_BRANCH_TOPP}"
  append_arg args "selection_conf_alpha" "${MODEL_SELECTION_CONF_ALPHA}"
  append_arg args "branch_verification_mode" "${MODEL_BRANCH_VERIFICATION_MODE}"
  append_arg args "base_branch_competition" "${MODEL_BASE_BRANCH_COMPETITION}"
  append_arg args "verification_force_base_winner" "${MODEL_FORCE_BASE_BRANCH}"
  append_arg args "branch_confidence_decay" "${MODEL_BRANCH_CONFIDENCE_DECAY}"
  append_arg args "use_uncertainty_logic" "${MODEL_USE_UNCERTAINTY_LOGIC}"
  append_arg args "use_full_attention" "${MODEL_USE_FULL_ATTENTION}"
  append_arg args "use_sage_attention" "${MODEL_USE_SAGE_ATTENTION}"
  append_arg args "attn_implementation" "${MODEL_ATTENTION_IMPL}"
  append_arg args "torch_compile" "${MODEL_TORCH_COMPILE}"
  append_arg args "torch_compile_mode" "${MODEL_TORCH_COMPILE_MODE}"
  append_arg args "save_dir" "${MODEL_SAVE_DIR}"
  append_arg args "show_speed" "${MODEL_SHOW_SPEED}"
  append_arg args "show_branch_details" "${MODEL_SHOW_BRANCH_DETAILS}"
  append_arg args "profile_logging" "${MODEL_PROFILE_LOGGING}"
  append_arg args "alg" "${MODEL_ALG}"
  append_arg args "alg_temp" "${MODEL_ALG_TEMP}"
  append_arg args "nll_type" "${MODEL_NLL_TYPE}"
  append_arg args "log_type" "${MODEL_LOG_TYPE}"
  append_arg args "mc_num" "${MODEL_MC_NUM}"
  append_arg args "classifier_free_guidance" "${MODEL_CLASSIFIER_FREE_GUIDANCE}"
  append_arg args "sampling_eps" "${MODEL_SAMPLING_EPS}"
  append_arg args "parallelize" "${MODEL_PARALLELIZE}"
  append_arg args "autogptq" "${MODEL_AUTOGPTQ}"
  append_arg args "trust_remote_code" "${MODEL_TRUST_REMOTE_CODE}"
  append_arg args "debug_print" "${MODEL_DEBUG_PRINT}"
  append_arg args "experiment_label" "${MODEL_EXPERIMENT_LABEL}"
  append_arg args "merge_lora_weights" "${MODEL_MERGE_LORA_WEIGHTS}"
  append_arg args "enable_prefix_cache" "${MODEL_ENABLE_PREFIX_CACHE}"
  append_arg args "merge_qkv_projections" "${MODEL_MERGE_QKV_PROJECTIONS}"

  local IFS=','
  local joined="${args[*]}"
  if [[ -n "${MODEL_ARGS_APPEND}" ]]; then
    if [[ -n "${joined}" ]]; then
      printf '%s\n' "${joined},${MODEL_ARGS_APPEND}"
    else
      printf '%s\n' "${MODEL_ARGS_APPEND}"
    fi
  else
    printf '%s\n' "${joined}"
  fi
}

require_equal_length() {
  local expected="$1"
  shift
  for name in "$@"; do
    declare -n arr_ref="${name}"
    if [[ ${#arr_ref[@]} -ne ${expected} ]]; then
      echo "Configuration error: array ${name} has length ${#arr_ref[@]} but expected ${expected}" >&2
      exit 1
    fi
  done
}

expand_array_to_length() {
  local name="$1"
  local target="$2"
  declare -n arr_ref="${name}"
  local current=${#arr_ref[@]}
  if [[ ${target} -le 0 ]]; then
    return
  fi
  if [[ ${current} -eq 0 ]]; then
    echo "Configuration error: array ${name} is empty but needs ${target} entries" >&2
    exit 1
  fi
  if [[ ${current} -eq 1 && ${target} -gt 1 ]]; then
    local value="${arr_ref[0]}"
    arr_ref=()
    for ((i=0; i<target; i++)); do
      arr_ref+=("${value}")
    done
  elif [[ ${current} -ne ${target} ]]; then
    echo "Configuration error: array ${name} length ${current} incompatible with target ${target}" >&2
    exit 1
  fi
}

build_cmd_and_run() {
  local run_slug="$1"
  local model_args="$2"
  local task_name="$3"
  local limit="$4"
  local nshot="$5"
  local output_path="$6"
  local apply_chat="$7"
  local fewshot_multiturn="$8"

  mkdir -p "$(dirname "${output_path}")"

  local -a cmd=(
    "${PYTHON_BIN}" -m torch.distributed.run
    --nproc_per_node "${NUM_BRANCHES}"
    --standalone
    --rdzv_backend c10d
    --rdzv_endpoint "localhost:${MASTER_PORT}"
    "${TARGET_SCRIPT}"
    --model dream_lora_bp
    --model_args "${model_args}"
    --tasks "${task_name}"
    --limit "${limit}"
    --num_fewshot "${nshot}"
    --batch_size "${EVAL_BATCH_SIZE}"
    --output_path "${output_path}"
    --confirm_run_unsafe_code
  )

  if [[ "${apply_chat}" == "true" ]]; then
    cmd+=(--apply_chat_template)
  fi
  if [[ "${fewshot_multiturn}" == "true" ]]; then
    cmd+=(--fewshot_as_multiturn)
  fi
  if [[ "${ENABLE_SAMPLE_LOG}" == "true" ]]; then
    cmd+=(--log_samples)
  fi
  if [[ -n "${EVAL_EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    local -a extra_args=( ${EVAL_EXTRA_ARGS} )
    cmd+=("${extra_args[@]}")
  fi

  echo "--------------------------------------------------------------------------------"
  echo "[Run ${run_slug}] task=${task_name} shots=${nshot} limit=${limit} save_dir=${MODEL_SAVE_DIR}"
  echo "MODEL_ARGS=${model_args}"

  if [[ "${DRY_RUN}" == "true" ]]; then
    printf 'DRY-RUN CMD: '
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return
  fi

  local log_target=""
  if [[ -n "${GLOBAL_LOG_FILE}" ]]; then
    log_target="${GLOBAL_LOG_FILE}"
  elif [[ "${ENABLE_RUN_LOG}" == "true" ]]; then
    log_target="${LOG_DIR}/${run_slug}.log"
  fi

  if [[ -n "${log_target}" ]]; then
    mkdir -p "$(dirname "${log_target}")"
    "${cmd[@]}" 2>&1 | tee "${log_target}"
  else
    "${cmd[@]}"
  fi
}

# --------------------------------------------------------------------------------------
# Convert string configs to arrays & validate
# --------------------------------------------------------------------------------------
read -r -a TASKS_ARRAY <<< "${TASKS}"
read -r -a NSHOTS_ARRAY <<< "${NSHOTS}"
read -r -a LENGTH_ARRAY <<< "${LENGTHS}"
read -r -a TEMP_ARRAY <<< "${TEMPERATURES}"
read -r -a LIMITS_ARRAY <<< "${LIMITS}"
read -r -a DIFFUSION_STEPS_ARRAY <<< "${DIFFUSION_STEPS}"
read -r -a BLOCK_SIZES_ARRAY <<< "${BLOCK_SIZES}"
read -r -a BLOCK_ADD_THRESHOLDS_ARRAY <<< "${BLOCK_ADD_THRESHOLDS}"
read -r -a DECODED_TOKEN_THRESHOLDS_ARRAY <<< "${DECODED_TOKEN_THRESHOLDS}"
read -r -a SKIP_THRESHOLDS_ARRAY <<< "${SKIP_THRESHOLDS}"
read -r -a TOP_PS_ARRAY <<< "${TOP_PS}"
read -r -a DTYPES_ARRAY <<< "${DTYPES}"
read -r -a SAMPLING_STRATEGIES_ARRAY <<< "${SAMPLING_STRATEGIES}"
read -r -a MAX_BRANCHES_KEPTS_ARRAY <<< "${MAX_BRANCHES_KEPTS}"
read -r -a BRANCHING_FACTORS_ARRAY <<< "${BRANCHING_FACTORS}"
read -r -a BRANCH_TOPPS_ARRAY <<< "${BRANCH_TOPPS}"
read -r -a SELECTION_CONF_ALPHAS_ARRAY <<< "${SELECTION_CONF_ALPHAS}"
read -r -a BRANCH_VERIFICATION_MODES_ARRAY <<< "${BRANCH_VERIFICATION_MODES}"
read -r -a BASE_BRANCH_COMPETITIONS_ARRAY <<< "${BASE_BRANCH_COMPETITIONS}"
read -r -a VERIFICATION_FORCE_BASE_WINNERS_ARRAY <<< "${VERIFICATION_FORCE_BASE_WINNERS}"

TASK_COUNT=${#TASKS_ARRAY[@]}
if [[ ${TASK_COUNT} -gt 0 ]]; then
  require_equal_length "${TASK_COUNT}" \
    NSHOTS_ARRAY LENGTH_ARRAY TEMP_ARRAY LIMITS_ARRAY DIFFUSION_STEPS_ARRAY BLOCK_SIZES_ARRAY \
    BLOCK_ADD_THRESHOLDS_ARRAY DECODED_TOKEN_THRESHOLDS_ARRAY SKIP_THRESHOLDS_ARRAY \
    TOP_PS_ARRAY DTYPES_ARRAY SAMPLING_STRATEGIES_ARRAY
  expand_array_to_length MAX_BRANCHES_KEPTS_ARRAY "${TASK_COUNT}"
  expand_array_to_length BRANCHING_FACTORS_ARRAY "${TASK_COUNT}"
  expand_array_to_length BRANCH_TOPPS_ARRAY "${TASK_COUNT}"
  expand_array_to_length SELECTION_CONF_ALPHAS_ARRAY "${TASK_COUNT}"
  expand_array_to_length BRANCH_VERIFICATION_MODES_ARRAY "${TASK_COUNT}"
  expand_array_to_length BASE_BRANCH_COMPETITIONS_ARRAY "${TASK_COUNT}"
  expand_array_to_length VERIFICATION_FORCE_BASE_WINNERS_ARRAY "${TASK_COUNT}"
fi

read -r -a HUMANEVAL_NSHOTS_ARRAY <<< "${HUMANEVAL_NSHOTS}"
read -r -a HUMANEVAL_LENGTHS_ARRAY <<< "${HUMANEVAL_LENGTHS}"
read -r -a HUMANEVAL_TEMPS_ARRAY <<< "${HUMANEVAL_TEMPERATURES}"
read -r -a HUMANEVAL_LIMITS_ARRAY <<< "${HUMANEVAL_LIMITS}"
read -r -a HUMANEVAL_DIFFUSION_STEPS_ARRAY <<< "${HUMANEVAL_DIFFUSION_STEPS}"
read -r -a HUMANEVAL_BLOCK_SIZES_ARRAY <<< "${HUMANEVAL_BLOCK_SIZES}"
read -r -a HUMANEVAL_BLOCK_ADD_THRESHOLDS_ARRAY <<< "${HUMANEVAL_BLOCK_ADD_THRESHOLDS}"
read -r -a HUMANEVAL_DECODED_TOKEN_THRESHOLDS_ARRAY <<< "${HUMANEVAL_DECODED_TOKEN_THRESHOLDS}"
read -r -a HUMANEVAL_SKIP_THRESHOLDS_ARRAY <<< "${HUMANEVAL_SKIP_THRESHOLDS}"
read -r -a HUMANEVAL_TOP_PS_ARRAY <<< "${HUMANEVAL_TOP_PS}"
read -r -a HUMANEVAL_DTYPES_ARRAY <<< "${HUMANEVAL_DTYPES}"
read -r -a HUMANEVAL_SAMPLING_STRATEGIES_ARRAY <<< "${HUMANEVAL_SAMPLING_STRATEGIES}"
read -r -a HUMANEVAL_MAX_BRANCHES_KEPTS_ARRAY <<< "${HUMANEVAL_MAX_BRANCHES_KEPTS}"
read -r -a HUMANEVAL_BRANCHING_FACTORS_ARRAY <<< "${HUMANEVAL_BRANCHING_FACTORS}"
read -r -a HUMANEVAL_BRANCH_TOPPS_ARRAY <<< "${HUMANEVAL_BRANCH_TOPPS}"
read -r -a HUMANEVAL_SELECTION_CONF_ALPHAS_ARRAY <<< "${HUMANEVAL_SELECTION_CONF_ALPHAS}"
read -r -a HUMANEVAL_BRANCH_VERIFICATION_MODES_ARRAY <<< "${HUMANEVAL_BRANCH_VERIFICATION_MODES}"
read -r -a HUMANEVAL_BASE_BRANCH_COMPETITIONS_ARRAY <<< "${HUMANEVAL_BASE_BRANCH_COMPETITIONS}"
read -r -a HUMANEVAL_VERIFICATION_FORCE_BASE_WINNERS_ARRAY <<< "${HUMANEVAL_VERIFICATION_FORCE_BASE_WINNERS}"

HUMANEVAL_COUNT=${#HUMANEVAL_LENGTHS_ARRAY[@]}
if [[ ${HUMANEVAL_COUNT} -gt 0 ]]; then
  require_equal_length "${HUMANEVAL_COUNT}" \
    HUMANEVAL_NSHOTS_ARRAY HUMANEVAL_TEMPS_ARRAY HUMANEVAL_LIMITS_ARRAY \
    HUMANEVAL_DIFFUSION_STEPS_ARRAY HUMANEVAL_BLOCK_SIZES_ARRAY \
    HUMANEVAL_BLOCK_ADD_THRESHOLDS_ARRAY HUMANEVAL_DECODED_TOKEN_THRESHOLDS_ARRAY \
    HUMANEVAL_SKIP_THRESHOLDS_ARRAY HUMANEVAL_TOP_PS_ARRAY HUMANEVAL_DTYPES_ARRAY \
    HUMANEVAL_SAMPLING_STRATEGIES_ARRAY
  expand_array_to_length HUMANEVAL_MAX_BRANCHES_KEPTS_ARRAY "${HUMANEVAL_COUNT}"
  expand_array_to_length HUMANEVAL_BRANCHING_FACTORS_ARRAY "${HUMANEVAL_COUNT}"
  expand_array_to_length HUMANEVAL_BRANCH_TOPPS_ARRAY "${HUMANEVAL_COUNT}"
  expand_array_to_length HUMANEVAL_SELECTION_CONF_ALPHAS_ARRAY "${HUMANEVAL_COUNT}"
  expand_array_to_length HUMANEVAL_BRANCH_VERIFICATION_MODES_ARRAY "${HUMANEVAL_COUNT}"
  expand_array_to_length HUMANEVAL_BASE_BRANCH_COMPETITIONS_ARRAY "${HUMANEVAL_COUNT}"
  expand_array_to_length HUMANEVAL_VERIFICATION_FORCE_BASE_WINNERS_ARRAY "${HUMANEVAL_COUNT}"
fi

# --------------------------------------------------------------------------------------
# Execute sweeps
# --------------------------------------------------------------------------------------
run_idx=0
for lora_model in "${LORA_MODELS[@]}"; do
  lora_slug=$(slugify "$(basename "${lora_model}")")
  MODEL_LORA_PATH="${lora_model}"

  # Run HumanEval sweep first
  if [[ "${RUN_HUMANEVAL_SWEEP}" == "true" && ${HUMANEVAL_COUNT} -gt 0 ]]; then
    for idx in "${!HUMANEVAL_LENGTHS_ARRAY[@]}"; do
      task="humaneval"
      shots="${HUMANEVAL_NSHOTS_ARRAY[$idx]}"
      length="${HUMANEVAL_LENGTHS_ARRAY[$idx]}"
      temp="${HUMANEVAL_TEMPS_ARRAY[$idx]}"
      limit="${HUMANEVAL_LIMITS_ARRAY[$idx]}"
      diffusion_steps="${HUMANEVAL_DIFFUSION_STEPS_ARRAY[$idx]}"
      block_size="${HUMANEVAL_BLOCK_SIZES_ARRAY[$idx]}"
      block_add="${HUMANEVAL_BLOCK_ADD_THRESHOLDS_ARRAY[$idx]}"
      decoded_thresh="${HUMANEVAL_DECODED_TOKEN_THRESHOLDS_ARRAY[$idx]}"
      skip_thresh="${HUMANEVAL_SKIP_THRESHOLDS_ARRAY[$idx]}"
      top_p_value="${HUMANEVAL_TOP_PS_ARRAY[$idx]}"
      dtype_value="${HUMANEVAL_DTYPES_ARRAY[$idx]}"
      sampling_strategy="${HUMANEVAL_SAMPLING_STRATEGIES_ARRAY[$idx]}"
      max_branches="${HUMANEVAL_MAX_BRANCHES_KEPTS_ARRAY[$idx]}"
      branching_factor="${HUMANEVAL_BRANCHING_FACTORS_ARRAY[$idx]}"
      branch_topp="${HUMANEVAL_BRANCH_TOPPS_ARRAY[$idx]}"
      sel_conf_alpha="${HUMANEVAL_SELECTION_CONF_ALPHAS_ARRAY[$idx]}"
      branch_verify="${HUMANEVAL_BRANCH_VERIFICATION_MODES_ARRAY[$idx]}"
      base_compete="${HUMANEVAL_BASE_BRANCH_COMPETITIONS_ARRAY[$idx]}"
      force_base="${HUMANEVAL_VERIFICATION_FORCE_BASE_WINNERS_ARRAY[$idx]}"

      run_idx=$((run_idx+1))
      run_slug=$(slugify "humaneval-${lora_slug}-ns${shots}-len${length}-bf${branching_factor}-bs${block_size}-run${run_idx}")
      
      output_path="${OUTPUT_ROOT}/${run_slug}.jsonl"
      MODEL_SAVE_DIR="${RESULT_ROOT}/${run_slug}"
      mkdir -p "${MODEL_SAVE_DIR}"

      MODEL_MAX_NEW_TOKENS="${length}"
      MODEL_DIFFUSION_STEPS="${diffusion_steps}"
      MODEL_TEMPERATURE="${temp}"
      if [[ "${top_p_value}" == "none" || -z "${top_p_value}" ]]; then
        MODEL_TOP_P=""
      else
        MODEL_TOP_P="${top_p_value}"
      fi
      MODEL_BLOCK_SIZE="${block_size}"
      MODEL_BLOCK_ADD_THRESHOLD="${block_add}"
      MODEL_DECODED_TOKEN_THRESHOLD="${decoded_thresh}"
      MODEL_SKIP_THRESHOLD="${skip_thresh}"
      MODEL_DTYPE="${dtype_value}"
      MODEL_SAMPLING_STRATEGY="${sampling_strategy}"
      MODEL_MAX_BRANCHES_KEPT="${max_branches}"
      MODEL_BRANCHING_FACTOR="${branching_factor}"
      MODEL_BRANCH_TOPP="${branch_topp}"
      MODEL_SELECTION_CONF_ALPHA="${sel_conf_alpha}"
      MODEL_BRANCH_VERIFICATION_MODE="${branch_verify}"
      MODEL_BASE_BRANCH_COMPETITION="${base_compete}"
      MODEL_FORCE_BASE_BRANCH="${force_base}"
      MODEL_EXPERIMENT_LABEL="${run_slug}"

      MODEL_ARGS=$(build_model_args)
      build_cmd_and_run "${run_slug}" "${MODEL_ARGS}" "${task}" "${limit}" "${shots}" "${output_path}" "${APPLY_CHAT_TEMPLATE_HUMANEVAL}" "${FEWSHOT_AS_MULTITURN_HUMANEVAL}"
    done
  fi

  # Run other task sweeps after HumanEval
  if [[ "${RUN_TASK_SWEEP}" == "true" && ${TASK_COUNT} -gt 0 ]]; then
    for idx in "${!TASKS_ARRAY[@]}"; do
      task="${TASKS_ARRAY[$idx]}"
      shots="${NSHOTS_ARRAY[$idx]}"
      length="${LENGTH_ARRAY[$idx]}"
      temp="${TEMP_ARRAY[$idx]}"
      limit="${LIMITS_ARRAY[$idx]}"
      diffusion_steps="${DIFFUSION_STEPS_ARRAY[$idx]}"
      block_size="${BLOCK_SIZES_ARRAY[$idx]}"
      block_add="${BLOCK_ADD_THRESHOLDS_ARRAY[$idx]}"
      decoded_thresh="${DECODED_TOKEN_THRESHOLDS_ARRAY[$idx]}"
      skip_thresh="${SKIP_THRESHOLDS_ARRAY[$idx]}"
      top_p_value="${TOP_PS_ARRAY[$idx]}"
      dtype_value="${DTYPES_ARRAY[$idx]}"
      sampling_strategy="${SAMPLING_STRATEGIES_ARRAY[$idx]}"
      max_branches="${MAX_BRANCHES_KEPTS_ARRAY[$idx]}"
      branching_factor="${BRANCHING_FACTORS_ARRAY[$idx]}"
      branch_topp="${BRANCH_TOPPS_ARRAY[$idx]}"
      sel_conf_alpha="${SELECTION_CONF_ALPHAS_ARRAY[$idx]}"
      branch_verify="${BRANCH_VERIFICATION_MODES_ARRAY[$idx]}"
      base_compete="${BASE_BRANCH_COMPETITIONS_ARRAY[$idx]}"
      force_base="${VERIFICATION_FORCE_BASE_WINNERS_ARRAY[$idx]}"

      run_idx=$((run_idx+1))
      # DEBUG: Print progress to locate why the script might exit early
      echo "DEBUG: after increment, run_idx=${run_idx}, task=${task}, shots=${shots}, length=${length}"
      run_slug=$(slugify "${task}-${lora_slug}-ns${shots}-len${length}-bf${branching_factor}-bs${block_size}-temp${temp}-run${run_idx}")
      
      output_path="${OUTPUT_ROOT}/${run_slug}.jsonl"
      MODEL_SAVE_DIR="${RESULT_ROOT}/${run_slug}"
      mkdir -p "${MODEL_SAVE_DIR}"

      MODEL_MAX_NEW_TOKENS="${length}"
      MODEL_DIFFUSION_STEPS="${diffusion_steps}"
      MODEL_TEMPERATURE="${temp}"
      if [[ "${top_p_value}" == "none" || -z "${top_p_value}" ]]; then
        MODEL_TOP_P=""
      else
        MODEL_TOP_P="${top_p_value}"
      fi
      MODEL_BLOCK_SIZE="${block_size}"
      MODEL_BLOCK_ADD_THRESHOLD="${block_add}"
      MODEL_DECODED_TOKEN_THRESHOLD="${decoded_thresh}"
      MODEL_SKIP_THRESHOLD="${skip_thresh}"
      MODEL_DTYPE="${dtype_value}"
      MODEL_SAMPLING_STRATEGY="${sampling_strategy}"
      MODEL_MAX_BRANCHES_KEPT="${max_branches}"
      MODEL_BRANCHING_FACTOR="${branching_factor}"
      MODEL_BRANCH_TOPP="${branch_topp}"
      MODEL_SELECTION_CONF_ALPHA="${sel_conf_alpha}"
      MODEL_BRANCH_VERIFICATION_MODE="${branch_verify}"
      MODEL_BASE_BRANCH_COMPETITION="${base_compete}"
      MODEL_FORCE_BASE_BRANCH="${force_base}"
      MODEL_EXPERIMENT_LABEL="${run_slug}"

      MODEL_ARGS=$(build_model_args)
      build_cmd_and_run "${run_slug}" "${MODEL_ARGS}" "${task}" "${limit}" "${shots}" "${output_path}" "${APPLY_CHAT_TEMPLATE_TASKS}" "${FEWSHOT_AS_MULTITURN_TASKS}"
    done
  fi

done

echo "All sweep evaluations completed (${run_idx} runs)."
