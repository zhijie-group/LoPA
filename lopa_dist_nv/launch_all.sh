#!/bin/bash

# ====================================================
# 1. Base environment configuration (unchanged)
# ====================================================

# If your machine is offline, you can set the following environment variables to true
# export HF_HUB_OFFLINE=1
# export HF_DATASETS_OFFLINE=1
# export HF_EVALUATE_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1
# export WANDB_DISABLED=true

export HF_HOME="<YOUR_HF_HOME>"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_METRICS_CACHE="$HF_HOME/metrics"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export RUN_HUMANEVAL_SWEEP=false
export PROFILE=false
export ENABLE_RUN_LOG=$PROFILE
export ENABLE_SAMPLE_LOG=$PROFILE
export ENABLE_PROFILING=$PROFILE

# Prevent NCCL deadlocks
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Ensure log directory exists
mkdir -p log

# ====================================================
# 2. Experiment loop
# ====================================================

# Define mode order (Instruct, then Base)
MODES=(
    "instruct" 
    "base"
)

# Define branch sizes to run: 4 then 8
BRANCH_SIZES=(4 8)

for MODE in "${MODES[@]}"; do
    # --- Set model paths for the current mode ---
    export RUN_MODE="$MODE"
    
    if [ "$RUN_MODE" = "instruct" ]; then
        echo ">>> Setting up for INSTRUCT mode..."
        export MODEL_PRETRAINED="Dream-org/Dream-v0-Instruct-7B"
        export MODEL_LORA_PATH_DEFAULT="SJTU-Deng-Lab/D2F_Dream_v0_Instruct_LoRA"
    elif [ "$RUN_MODE" = "base" ]; then
        echo ">>> Setting up for BASE mode..."
        export MODEL_PRETRAINED="Dream-org/Dream-v0-Base-7B"
        export MODEL_LORA_PATH_DEFAULT="SJTU-Deng-Lab/D2F_Dream_v0_Base_Lora"
    else
        echo "Error: Unknown mode $RUN_MODE"
        exit 1
    fi

    # --- Iterate over branch sizes ---
    for BRANCH in "${BRANCH_SIZES[@]}"; do
        export NUM_BRANCHES=$BRANCH
        
        # Define log name (keep original format)
        LOG_NAME="lopa_profile_${PROFILE}_bp${NUM_BRANCHES}_${RUN_MODE}.log"

        echo "========================================================================"
        echo "STARTING EXPERIMENT: Mode=${RUN_MODE} | Branch=${NUM_BRANCHES}"
        echo "Model Path: ${MODEL_PRETRAINED}"
        echo "Log File:   log/${LOG_NAME}"
        echo "========================================================================"

        # --- Run experiment script ---
        # Use 2>&1 | tee to output to both screen and file
        test_lopa_dist_nv_dream.sh 2>&1 | tee "log/${LOG_NAME}"
        
        # Check exit status of the previous command (optional)
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            echo "!!! WARNING: Experiment ${RUN_MODE} (BP=${NUM_BRANCHES}) failed or was interrupted."
            # Uncomment below to stop the entire script on failure
            # exit 1 
        fi
        
        echo "------------------------------------------------------------------------"
        echo "FINISHED: Mode=${RUN_MODE} | Branch=${NUM_BRANCHES}"
        echo "------------------------------------------------------------------------"
        echo ""
        
        # Optional: sleep a few seconds after each run to free GPU memory
        sleep 5
    done
done

echo "All experiments completed."