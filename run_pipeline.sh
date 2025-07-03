#!/bin/bash

# Set global parameters
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SEED=42
LOG_DIR="logs_${TIMESTAMP}"
mkdir -p ${LOG_DIR}

# Enhanced model parameters
LATENT_DIM=768
TRANSFORMER_LAYERS=3
VICREG_WEIGHTS="--sim_weight 30.0 --var_weight 30.0 --cov_weight 0.5"
TEMPORAL_CONSISTENCY="--temp_weight 0.5"
TRAINING_SETUP="--warmup_epochs 15 --swa_start 80"

# Robust Augmentations
AUGMENTATIONS="--time_warp_limit 0.3 --channel_drop_prob 0.4 --acc_noise_std 0.15 --gyro_noise_std 0.2 --scale_min 0.7 --scale_max 1.3"

# Advanced Alignment
ALIGNMENT_PARAMS="--feature_weight 0.85"

# PAMAP2 Experiments
echo -e "\n=== Running PAMAP2 Experiments ===\n"

# Pretraining
LOG_FILE="${LOG_DIR}/pamap2_pretrain.log"
echo "[RUNNING] PAMAP2 Pretraining" | tee ${LOG_FILE}
python src/train.py --mode pretrain --dataset PAMAP2 \
    --latent_dim ${LATENT_DIM} --num_transformer_layers ${TRANSFORMER_LAYERS} \
    --epochs 120 --lr 0.0002 --window_size 192 --window_step 64 \
    ${VICREG_WEIGHTS} ${TEMPORAL_CONSISTENCY} ${AUGMENTATIONS} ${TRAINING_SETUP} \
    2>&1 | tee -a ${LOG_FILE}
if [ $? -ne 0 ]; then
    echo "❌ ERROR: Pretraining failed"
fi

# Linear Probing
LOG_FILE="${LOG_DIR}/pamap2_linear_probe.log"
echo -e "\n[RUNNING] PAMAP2 Linear Probing" | tee ${LOG_FILE}
python src/train.py --mode linear_probe --dataset PAMAP2 \
    --latent_dim ${LATENT_DIM} --num_transformer_layers ${TRANSFORMER_LAYERS} \
    --window_size 192 --window_step 64 \
    2>&1 | tee -a ${LOG_FILE}
if [ $? -ne 0 ]; then
    echo "❌ ERROR: Linear probing failed"
fi

# Alignment
LOG_FILE="${LOG_DIR}/pamap2_alignment.log"
echo -e "\n[RUNNING] PAMAP2 Alignment" | tee ${LOG_FILE}
python src/train.py --mode alignment --dataset PAMAP2 \
    --latent_dim ${LATENT_DIM} --num_transformer_layers ${TRANSFORMER_LAYERS} \
    --window_size 192 --window_step 64 \
    ${ALIGNMENT_PARAMS} ${TEMPORAL_CONSISTENCY} \
    2>&1 | tee -a ${LOG_FILE}
if [ $? -ne 0 ]; then
    echo "❌ ERROR: Alignment failed"
fi

# UCI HAR Experiments
echo -e "\n=== Running UCI HAR Experiments ===\n"

# Pretraining
LOG_FILE="${LOG_DIR}/uci_pretrain.log"
echo "[RUNNING] UCI HAR Pretraining" | tee ${LOG_FILE}
python src/train.py --mode pretrain --dataset UCI \
    --latent_dim ${LATENT_DIM} --num_transformer_layers ${TRANSFORMER_LAYERS} \
    --epochs 120 --lr 0.0002 --window_size 128 --window_step 32 \
    ${VICREG_WEIGHTS} ${TEMPORAL_CONSISTENCY} ${AUGMENTATIONS} ${TRAINING_SETUP} \
    2>&1 | tee -a ${LOG_FILE}
if [ $? -ne 0 ]; then
    echo "❌ ERROR: Pretraining failed"
fi

# Linear Probing
LOG_FILE="${LOG_DIR}/uci_linear_probe.log"
echo -e "\n[RUNNING] UCI HAR Linear Probing" | tee ${LOG_FILE}
python src/train.py --mode linear_probe --dataset UCI \
    --latent_dim ${LATENT_DIM} --num_transformer_layers ${TRANSFORMER_LAYERS} \
    --window_size 128 --window_step 32 \
    2>&1 | tee -a ${LOG_FILE}
if [ $? -ne 0 ]; then
    echo "❌ ERROR: Linear probing failed"
fi

# Alignment
LOG_FILE="${LOG_DIR}/uci_alignment.log"
echo -e "\n[RUNNING] UCI HAR Alignment" | tee ${LOG_FILE}
python src/train.py --mode alignment --dataset UCI \
    --latent_dim ${LATENT_DIM} --num_transformer_layers ${TRANSFORMER_LAYERS} \
    --window_size 128 --window_step 32 \
    ${ALIGNMENT_PARAMS} ${TEMPORAL_CONSISTENCY} \
    2>&1 | tee -a ${LOG_FILE}
if [ $? -ne 0 ]; then
    echo "❌ ERROR: Alignment failed"
fi

echo -e "\n=== Pipeline Complete ==="
