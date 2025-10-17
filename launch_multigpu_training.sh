#!/bin/bash

# Multi-GPU Training Launcher Script for D3 Detection

# Check if number of GPUs is provided
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <num_gpus> [config_file]"
    echo "Example: $0 4 configs/D3_multigpu.yaml"
    exit 1
fi

NUM_GPUS=$1
CONFIG_FILE=${2:-"configs/D3_multigpu.yaml"}

echo "Starting multi-GPU training with $NUM_GPUS GPUs"
echo "Using config file: $CONFIG_FILE"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file $CONFIG_FILE not found!"
    exit 1
fi

# Check available GPUs
AVAILABLE_GPUS=$(nvidia-smi -L | wc -l)
echo "Available GPUs: $AVAILABLE_GPUS"

if [ "$NUM_GPUS" -gt "$AVAILABLE_GPUS" ]; then
    echo "Error: Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS available!"
    exit 1
fi

# Set environment variables for better performance
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=1

# Launch training
python train_multigpu.py \
    --config="$CONFIG_FILE" \
    --world-size="$NUM_GPUS"

echo "Training completed!"