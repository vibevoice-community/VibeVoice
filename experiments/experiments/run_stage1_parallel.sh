#!/bin/bash
# Parallel Bayesian Optimization Runner
# Stage: stage1
# Agent: bayesopt-agent

set -e

echo "Running 5 parallel experiments for stage1"
echo "================================================================"

# Create logs directory
mkdir -p experiments/logs

# Run experiments in background
PIDS=()


echo "Starting experiment 1/5..."
bash experiments/scripts/train_stage1_config1.sh > experiments/logs/stage1_config1.log 2>&1 &
PIDS+=($!)
sleep 5  # Stagger starts to avoid resource contention

echo "Starting experiment 2/5..."
bash experiments/scripts/train_stage1_config2.sh > experiments/logs/stage1_config2.log 2>&1 &
PIDS+=($!)
sleep 5  # Stagger starts to avoid resource contention

echo "Starting experiment 3/5..."
bash experiments/scripts/train_stage1_config3.sh > experiments/logs/stage1_config3.log 2>&1 &
PIDS+=($!)
sleep 5  # Stagger starts to avoid resource contention

echo "Starting experiment 4/5..."
bash experiments/scripts/train_stage1_config4.sh > experiments/logs/stage1_config4.log 2>&1 &
PIDS+=($!)
sleep 5  # Stagger starts to avoid resource contention

echo "Starting experiment 5/5..."
bash experiments/scripts/train_stage1_config5.sh > experiments/logs/stage1_config5.log 2>&1 &
PIDS+=($!)
sleep 5  # Stagger starts to avoid resource contention

echo "================================================================"
echo "All experiments started. Waiting for completion..."
echo "Monitor logs in: experiments/logs/"
echo "================================================================"

# Wait for all to complete
FAILED=0
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    if wait $PID; then
        echo "Experiment $((i+1)) completed successfully"
    else
        echo "Experiment $((i+1)) failed"
        FAILED=$((FAILED+1))
    fi
done

echo "================================================================"
echo "All experiments completed"
echo "Successful: $((${#PIDS[@]} - FAILED))/${#PIDS[@]}"
echo "Failed: $FAILED/${#PIDS[@]}"
echo "================================================================"

# Aggregate results
python experiments/analyze_results.py
