#!/bin/bash
#SBATCH -J vi
#SBATCH -t 00:15:00
#SBATCH -p gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=
#SBATCH --array=0-4
#SBATCH -o test_logs/log_%a.out
#SBATCH -e test_logs/err_%a.err

source .venv/bin/activate
DATASET_PATH=" "

TOTAL=500
CHUNK=100
START=$((SLURM_ARRAY_TASK_ID * CHUNK))
END=$((START + CHUNK))

if [ "$END" -gt "$TOTAL" ]; then
    END=$TOTAL
fi


OUTPUT_DIR="test_results/"

mkdir -p test_logs
mkdir -p "$OUTPUT_DIR"

python testing.py --output_dir "$OUTPUT_DIR" --start "$START" --end "$END" --job_id "$SLURM_ARRAY_TASK_ID" --dtst_path "$DATASET_PATH"
