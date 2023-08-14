#!/bin/bash
#SBATCH --array=301-600
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8000
#SBATCH -p sched_mit_sloan_batch
#SBATCH --time=0-12:00:00
#SBATCH -o ./sh/output.%A.%a.out # STDOUT

python3 run_single_gbm_aug2023.py $SLURM_ARRAY_TASK_ID
