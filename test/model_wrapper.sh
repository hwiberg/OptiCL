#!/bin/bash
#SBATCH --array=0-5
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8000
#SBATCH -p sched_mit_sloan_batch
#SBATCH --time=0-06:00:00
#

python3 run_single.py $SLURM_ARRAY_TASK_ID