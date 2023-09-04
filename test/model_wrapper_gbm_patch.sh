#!/bin/bash
#SBATCH --array=0-299
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8000
#SBATCH -p sched_mit_sloan_batch
#SBATCH --time=05:00:00
#SBATCH -o ./sh/output.%A.%a.out # STDOUT

python3 run_time_limit_sep2023.py $SLURM_ARRAY_TASK_ID
