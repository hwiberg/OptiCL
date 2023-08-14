#!/bin/bash
#SBATCH --array=100-299
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8000
#SBATCH -p sched_mit_sloan_batch
#SBATCH --time=0-48:00:00
#SBATCH -o ./sh/output.%A.%a.out # STDOUT

python3 run_single_nov22.py $SLURM_ARRAY_TASK_ID
