#!/bin/bash
#SBATCH --job-name=merge-stats
#SBATCH --qos=qos_gpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8g
#SBATCH --output=slurm_out/clevr/merge_states-%j.out
#SBATCH --error=slurm_out/clevr/merge_stats-%j.err
#SBATCH --time=20:00:00

export TMPDIR=$JOBSCRATCH
module purge
module load  pytorch-gpu/py3/1.7.1
conda activate rl-nlp-2

export PYTHONPATH=src:${PYTHONPATH}

OUT_PATH="output/RL/CLEVR_lm-ext/FINAL_EXP"

set -x
srun python -u merge_stats.py -path $OUT_PATH
#wait
