#!/bin/bash
#SBATCH --job-name=metrics
#SBATCH --qos=qos_gpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8g
#SBATCH --output=slurm_out/clevr/metrics-%j.out
#SBATCH --error=slurm_out/clevr/metrics-%j.err
#SBATCH --time=20:00:00

export TMPDIR=$JOBSCRATCH
module purge
module load  pytorch-gpu/py3/1.7.1
conda activate rl-nlp-2

export PYTHONPATH=src:${PYTHONPATH}

PATH="output/RL/CLEVR_lm/FINAL_EXP"

set -x
srun python -u merge_metrics.py -path $PATH
#wait
