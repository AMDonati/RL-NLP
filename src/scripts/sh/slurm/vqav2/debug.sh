#!/bin/bash
#SBATCH --job-name=debug
#SBATCH --qos=qos_gpu-t4
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8g
#SBATCH --output=slurm_out/vqa/debug%j.out
#SBATCH --error=slurm_out/vqa/debug%j.err
#SBATCH --time=100:00:00

export TMPDIR=$JOBSCRATCH
module purge
module load  pytorch-gpu/py3/1.7.1
conda activate rl-nlp-2

export PYTHONPATH=src:${PYTHONPATH}


srun python -u src/data_provider/vqa_dataset.py
