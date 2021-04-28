#!/bin/bash
#SBATCH --job-name=C-T-pretrain
#SBATCH --qos=qos_gpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8g
#SBATCH --output=slurm_out/clevr/T-pretrain-%j.out
#SBATCH --error=slurm_out/clevr/T-pretrain-%j.err
#SBATCH --time=20:00:00


export TMPDIR=$JOBSCRATCH
module purge
module load  pytorch-gpu/py3/1.7.1
conda activate rl-nlp-2
export PYTHONPATH=src:${PYTHONPATH}


MODEL_PATH="output/RL/CLEVR_lm/FINAL_EXP_test/pretrain"
NUM_EPISODES_TEST=5000

set -x
srun python -u src/scripts/test.py -models_path $MODEL_PATH -num_episodes_test $NUM_EPISODES_TEST -test_metrics "ppl_dialog_lm" "language_score"