#!/bin/bash
#SBATCH --job-name=C-T-G-pth0.1
#SBATCH --qos=qos_gpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8g
#SBATCH --output=slurm_out/clevr/T-E-pth0.1-%j.out
#SBATCH --error=slurm_out/clevr/T-E-pth0.1-%j.err
#SBATCH --time=20:00:00


export TMPDIR=$JOBSCRATCH
module purge
module load  pytorch-gpu/py3/1.7.1
conda activate rl-nlp-2
export PYTHONPATH=src:${PYTHONPATH}


MODEL_PATH="output/RL/CLEVR_lm-ext/trunc_policy/1/clevr_vqa_PPO_answ-img_sampling_proba_thr0.1_adam_0.001_ent0.01_epsclip0.02_graclip1.0"
NUM_EPISODES_TEST=5000

set -x
srun python -u src/scripts/test.py -models_path $MODEL_PATH -num_episodes_test $NUM_EPISODES_TEST -eval_modes sampling_ranking_lm