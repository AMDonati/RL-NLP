#!/bin/bash
#SBATCH --job-name=clevr-pth0.1
#SBATCH --qos=qos_gpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8g
#SBATCH --output=slurm_out/clevr/pth0.1-%j.out
#SBATCH --error=slurm_out/clevr/pth0.1-%j.err
#SBATCH --time=20:00:00

export TMPDIR=$JOBSCRATCH
module purge
module load  pytorch-gpu/py3/1.7.1
conda activate rl-nlp-2

export PYTHONPATH=src:${PYTHONPATH}

DATA_PATH="data"
LM_PATH="output/lm_ext/model.pt"
OUTPUT_PATH="output/RL/CLEVR_lm-ext"
POLICY_PATH="output/SL_LSTM_32_64/model.pt"
POLICY_PATH_VQA="output/SL_LSTM_32_64_vqa/model.pt"
K_EPOCHS=20
MAX_LEN=20
UPDATE_EVERY=128
NUM_EPISODE_TRAIN=50000
ENV_="clevr"
MODEL="lstm"
AGENT="PPO"
LR=0.001
WORD_EMB_SIZE=32
HIDDEN_SIZE=64
NUM_EPISODE_TEST=5000
EPS_CLIP=0.02
REWARD="vqa"
CONDITION_ANSWER="after_fusion"
DEBUG="0,20000"
REWARD_PATH="output/vqa_model_film/model.pt"
REWARD_VOCAB="data/closure_vocab.json"

set -x
srun python -u src/scripts/test.py -models_path "output/RL/CLEVR_lm-ext/pth_0.1" -eval_modes "sampling" "greedy" "sampling_ranking_lm"
