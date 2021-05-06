#!/bin/bash
#SBATCH --job-name=C-tr-pth1:V
#SBATCH --qos=qos_gpu-t3
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --array=1-5
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8g
#SBATCH --output=slurm_out/clevr/trunc-pth1:V-%j.out
#SBATCH --error=slurm_out/clevr/trunc-pth1:V-%j.err
#SBATCH --time=20:00:00

export TMPDIR=$JOBSCRATCH
module purge
module load  pytorch-gpu/py3/1.7.1
conda activate rl-nlp-2
export PYTHONPATH=src:${PYTHONPATH}

DATA_PATH="data"
LM_PATH="output/lm_model/model.pt"
OUTPUT_PATH="output/RL/CLEVR_lm/trunc_policy"
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
echo "now processing task id:: " ${SLURM_ARRAY_TASK_ID}
OUT_PATH=output/RL/CLEVR_lm/trunc_policy/${SLURM_ARRAY_TASK_ID}
srun python -u src/scripts/run.py -env $ENV_ -max_len $MAX_LEN -data_path $DATA_PATH -out_path ${OUT_PATH} -model $MODEL -update_every $UPDATE_EVERY -agent $AGENT -K_epochs $K_EPOCHS -eps_clip $EPS_CLIP -lr $LR -word_emb_size $WORD_EMB_SIZE -hidden_size $HIDDEN_SIZE -num_episodes_train $NUM_EPISODE_TRAIN -lm_path $LM_PATH -reward $REWARD -num_episodes_test $NUM_EPISODE_TEST -mask_answers 1 -grad_clip 1 -condition_answer $CONDITION_ANSWER -truncate_mode "proba_thr" -debug $DEBUG -reward_vocab $REWARD_VOCAB -reward_path $REWARD_PATH -truncation_optim 1

