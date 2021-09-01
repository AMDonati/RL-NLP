#!/bin/bash
#SBATCH --job-name=gpt_with_grads
#SBATCH --qos=qos_gpu-t4
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8g
#SBATCH --output=slurm_out/vqa_lr/gptmodel-%j.out
#SBATCH --error=slurm_out/vqa_lr/gptmodel-%j.err
#SBATCH --time=100:00:00
#SBATCH -A ktz@gpu

export TMPDIR=$JOBSCRATCH
module purge
module load  pytorch-gpu/py3/1.7.1
conda activate rl-nlp-2

export PYTHONPATH=src:${PYTHONPATH}


DATA_PATH="data/vqa-v2/"
FEATURES_PATH="data/vqa-v2/coco_trainval.lmdb/"
LM_PATH="output/vqa_lm_model/model.pt"
LM_PATH_MIN="output/vqa_lm_model_smallvocab/model.pt"
OUTPUT_PATH="output/RL/VQAv2"
POLICY_PATH="output/vqa_policy_128_256_answer/model.pt"
POLICY_PATH_MIN="output/vqa_policy_128_256_answer_smallvocab/model.pt"
VILBERT_VOCAB="output/vilbert_vqav2/bert_base_6layer_6conect.json"
VILBERT_PATH="output/vilbert_vqav2/model.bin"
K_EPOCHS=2
MAX_LEN=10
UPDATE_EVERY=16
DEBUG="0,20000"
NUM_EPISODE_TRAIN=100000
NUM_EPISODE_TEST=20000
ENV_="vqa"
MODEL="gpt"
AGENT="PPO"
LR=0.00005
WORD_EMB_SIZE=128
HIDDEN_SIZE=256
EPS_CLIP=0.01
REWARD="vilbert_rank2"
FUSION="average"
CONDITION_ANSWER="after_fusion"


OUT_PATH=output/RL/VQAv2_full/gptmodel/
srun python -u src/scripts/run.py -env $ENV_ -max_len $MAX_LEN -data_path $DATA_PATH -out_path ${OUT_PATH} -model $MODEL -update_every $UPDATE_EVERY -agent $AGENT -K_epochs $K_EPOCHS -eps_clip $EPS_CLIP -lr $LR -word_emb_size $WORD_EMB_SIZE -hidden_size $HIDDEN_SIZE -num_episodes_train $NUM_EPISODE_TRAIN -lm_path $LM_PATH -reward $REWARD -num_episodes_test $NUM_EPISODE_TEST -mask_answers 1 -fusion $FUSION -condition_answer $CONDITION_ANSWER -features_path $FEATURES_PATH -reward_vocab $VILBERT_VOCAB -reward_path $VILBERT_PATH  -grad_clip None
