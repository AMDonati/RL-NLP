#!/usr/bin/env bash
#run this file:
#src/scripts/sh/debug/debug_script_vqav2.sh   > output/RL/test_before_merge_$(date +"%Y_%m_%d_%I_%M_%p").txt 2>&1

echo "---------------------------------------------------------------------------------------------------------------------------------"
echo "------------------------- CLEVR -------------------------------------------------------------------------------------------------"
echo "---------------------------------------------------------------------------------------------------------------------------------"

DATA_PATH="data"
LM_PATH="output/lm_model/model.pt"
OUTPUT_PATH="output/RL/debug"
POLICY_PATH="output/SL_LSTM_32_64/model.pt"
POLICY_PATH_VQA="output/SL_LSTM_32_64_vqa/model.pt"
K_EPOCHS=20
MAX_LEN=5
UPDATE_EVERY=128
ENV_="clevr"
MODEL="lstm"
AGENT="PPO"
LR=0.001
WORD_EMB_SIZE=32
HIDDEN_SIZE=64
NUM_EPISODE_TRAIN=10000
NUM_EPISODE_TEST=5
EPS_CLIP=0.01
REWARD="lv_norm"
FUSION="cat"
CONDITION_ANSWER="after_fusion"
DEBUG="0,50000"
REWARD_PATH="output/vqa_model_film/model.pt"
REWARD_VOCAB="data/closure_vocab.json"

echo "$(date +"%Y_%m_%d_%I_%M_%S_%p")------------------------- pretrain ----------------------------------------------------------------------------------------------------"
python src/scripts/run.py -env $ENV_ -max_len $MAX_LEN -data_path $DATA_PATH -out_path $OUTPUT_PATH -model $MODEL -update_every $UPDATE_EVERY -agent $AGENT -K_epochs $K_EPOCHS -eps_clip $EPS_CLIP -lr $LR -word_emb_size $WORD_EMB_SIZE -hidden_size $HIDDEN_SIZE -num_episodes_train $NUM_EPISODE_TRAIN -lm_path $LM_PATH -reward $REWARD -num_episodes_test $NUM_EPISODE_TEST -mask_answers 1 -grad_clip 1 -fusion $FUSION -condition_answer $CONDITION_ANSWER  -policy_path $POLICY_PATH_VQA -debug $DEBUG
