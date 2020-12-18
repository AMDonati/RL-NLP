#!/usr/bin/env bash

echo "---------------------------------------------------------------------------------------------------------------------------------"
echo "------------------------- VQAv2 -------------------------------------------------------------------------------------------------"
echo "---------------------------------------------------------------------------------------------------------------------------------"

DATA_PATH="data/vqa-v2/"
FEATURES_PATH="data/vqa-v2/coco_trainval.lmdb/"
LM_PATH="output/vqa_lm_model_smallvocab/model.pt"
OUTPUT_PATH="output/RL/VQAv2_2/"
POLICY_PATH="output/VQAv2_SL_32_64/model.pt"
K_EPOCHS=5
MAX_LEN=5
UPDATE_EVERY=32
NUM_EPISODE_TRAIN=5000
NUM_EPISODE_TEST=10
ENV_="vqa"
MODEL="lstm"
AGENT="PPO"
LR=0.001
WORD_EMB_SIZE=32
HIDDEN_SIZE=64
EPS_CLIP=0.01
REWARD="lv_norm"
FUSION="average"
CONDITION_ANSWER="none"
MIN_DATA=1
DEBUG="0,1"
NUM_QUESTIONS=1

if [ -n "$1" ]; then
  OUTPUT_PATH=$1
fi

echo "$(date +"%Y_%m_%d_%I_%M_%p")-------------------------  top p ---------------------------------------------------------------------------------------------------------------"
python src/scripts/run.py -env $ENV_ -max_len $MAX_LEN -data_path $DATA_PATH -out_path $OUTPUT_PATH -model $MODEL -update_every $UPDATE_EVERY -agent $AGENT -K_epochs $K_EPOCHS -eps_clip $EPS_CLIP -lr $LR -word_emb_size $WORD_EMB_SIZE -hidden_size $HIDDEN_SIZE -num_episodes_train $NUM_EPISODE_TRAIN -lm_path $LM_PATH -reward $REWARD -num_episodes_test $NUM_EPISODE_TEST -mask_answers 1 -grad_clip 1 -fusion $FUSION -condition_answer $CONDITION_ANSWER -features_path $FEATURES_PATH -debug $DEBUG -min_data $MIN_DATA -num_questions $NUM_QUESTIONS -truncate_mode "top_p" -top_p 0.96

DEBUG="0,2"
echo "$(date +"%Y_%m_%d_%I_%M_%p")-------------------------  top p 2 images ---------------------------------------------------------------------------------------------------------------"
python src/scripts/run.py -env $ENV_ -max_len $MAX_LEN -data_path $DATA_PATH -out_path $OUTPUT_PATH -model $MODEL -update_every $UPDATE_EVERY -agent $AGENT -K_epochs $K_EPOCHS -eps_clip $EPS_CLIP -lr $LR -word_emb_size $WORD_EMB_SIZE -hidden_size $HIDDEN_SIZE -num_episodes_train $NUM_EPISODE_TRAIN -lm_path $LM_PATH -reward $REWARD -num_episodes_test $NUM_EPISODE_TEST -mask_answers 1 -grad_clip 1 -fusion $FUSION -condition_answer $CONDITION_ANSWER -features_path $FEATURES_PATH -debug $DEBUG -min_data $MIN_DATA -num_questions $NUM_QUESTIONS -truncate_mode "top_p" -top_p 0.96

DEBUG="0,3"
echo "$(date +"%Y_%m_%d_%I_%M_%p")-------------------------  top p 2 images ---------------------------------------------------------------------------------------------------------------"
python src/scripts/run.py -env $ENV_ -max_len $MAX_LEN -data_path $DATA_PATH -out_path $OUTPUT_PATH -model $MODEL -update_every $UPDATE_EVERY -agent $AGENT -K_epochs $K_EPOCHS -eps_clip $EPS_CLIP -lr $LR -word_emb_size $WORD_EMB_SIZE -hidden_size $HIDDEN_SIZE -num_episodes_train $NUM_EPISODE_TRAIN -lm_path $LM_PATH -reward $REWARD -num_episodes_test $NUM_EPISODE_TEST -mask_answers 1 -grad_clip 1 -fusion $FUSION -condition_answer $CONDITION_ANSWER -features_path $FEATURES_PATH -debug $DEBUG -min_data $MIN_DATA -num_questions $NUM_QUESTIONS -truncate_mode "top_p" -top_p 0.96
