#!/bin/bash
export PYTHONPATH=src:${PYTHONPATH}

DATA_PATH="data"
LM_PATH="output/lm_model/model.pt"
OUTPUT_PATH="output/RL/CLEVR_lm"
POLICY_PATH="output/SL_LSTM_32_64/model.pt"
POLICY_PATH_VQA="output/SL_LSTM_32_64_vqa/model.pt"
K_EPOCHS=10
MAX_LEN=20
UPDATE_EVERY=64
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


python src/scripts/run.py -env $ENV_ -max_len $MAX_LEN -data_path $DATA_PATH -out_path $OUTPUT_PATH -model $MODEL -update_every $UPDATE_EVERY -agent $AGENT -K_epochs $K_EPOCHS -eps_clip $EPS_CLIP -lr $LR -word_emb_size $WORD_EMB_SIZE -hidden_size $HIDDEN_SIZE -num_episodes_train $NUM_EPISODE_TRAIN -debug $DEBUG -lm_path $LM_PATH -reward $REWARD -reward_path $REWARD_PATH -condition_answer $CONDITION_ANSWER -num_episodes_test $NUM_EPISODE_TEST -reward_vocab $REWARD_VOCAB -mask_answers 1 -truncate_mode "proba_thr" -p_th 0.05 -grad_clip 1 -KL_coeff 0.05


