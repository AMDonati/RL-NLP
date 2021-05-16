export PYTHONPATH=src:${PYTHONPATH}

DATA_PATH="data"
LM_PATH="output/lm_model/model.pt"
OUTPUT_PATH="output/RL/CLEVR_lm"
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
REWARD="bleu"
CONDITION_ANSWER="after_fusion"
DEBUG="0,20000"
REWARD_PATH="output/vqa_model_film/model.pt"
REWARD_VOCAB="data/closure_vocab.json"

OUT_PATH=output/RL/CLEVR_lm_
python src/scripts/run.py -env $ENV_ -max_len $MAX_LEN -data_path $DATA_PATH -out_path ${OUT_PATH} -model $MODEL -update_every $UPDATE_EVERY -agent $AGENT -K_epochs $K_EPOCHS -eps_clip $EPS_CLIP -lr $LR -word_emb_size $WORD_EMB_SIZE -hidden_size $HIDDEN_SIZE -num_episodes_train $NUM_EPISODE_TRAIN -lm_path $LM_PATH -reward $REWARD -num_episodes_test $NUM_EPISODE_TEST -mask_answers 1 -condition_answer $CONDITION_ANSWER -debug $DEBUG -reward_vocab $REWARD_VOCAB -reward_path $REWARD_PATH -policy_path $POLICY_PATH_VQA