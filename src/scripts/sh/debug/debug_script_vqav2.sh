#!/usr/bin/env bash
#run this file:
#src/scripts/sh/debug/debug_script_vqav2.sh   > output/RL/test_before_merge_$(date +"%Y_%m_%d_%I_%M_%p").txt 2>&1
# VQA REWARD:

echo "---------------------------------------------------------------------------------------------------------------------------------"
echo "------------------------- VQAv2 -------------------------------------------------------------------------------------------------"
echo "---------------------------------------------------------------------------------------------------------------------------------"

DATA_PATH="data/vqa-v2/"
FEATURES_PATH="data/vqa_v2/coco_trainval.lmdb/"
LM_PATH="output/vqa_lm_model/model.pt"
OUTPUT_PATH="output/RL/debug"
#POLICY_PATH="output/SL_LSTM_32_64_vqa/model.pt"
K_EPOCHS=10
MAX_LEN=5
UPDATE_EVERY=7
NUM_EPISODE_TRAIN=10
ENV_="vqa"
MODEL="lstm"
AGENT="PPO"
LR=0.001
WORD_EMB_SIZE=32
HIDDEN_SIZE=64
NUM_EPISODE_TRAIN=10
NUM_EPISODE_TEST=5
EPS_CLIP=0.01
REWARD="lv_norm"
FUSION="average"
CONDITION_ANSWER="after_fusion"


echo "-------------------------- Scratch -------------------$(date +"%Y_%m_%d_%I_%M_%p")--------------------------------------------------------------------------------"
python src/scripts/run.py -env $ENV_ -max_len $MAX_LEN -data_path $DATA_PATH -out_path $OUTPUT_PATH -model $MODEL -update_every $UPDATE_EVERY -agent $AGENT -K_epochs $K_EPOCHS -eps_clip $EPS_CLIP -lr $LR -word_emb_size $WORD_EMB_SIZE -hidden_size $HIDDEN_SIZE -num_episodes_train $NUM_EPISODE_TRAIN -lm_path $LM_PATH -reward $REWARD -num_episodes_test $NUM_EPISODE_TEST -mask_answers 1 -grad_clip 1 -fusion $FUSION -condition_answer $CONDITION_ANSWER -features_path $FEATURES_PATH
echo "------------------------- pretrain -------------------$(date +"%Y_%m_%d_%I_%M_%p")---------------------------------------------------------------------------------"
#python src/scripts/run.py -env $ENV_ -max_len $MAX_LEN -data_path $DATA_PATH -out_path $OUTPUT_PATH -model $MODEL -update_every $UPDATE_EVERY -agent $AGENT -K_epochs $K_EPOCHS -eps_clip $EPS_CLIP -lr $LR -word_emb_size $WORD_EMB_SIZE -hidden_size $HIDDEN_SIZE -num_episodes_train $NUM_EPISODE_TRAIN -lm_path $LM_PATH -reward $REWARD -num_episodes_test $NUM_EPISODE_TEST -mask_answers 1 -grad_clip 1 -fusion $FUSION -condition_answer $CONDITION_ANSWER -features_path $FEATURES_PATH -policy_path $POLICY_PATH
echo "------------------------- top k ----------------------$(date +"%Y_%m_%d_%I_%M_%p")------------------------------------------------------------------------------"
python src/scripts/run.py -env $ENV_ -max_len $MAX_LEN -data_path $DATA_PATH -out_path $OUTPUT_PATH -model $MODEL -update_every $UPDATE_EVERY -agent $AGENT -K_epochs $K_EPOCHS -eps_clip $EPS_CLIP -lr $LR -word_emb_size $WORD_EMB_SIZE -hidden_size $HIDDEN_SIZE -num_episodes_train $NUM_EPISODE_TRAIN -lm_path $LM_PATH -reward $REWARD -num_episodes_test $NUM_EPISODE_TEST -mask_answers 1 -grad_clip 1 -fusion $FUSION -condition_answer $CONDITION_ANSWER -features_path $FEATURES_PATH -truncate_mode "top_k" -num_truncated 20
echo "------------------------- sample_va ------------------$(date +"%Y_%m_%d_%I_%M_%p")----------------------------------------------------------------------------------"
python src/scripts/run.py -env $ENV_ -max_len $MAX_LEN -data_path $DATA_PATH -out_path $OUTPUT_PATH -model $MODEL -update_every $UPDATE_EVERY -agent $AGENT -K_epochs $K_EPOCHS -eps_clip $EPS_CLIP -lr $LR -word_emb_size $WORD_EMB_SIZE -hidden_size $HIDDEN_SIZE -num_episodes_train $NUM_EPISODE_TRAIN -lm_path $LM_PATH -reward $REWARD -num_episodes_test $NUM_EPISODE_TEST -mask_answers 1 -grad_clip 1 -fusion $FUSION -condition_answer $CONDITION_ANSWER -features_path $FEATURES_PATH -truncate_mode "sample_va"
echo "------------------------- proba threshold ----------------------------------------------------------------------------------------------------"
python src/scripts/run.py -env $ENV_ -max_len $MAX_LEN -data_path $DATA_PATH -out_path $OUTPUT_PATH -model $MODEL -update_every $UPDATE_EVERY -agent $AGENT -K_epochs $K_EPOCHS -eps_clip $EPS_CLIP -lr $LR -word_emb_size $WORD_EMB_SIZE -hidden_size $HIDDEN_SIZE -num_episodes_train $NUM_EPISODE_TRAIN -lm_path $LM_PATH -reward $REWARD -num_episodes_test $NUM_EPISODE_TEST -mask_answers 1 -grad_clip 1 -fusion $FUSION -condition_answer $CONDITION_ANSWER -features_path $FEATURES_PATH -truncate_mode "proba_thr"
echo "-------------------------  top p ---------------------$(date +"%Y_%m_%d_%I_%M_%p")------------------------------------------------------------------------------------------"
python src/scripts/run.py -env $ENV_ -max_len $MAX_LEN -data_path $DATA_PATH -out_path $OUTPUT_PATH -model $MODEL -update_every $UPDATE_EVERY -agent $AGENT -K_epochs $K_EPOCHS -eps_clip $EPS_CLIP -lr $LR -word_emb_size $WORD_EMB_SIZE -hidden_size $HIDDEN_SIZE -num_episodes_train $NUM_EPISODE_TRAIN -lm_path $LM_PATH -reward $REWARD -num_episodes_test $NUM_EPISODE_TEST -mask_answers 1 -grad_clip 1 -fusion $FUSION -condition_answer $CONDITION_ANSWER -features_path $FEATURES_PATH -truncate_mode "top_p" -top_p 0.8
echo "------------------------- sample_va + epsilon truncation --------$(date +"%Y_%m_%d_%I_%M_%p")--------------------------------------------------------------------------------------------"
python src/scripts/run.py -env $ENV_ -max_len $MAX_LEN -data_path $DATA_PATH -out_path $OUTPUT_PATH -model $MODEL -update_every $UPDATE_EVERY -agent $AGENT -K_epochs $K_EPOCHS -eps_clip $EPS_CLIP -lr $LR -word_emb_size $WORD_EMB_SIZE -hidden_size $HIDDEN_SIZE -num_episodes_train $NUM_EPISODE_TRAIN -lm_path $LM_PATH -reward $REWARD -num_episodes_test $NUM_EPISODE_TEST -mask_answers 1 -grad_clip 1 -fusion $FUSION -condition_answer $CONDITION_ANSWER -features_path $FEATURES_PATH -truncate_mode "sample_va" -epsilon_truncated 0.2
echo "------------------------- sample_va + GPT2 ----------------------$(date +"%Y_%m_%d_%I_%M_%p")------------------------------------------------------------------------------"
python src/scripts/run.py -env $ENV_ -max_len $MAX_LEN -data_path $DATA_PATH -out_path $OUTPUT_PATH -model $MODEL -update_every $UPDATE_EVERY -agent $AGENT -K_epochs $K_EPOCHS -eps_clip $EPS_CLIP -lr $LR -word_emb_size $WORD_EMB_SIZE -hidden_size $HIDDEN_SIZE -num_episodes_train $NUM_EPISODE_TRAIN -lm_path $LM_PATH -reward $REWARD -num_episodes_test $NUM_EPISODE_TEST -mask_answers 1 -grad_clip 1 -fusion $FUSION -condition_answer $CONDITION_ANSWER -features_path $FEATURES_PATH -truncate_mode "sample_va"

echo "---------------------------------------------------------------------------------------------------------------------------------"
echo "------------------------- CLEVR -------------------------------------------------------------------------------------------------"
echo "---------------------------------------------------------------------------------------------------------------------------------"

DATA_PATH="data"
LM_PATH="output/lm_model/model.pt"
OUTPUT_PATH="output/RL/debug"
#POLICY_PATH="output/SL_LSTM_32_64_vqa/model.pt"
K_EPOCHS=10
MAX_LEN=5
UPDATE_EVERY=7
NUM_EPISODE_TRAIN=10
ENV_="clevr"
MODEL="lstm"
AGENT="PPO"
LR=0.001
WORD_EMB_SIZE=32
HIDDEN_SIZE=64
NUM_EPISODE_TRAIN=10
NUM_EPISODE_TEST=5
EPS_CLIP=0.01
REWARD="bleu"
FUSION="cat"
CONDITION_ANSWER="after_fusion"
DEBUG="0,10"
REWARD_PATH="output/vqa_model_film/model.pt"
REWARD_VOCAB="data/closure_vocab.json"

echo "$(date +"%Y_%m_%d_%I_%M_%S_%p")-------------------------- Scratch ---------------------------------------------------------------------------------------------------"
python src/scripts/run.py -env $ENV_ -max_len $MAX_LEN -data_path $DATA_PATH -out_path $OUTPUT_PATH -model $MODEL -update_every $UPDATE_EVERY -agent $AGENT -K_epochs $K_EPOCHS -eps_clip $EPS_CLIP -lr $LR -word_emb_size $WORD_EMB_SIZE -hidden_size $HIDDEN_SIZE -num_episodes_train $NUM_EPISODE_TRAIN -lm_path $LM_PATH -reward $REWARD -num_episodes_test $NUM_EPISODE_TEST -mask_answers 1 -grad_clip 1 -fusion $FUSION -condition_answer $CONDITION_ANSWER -debug $DEBUG
echo "$(date +"%Y_%m_%d_%I_%M_%S_%p")------------------------- pretrain ----------------------------------------------------------------------------------------------------"
#python src/scripts/run.py -env $ENV_ -max_len $MAX_LEN -data_path $DATA_PATH -out_path $OUTPUT_PATH -model $MODEL -update_every $UPDATE_EVERY -agent $AGENT -K_epochs $K_EPOCHS -eps_clip $EPS_CLIP -lr $LR -word_emb_size $WORD_EMB_SIZE -hidden_size $HIDDEN_SIZE -num_episodes_train $NUM_EPISODE_TRAIN -lm_path $LM_PATH -reward $REWARD -num_episodes_test $NUM_EPISODE_TEST -mask_answers 1 -grad_clip 1 -fusion $FUSION -condition_answer $CONDITION_ANSWER  -policy_path $POLICY_PATH -debug $DEBUG
echo "$(date +"%Y_%m_%d_%I_%M_%S_%p")------------------------- top k ----------------------------------------------------------------------------------------------------"
python src/scripts/run.py -env $ENV_ -max_len $MAX_LEN -data_path $DATA_PATH -out_path $OUTPUT_PATH -model $MODEL -update_every $UPDATE_EVERY -agent $AGENT -K_epochs $K_EPOCHS -eps_clip $EPS_CLIP -lr $LR -word_emb_size $WORD_EMB_SIZE -hidden_size $HIDDEN_SIZE -num_episodes_train $NUM_EPISODE_TRAIN -lm_path $LM_PATH -reward $REWARD -num_episodes_test $NUM_EPISODE_TEST -mask_answers 1 -grad_clip 1 -fusion $FUSION -condition_answer $CONDITION_ANSWER -truncate_mode "top_k" -num_truncated 20 -debug $DEBUG
echo "$(date +"%Y_%m_%d_%I_%M_%S_%p")------------------------- sample_va ----------------------------------------------------------------------------------------------------"
python src/scripts/run.py -env $ENV_ -max_len $MAX_LEN -data_path $DATA_PATH -out_path $OUTPUT_PATH -model $MODEL -update_every $UPDATE_EVERY -agent $AGENT -K_epochs $K_EPOCHS -eps_clip $EPS_CLIP -lr $LR -word_emb_size $WORD_EMB_SIZE -hidden_size $HIDDEN_SIZE -num_episodes_train $NUM_EPISODE_TRAIN -lm_path $LM_PATH -reward $REWARD -num_episodes_test $NUM_EPISODE_TEST -mask_answers 1 -grad_clip 1 -fusion $FUSION -condition_answer $CONDITION_ANSWER -truncate_mode "sample_va" -debug $DEBUG
echo "$(date +"%Y_%m_%d_%I_%M_%S_%p")------------------------- proba threshold ----------------------------------------------------------------------------------------------------"
python src/scripts/run.py -env $ENV_ -max_len $MAX_LEN -data_path $DATA_PATH -out_path $OUTPUT_PATH -model $MODEL -update_every $UPDATE_EVERY -agent $AGENT -K_epochs $K_EPOCHS -eps_clip $EPS_CLIP -lr $LR -word_emb_size $WORD_EMB_SIZE -hidden_size $HIDDEN_SIZE -num_episodes_train $NUM_EPISODE_TRAIN -lm_path $LM_PATH -reward $REWARD -num_episodes_test $NUM_EPISODE_TEST -mask_answers 1 -grad_clip 1 -fusion $FUSION -condition_answer $CONDITION_ANSWER -truncate_mode "proba_thr" -debug $DEBUG
echo "$(date +"%Y_%m_%d_%I_%M_%S_%p")-------------------------  top p ---------------------------------------------------------------------------------------------------------------"
python src/scripts/run.py -env $ENV_ -max_len $MAX_LEN -data_path $DATA_PATH -out_path $OUTPUT_PATH -model $MODEL -update_every $UPDATE_EVERY -agent $AGENT -K_epochs $K_EPOCHS -eps_clip $EPS_CLIP -lr $LR -word_emb_size $WORD_EMB_SIZE -hidden_size $HIDDEN_SIZE -num_episodes_train $NUM_EPISODE_TRAIN -lm_path $LM_PATH -reward $REWARD -num_episodes_test $NUM_EPISODE_TEST -mask_answers 1 -grad_clip 1 -fusion $FUSION -condition_answer $CONDITION_ANSWER -truncate_mode "top_p" -top_p 0.8 -debug $DEBUG
echo "$(date +"%Y_%m_%d_%I_%M_%S_%p")------------------------- sample_va + epsilon truncation ----------------------------------------------------------------------------------------------------"
python src/scripts/run.py -env $ENV_ -max_len $MAX_LEN -data_path $DATA_PATH -out_path $OUTPUT_PATH -model $MODEL -update_every $UPDATE_EVERY -agent $AGENT -K_epochs $K_EPOCHS -eps_clip $EPS_CLIP -lr $LR -word_emb_size $WORD_EMB_SIZE -hidden_size $HIDDEN_SIZE -num_episodes_train $NUM_EPISODE_TRAIN -lm_path $LM_PATH -reward $REWARD -num_episodes_test $NUM_EPISODE_TEST -mask_answers 1 -grad_clip 1 -fusion $FUSION -condition_answer $CONDITION_ANSWER -truncate_mode "sample_va" -epsilon_truncated 0.2 -debug $DEBUG
echo "$(date +"%Y_%m_%d_%I_%M_%S_%p")------------------------- sample_va + GPT2 ----------------------------------------------------------------------------------------------------"
python src/scripts/run.py -env $ENV_ -max_len $MAX_LEN -data_path $DATA_PATH -out_path $OUTPUT_PATH -model $MODEL -update_every $UPDATE_EVERY -agent $AGENT -K_epochs $K_EPOCHS -eps_clip $EPS_CLIP -lr $LR -word_emb_size $WORD_EMB_SIZE -hidden_size $HIDDEN_SIZE -num_episodes_train $NUM_EPISODE_TRAIN -lm_path $LM_PATH -reward $REWARD -num_episodes_test $NUM_EPISODE_TEST -mask_answers 1 -grad_clip 1 -fusion $FUSION -condition_answer $CONDITION_ANSWER -truncate_mode "sample_va" -debug $DEBUG

REWARD="vqa"
echo "$(date +"%Y_%m_%d_%I_%M_%S_%p")------------------------- sample_va + oracle reward ----------------------------------------------------------------------------------------------------"
python src/scripts/run.py -env $ENV_ -max_len $MAX_LEN -data_path $DATA_PATH -out_path $OUTPUT_PATH -model $MODEL -update_every $UPDATE_EVERY -agent $AGENT -K_epochs $K_EPOCHS -eps_clip $EPS_CLIP -lr $LR -word_emb_size $WORD_EMB_SIZE -hidden_size $HIDDEN_SIZE -num_episodes_train $NUM_EPISODE_TRAIN -lm_path $LM_PATH -reward $REWARD -num_episodes_test $NUM_EPISODE_TEST -mask_answers 1 -grad_clip 1 -fusion $FUSION -condition_answer $CONDITION_ANSWER -debug $DEBUG -reward_path $REWARD_PATH -reward_vocab $REWARD_VOCAB -truncate_mode "sample_va"

#!/usr/bin/env bash
#echo "$(date +"%Y_%m_%d_%I_%M_%S_%p")-------------------------- train LM on CLEVR ---------------------------------------------------------------------------------------------------"
#python src/train/launch_train.py -task "lm" -dataset "clevr" -model "lstm" -num_layers 1 -emb_size 16 -hidden_size 16 -p_drop 0.1 -lr 0.001 -data_path "data" -out_path "output/temp" -bs 6 -ep 1 -num_workers 6
#echo "$(date +"%Y_%m_%d_%I_%M_%S_%p")-------------------------- train LM on VQA ---------------------------------------------------------------------------------------------------"
#python src/train/launch_train.py -task "lm" -dataset "vqa" -model "lstm" -num_layers 1 -emb_size 16 -hidden_size 16 -p_drop 0.1 -lr 0.001 -data_path "data/vqa-v2" -features_path "data/vqa-v2/coco_trainval.lmdb" -out_path "output/temp" -bs 512 -ep 1 -num_workers 6
#echo "$(date +"%Y_%m_%d_%I_%M_%S_%p")-------------------------- train policy on CLEVR ---------------------------------------------------------------------------------------------------"
#python src/train/launch_train.py -task "policy" -dataset "clevr" -data_path "data" -out_path "output/temp" -emb_size 16 -hidden_size 16 -bs 512 -ep 1 -num_workers 6 -fusion "cat" -max_samples 5000
#echo "$(date +"%Y_%m_%d_%I_%M_%S_%p")-------------------------- train policy on CLEVR - condition_answer  ---------------------------------------------------------------------------------------------------"
#python src/train/launch_train.py -task "policy" -dataset "clevr" -data_path "data" -out_path "output/temp" -emb_size 16 -hidden_size 16 -bs 512 -ep 1 -num_workers 6 -fusion "cat" -condition_answer "after_fusion" -max_samples 5000
#echo "$(date +"%Y_%m_%d_%I_%M_%S_%p")-------------------------- train policy on VQA  ---------------------------------------------------------------------------------------------------"
#python src/train/launch_train.py -task "policy" -dataset "vqa" -data_path "data" -out_path "output/temp" -emb_size 16 -hidden_size 16 -bs 512 -ep 1 -num_workers 6 -fusion "average"
#echo "$(date +"%Y_%m_%d_%I_%M_%S_%p")-------------------------- train policy on VQA - condition_answer  ---------------------------------------------------------------------------------"
#python src/train/launch_train.py -task "policy" -dataset "vqa" -data_path "data/vqa-v2" --features_path "data/vqa-v2/coco_trainval.lmdb" -out_path "output/temp" -emb_size 16 -hidden_size 16 -bs 512 -ep 1 -num_workers 6 -fusion "average" -condition_answer "after_fusion"
#echo "$(date +"%Y_%m_%d_%I_%M_%S_%p")----------------------END-------------------------------------------------------------------"
