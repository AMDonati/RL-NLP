#!/bin/bash
#SBATCH --job-name=debug-vqa
#SBATCH --qos=qos_gpu-t3
#SBATCH -C v100-32g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8g
#SBATCH --output=slurm_out/debugvqa%j.out
#SBATCH --error=slurm_out/debugvqa%j.err
#SBATCH --time=03:00:00

export TMPDIR=$JOBSCRATCH
module purge
module load  pytorch-gpu/py3/1.7.1
conda activate rl-nlp-2

export PYTHONPATH=src:${PYTHONPATH}


echo "---------------------------------------------------------------------------------------------------------------------------------"
echo "------------------------- VQAv2 -------------------------------------------------------------------------------------------------"
echo "---------------------------------------------------------------------------------------------------------------------------------"

DATA_PATH="data/vqa-v2/"
FEATURES_PATH="data/vqa-v2/coco_trainval.lmdb/"
LM_PATH="output/vqa_lm_model/model.pt"
OUTPUT_PATH="output/RL/debug"
POLICY_PATH="output/vqa_policy_512_1024_answer/model.pt"
VILBERT_VOCAB="output/vilbert_vqav2/bert_base_6layer_6conect.json"
VILBERT_PATH="output/vilbert_vqav2/model.bin"
K_EPOCHS=10
MAX_LEN=5
UPDATE_EVERY=7
NUM_EPISODE_TRAIN=10
NUM_EPISODE_TEST=5
ENV_="vqa"
MODEL="lstm"
AGENT="PPO"
LR=0.001
WORD_EMB_SIZE=512
HIDDEN_SIZE=1024
EPS_CLIP=0.01
REWARD="lv_norm"
FUSION="average"
CONDITION_ANSWER="after_fusion"


echo "$(date +"%Y_%m_%d_%I_%M_%p")-------------------------- Scratch ---------------------------------------------------------------------------------------------------"
python src/scripts/run.py -env $ENV_ -max_len $MAX_LEN -data_path $DATA_PATH -out_path $OUTPUT_PATH -model $MODEL -update_every $UPDATE_EVERY -agent $AGENT -K_epochs $K_EPOCHS -eps_clip $EPS_CLIP -lr $LR -word_emb_size $WORD_EMB_SIZE -hidden_size $HIDDEN_SIZE -num_episodes_train $NUM_EPISODE_TRAIN -lm_path $LM_PATH -reward $REWARD -num_episodes_test $NUM_EPISODE_TEST -mask_answers 1 -grad_clip 1 -fusion $FUSION -condition_answer $CONDITION_ANSWER -features_path $FEATURES_PATH
echo "$(date +"%Y_%m_%d_%I_%M_%p")------------------------- pretrain ----------------------------------------------------------------------------------------------------"
python src/scripts/run.py -env $ENV_ -max_len $MAX_LEN -data_path $DATA_PATH -out_path $OUTPUT_PATH -model $MODEL -update_every $UPDATE_EVERY -agent $AGENT -K_epochs $K_EPOCHS -eps_clip $EPS_CLIP -lr $LR -word_emb_size $WORD_EMB_SIZE -hidden_size $HIDDEN_SIZE -num_episodes_train $NUM_EPISODE_TRAIN -lm_path $LM_PATH -reward $REWARD -num_episodes_test $NUM_EPISODE_TEST -mask_answers 1 -grad_clip 1 -fusion $FUSION -condition_answer $CONDITION_ANSWER -features_path $FEATURES_PATH -policy_path $POLICY_PATH
#echo "$(date +"%Y_%m_%d_%I_%M_%p")------------------------- top k ----------------------------------------------------------------------------------------------------"
#python src/scripts/run.py -env $ENV_ -max_len $MAX_LEN -data_path $DATA_PATH -out_path $OUTPUT_PATH -model $MODEL -update_every $UPDATE_EVERY -agent $AGENT -K_epochs $K_EPOCHS -eps_clip $EPS_CLIP -lr $LR -word_emb_size $WORD_EMB_SIZE -hidden_size $HIDDEN_SIZE -num_episodes_train $NUM_EPISODE_TRAIN -lm_path $LM_PATH -reward $REWARD -num_episodes_test $NUM_EPISODE_TEST -mask_answers 1 -grad_clip 1 -fusion $FUSION -condition_answer $CONDITION_ANSWER -features_path $FEATURES_PATH -truncate_mode "top_k" -num_truncated 20
#echo "$(date +"%Y_%m_%d_%I_%M_%p")------------------------- sample_va ----------------------------------------------------------------------------------------------------"
#python src/scripts/run.py -env $ENV_ -max_len $MAX_LEN -data_path $DATA_PATH -out_path $OUTPUT_PATH -model $MODEL -update_every $UPDATE_EVERY -agent $AGENT -K_epochs $K_EPOCHS -eps_clip $EPS_CLIP -lr $LR -word_emb_size $WORD_EMB_SIZE -hidden_size $HIDDEN_SIZE -num_episodes_train $NUM_EPISODE_TRAIN -lm_path $LM_PATH -reward $REWARD -num_episodes_test $NUM_EPISODE_TEST -mask_answers 1 -grad_clip 1 -fusion $FUSION -condition_answer $CONDITION_ANSWER -features_path $FEATURES_PATH -truncate_mode "sample_va"
echo "$(date +"%Y_%m_%d_%I_%M_%p")------------------------- proba threshold ----------------------------------------------------------------------------------------------------"
python src/scripts/run.py -env $ENV_ -max_len $MAX_LEN -data_path $DATA_PATH -out_path $OUTPUT_PATH -model $MODEL -update_every $UPDATE_EVERY -agent $AGENT -K_epochs $K_EPOCHS -eps_clip $EPS_CLIP -lr $LR -word_emb_size $WORD_EMB_SIZE -hidden_size $HIDDEN_SIZE -num_episodes_train $NUM_EPISODE_TRAIN -lm_path $LM_PATH -reward $REWARD -num_episodes_test $NUM_EPISODE_TEST -mask_answers 1 -grad_clip 1 -fusion $FUSION -condition_answer $CONDITION_ANSWER -features_path $FEATURES_PATH -truncate_mode "proba_thr"
echo "$(date +"%Y_%m_%d_%I_%M_%p")-------------------------  top p + data_min ---------------------------------------------------------------------------------------------------------------"
python src/scripts/run.py -env $ENV_ -max_len $MAX_LEN -data_path $DATA_PATH -out_path $OUTPUT_PATH -model $MODEL -update_every $UPDATE_EVERY -agent $AGENT -K_epochs $K_EPOCHS -eps_clip $EPS_CLIP -lr $LR -word_emb_size $WORD_EMB_SIZE -hidden_size $HIDDEN_SIZE -num_episodes_train $NUM_EPISODE_TRAIN -lm_path $LM_PATH -reward $REWARD -num_episodes_test $NUM_EPISODE_TEST -mask_answers 1 -grad_clip 1 -fusion $FUSION -condition_answer $CONDITION_ANSWER -features_path $FEATURES_PATH -truncate_mode "top_p" -top_p 0.8 -min_data 1
#echo "$(date +"%Y_%m_%d_%I_%M_%p")------------------------- sample_va + epsilon truncation ----------------------------------------------------------------------------------------------------"
#python src/scripts/run.py -env $ENV_ -max_len $MAX_LEN -data_path $DATA_PATH -out_path $OUTPUT_PATH -model $MODEL -update_every $UPDATE_EVERY -agent $AGENT -K_epochs $K_EPOCHS -eps_clip $EPS_CLIP -lr $LR -word_emb_size $WORD_EMB_SIZE -hidden_size $HIDDEN_SIZE -num_episodes_train $NUM_EPISODE_TRAIN -lm_path $LM_PATH -reward $REWARD -num_episodes_test $NUM_EPISODE_TEST -mask_answers 1 -grad_clip 1 -fusion $FUSION -condition_answer $CONDITION_ANSWER -features_path $FEATURES_PATH -truncate_mode "sample_va" -epsilon_truncated 0.2
echo "$(date +"%Y_%m_%d_%I_%M_%p")------------------------- sample_va + GPT2 ----------------------------------------------------------------------------------------------------"
python src/scripts/run.py -env $ENV_ -max_len $MAX_LEN -data_path $DATA_PATH -out_path $OUTPUT_PATH -model $MODEL -update_every $UPDATE_EVERY -agent $AGENT -K_epochs $K_EPOCHS -eps_clip $EPS_CLIP -lr $LR -word_emb_size $WORD_EMB_SIZE -hidden_size $HIDDEN_SIZE -num_episodes_train $NUM_EPISODE_TRAIN -lm_path $LM_PATH -reward $REWARD -num_episodes_test $NUM_EPISODE_TEST -mask_answers 1 -grad_clip 1 -fusion $FUSION -condition_answer $CONDITION_ANSWER -features_path $FEATURES_PATH -truncate_mode "sample_va"
#echo "$(date +"%Y_%m_%d_%I_%M_%p")------------------------- sample_va + GPT2 + datamin----------------------------------------------------------------------------------------------------"
#python src/scripts/run.py -env $ENV_ -max_len $MAX_LEN -data_path $DATA_PATH -out_path $OUTPUT_PATH -model $MODEL -update_every $UPDATE_EVERY -agent $AGENT -K_epochs $K_EPOCHS -eps_clip $EPS_CLIP -lr $LR -word_emb_size $WORD_EMB_SIZE -hidden_size $HIDDEN_SIZE -num_episodes_train $NUM_EPISODE_TRAIN -lm_path $LM_PATH -reward $REWARD -num_episodes_test $NUM_EPISODE_TEST -mask_answers 1 -grad_clip 1 -fusion $FUSION -condition_answer $CONDITION_ANSWER -features_path $FEATURES_PATH -truncate_mode "sample_va"
echo "$(date +"%Y_%m_%d_%I_%M_%p")------------------------- VILBERT proba threshold ----------------------------------------------------------------------------------------------------"
python src/scripts/run.py -env $ENV_ -max_len $MAX_LEN -data_path $DATA_PATH -out_path $OUTPUT_PATH -model $MODEL -reward "vilbert" -update_every $UPDATE_EVERY -agent $AGENT -K_epochs $K_EPOCHS -eps_clip $EPS_CLIP -lr $LR -word_emb_size $WORD_EMB_SIZE -hidden_size $HIDDEN_SIZE -num_episodes_train $NUM_EPISODE_TRAIN -lm_path $LM_PATH -num_episodes_test $NUM_EPISODE_TEST -mask_answers 1 -grad_clip 1 -fusion $FUSION -condition_answer $CONDITION_ANSWER -features_path $FEATURES_PATH -truncate_mode "proba_thr" -reward_vocab $VILBERT_VOCAB -reward_path $VILBERT_PATH -answer_sampl "uniform"