python run -max_len 10 -data_path "data"
-out_path "output/reinforce_debug" -model "lstm" -update_every 20
  -entropy_coeff
  -lm_path
  -policy_path
  -debug 0
  -agent REINFORCE
  -num_truncated 10


# debug mode:

python run -max_len 5 -data_path "data" \
-out_path "output/RL/reinforce_debug" \
-model "lstm_word " -update_every 1 \
-debug 1 \
-agent REINFORCE \

python src/scripts/run.py -max_len 5 -data_path "data" \
-out_path "output/RL/ppo_debug" \
-model "lstm" -update_every 20 \
-debug 1 \
-agent PPO -K_epochs 10 -eps_clip 0.02 -word_emb_size 8 -hidden_size 24

python src/scripts/run.py -max_len 10 -data_path "data" \
-out_path "output/RL/ppo_debug_len_10" \
-model "lstm" -update_every 20 \
-debug 1 \
-agent "PPO" -K_epochs 10 -eps_clip 0.02 -word_emb_size 8 -hidden_size 24

# with language model:

python src/scripts/run.py -max_len 10 -data_path "data" \
-out_path "output/RL/ppo_debug_len_10" \
-model "lstm" -update_every 20 \
-debug 1 \
-policy_path "output/SL_LSTMBatch_8_24.001/model.pt"
-agent "PPO" -K_epochs 10 -eps_clip 0.02 -word_emb_size 8 -hidden_size 24

## REINFORCE

python src/scripts/run.py -max_len 10 -data_path "data" \
-out_path "output/RL/reinforce_debug_len_10" \
-model "lstm" -update_every 1 \
-debug 1 \
--policy_path "output/SL_LSTMBatch_8_24.001/model.pt"
-agent "REINFORCE" -lr 0.005 -word_emb_size 8 -hidden_size 24


########################## scratch ############

python src/scripts/run.py -max_len 10 -data_path "data" -out_path "output/RL/20_img_len_10_pretrained" -model "lstm" -update_every 64 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.005 -word_emb_size 32 -hidden_size 64 -num_episodes_train 50000 -lm_path "output/lm_model/model.pt" -num_truncated 10 -debug "0,50" -grad_clip 1

python src/scripts/run.py -max_len 10 -data_path "data" -out_path "output/RL/20_img_len_10_pretrained" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 50000 -lm_path "output/lm_model/model.pt" -num_truncated 10 -debug "0,50" -grad_clip 2

python src/scripts/run.py -max_len 10 -data_path "data" -out_path "output/RL/20_img_len_10_pretrained" -model "lstm" -update_every 64 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 50000 -lm_path "output/lm_model/model.pt" -num_truncated 10 -debug "0,50" -grad_clip 1

python src/scripts/run.py -max_len 10 -data_path "data" -out_path "output/RL/50_img_len_10_pretrained" -model "lstm" -update_every 64 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -grad_clip 1 -lr 0.005 -word_emb_size 32 -hidden_size 64 -num_episodes_train 50000 -lm_path "output/lm_model/model.pt" -num_truncated 10 -debug "0,50" -policy_path "output/lstmwordbatch_pretraining/SL_LSTM_32_64/model.pt"

python src/scripts/run.py -max_len 10 -data_path "data" -out_path "output/RL/20_img_len_10_pretrained" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 5 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 50000 -lm_path "output/best_model/model.pt" -num_truncated 10 -debug "0,50" -grad_clip 1

# to launch:
python src/scripts/run.py -max_len 10 -data_path "data" -out_path "output/RL/100_img_len_10_pretrained" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 200000 -lm_path "output/lm_model/model.pt" -num_truncated 10 -debug "0,100" -grad_clip 2 -num_questions 8

python src/scripts/run.py -max_len 10 -data_path "data" -out_path "output/RL/100_img_len_10_pretrained" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 200000 -num_truncated 87 -debug "0,100" -grad_clip 2 -num_questions 8

# 17.7.2020
python src/scripts/run.py -max_len 10 -data_path "data" -out_path "output/RL/100_img_len_10_pretrained" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 200000 -lm_path "output/lm_model/model.pt" -num_truncated 10 -debug "0,100" -grad_clip 1 -num_questions 1

python src/scripts/run.py -max_len 10 -data_path "data" -out_path "output/RL/100_img_len_10_q1_pretrained" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 200000 -num_truncated 87 -debug "0,100" -grad_clip 1 -num_questions 1

#### PRETRAIN ############

python src/scripts/run.py -max_len 10 -data_path "data" -out_path "output/RL/20_img_len_10_pretrained" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 8 -hidden_size 24 -num_episodes_train 50000 -policy_path "output/lstmwordbatch_pretraining/SL_LSTM_8_24/model.pt" -debug "0,50"

python src/scripts/run.py -max_len 10 -data_path "data" -out_path "output/RL/50_img_len_10_pretrained" -model "lstm" -update_every 64 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 50000 -policy_path "output/lstmwordbatch_pretraining/SL_LSTM_32_64/model.pt" -debug "0,50"


# Telecom Paris Machine.
python src/scripts/run.py -max_len 10 -data_path "data" -out_path "output/RL/20_img_len_10_pretrained" -model "lstm" -update_every 64 -agent "PPO" -K_epochs 5 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 50000 -policy_path "output/SL_LSTM_32_64/model.pt" -debug "0,50"

#To launch.
python src/scripts/run.py -max_len 10 -data_path "data" -out_path "output/RL/100_img_len_10_q1_pretrained" -model "lstm" -update_every 64 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 200000 -policy_path "output/lstmwordbatch_pretraining/SL_LSTM_32_64/model.pt" -debug "0,100" -num_questions 1












