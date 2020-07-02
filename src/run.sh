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









