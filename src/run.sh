python run -max_len 10 -data_path "data"
-out_path "output/reinforce_debug" -model "lstm" -update_every 20
  -entropy_coeff
  -lm_path
  -policy_path
  -debug 0
  -agent REINFORCE
  -num_truncated 10

python run -max_len 5 -data_path "data" \
-out_path "output/RL/reinforce_debug" \
-model "lstm" -update_every 1 \
-debug 1 \
-agent REINFORCE \
