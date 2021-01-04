#!/usr/bin/env bash
echo "-------------------------- lr = 0.0001---------------------------------------------------------------------------------------------------"
python src/scripts/run.py -max_len 20 -data_path "data" -out_path "output/RL/2021_exp_clvr_vqa_20000img_len20/baseline" -model "lstm" -update_every 128 -agent "REINFORCE" -optimizer "sgd" -K_epochs 20 -eps_clip 0.02 -lr 0.0001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 30000 -debug "0,20000" -lm_path "output/lm_model/model.pt" -reward "vqa" -reward_path "output/vqa_model_film/model.pt" -condition_answer "after_fusion" -num_episodes_test 500 -reward_vocab "data/closure_vocab.json" -mask_answers 1 -policy_path "output/SL_LSTM_32_64_vqa/model.pt" -device_id 1
python src/scripts/run.py -max_len 20 -data_path "data" -out_path "output/RL/2021_exp_clvr_vqa_20000img_len20/baseline" -model "lstm" -update_every 128 -agent "REINFORCE" -K_epochs 20 -eps_clip 0.02 -lr 0.0001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 30000 -debug "0,20000" -lm_path "output/lm_model/model.pt" -reward "vqa" -reward_path "output/vqa_model_film/model.pt" -condition_answer "after_fusion" -num_episodes_test 500 -reward_vocab "data/closure_vocab.json" -mask_answers 1 -policy_path "output/SL_LSTM_32_64_vqa/model.pt" -device_id 1
echo "-------------------------- lr = 0.0005---------------------------------------------------------------------------------------------------"
python src/scripts/run.py -max_len 20 -data_path "data" -out_path "output/RL/2021_exp_clvr_vqa_20000img_len20/baseline" -model "lstm" -update_every 128 -agent "REINFORCE" -optimizer "sgd" -K_epochs 20 -eps_clip 0.02 -lr 0.0005 -word_emb_size 32 -hidden_size 64 -num_episodes_train 30000 -debug "0,20000" -lm_path "output/lm_model/model.pt" -reward "vqa" -reward_path "output/vqa_model_film/model.pt" -condition_answer "after_fusion" -num_episodes_test 500 -reward_vocab "data/closure_vocab.json" -mask_answers 1 -policy_path "output/SL_LSTM_32_64_vqa/model.pt" -device_id 1
python src/scripts/run.py -max_len 20 -data_path "data" -out_path "output/RL/2021_exp_clvr_vqa_20000img_len20/baseline" -model "lstm" -update_every 128 -agent "REINFORCE" -K_epochs 20 -eps_clip 0.02 -lr 0.0005 -word_emb_size 32 -hidden_size 64 -num_episodes_train 30000 -debug "0,20000" -lm_path "output/lm_model/model.pt" -reward "vqa" -reward_path "output/vqa_model_film/model.pt" -condition_answer "after_fusion" -num_episodes_test 500 -reward_vocab "data/closure_vocab.json" -mask_answers 1 -policy_path "output/SL_LSTM_32_64_vqa/model.pt" -device_id 1
echo "-------------------------- lr = 0.001---------------------------------------------------------------------------------------------------"
python src/scripts/run.py -max_len 20 -data_path "data" -out_path "output/RL/2021_exp_clvr_vqa_20000img_len20/baseline" -model "lstm" -update_every 128 -agent "REINFORCE" -optimizer "sgd" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 30000 -debug "0,20000" -lm_path "output/lm_model/model.pt" -reward "vqa" -reward_path "output/vqa_model_film/model.pt" -condition_answer "after_fusion" -num_episodes_test 500 -reward_vocab "data/closure_vocab.json" -mask_answers 1 -policy_path "output/SL_LSTM_32_64_vqa/model.pt" -device_id 1
python src/scripts/run.py -max_len 20 -data_path "data" -out_path "output/RL/2021_exp_clvr_vqa_20000img_len20/baseline" -model "lstm" -update_every 128 -agent "REINFORCE" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 30000 -debug "0,20000" -lm_path "output/lm_model/model.pt" -reward "vqa" -reward_path "output/vqa_model_film/model.pt" -condition_answer "after_fusion" -num_episodes_test 500 -reward_vocab "data/closure_vocab.json" -mask_answers 1 -policy_path "output/SL_LSTM_32_64_vqa/model.pt" -device_id 1
