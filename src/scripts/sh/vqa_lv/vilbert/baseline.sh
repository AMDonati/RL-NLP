#!/usr/bin/env bash
echo "------REINFORCE----------------"
python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_size_smallvocab_vilbert/baseline" -model "lstm" -update_every 128 -agent "REINFORCE" -K_epochs 20 -eps_clip 0.02 -lr 0.0001 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 30000 -debug "0,2000" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "vilbert" -reward_vocab "output/vilbert_vqav2/bert_base_6layer_6conect.json" -reward_path "output/vilbert_vqav2/model.bin" -num_episodes_test 100 -mask_answers 1 -fusion "average" -condition_answer "after_fusion" -device_id 0 -min_data 1 -policy_path "output/vqa_policy_512_1024_answer_smallvocab/model.pt"
python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_size_smallvocab_vilbert/baseline" -model "lstm" -update_every 128 -agent "REINFORCE" -K_epochs 20 -eps_clip 0.02 -lr 0.0001 -optimizer "sgd" -word_emb_size 512 -hidden_size 1024 -num_episodes_train 30000 -debug "0,2000" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "vilbert" -reward_vocab "../vilbert-multi-task/config/bert_base_6layer_6conect.json" -reward_path "../vilbert-multi-task/save/VQA_bert_base_6layer_6conect-finetune_from_multi_task_model_task1/pytorch_19.bin" -num_episodes_test 100 -mask_answers 1 -fusion "average" -condition_answer "after_fusion" -device_id 2 -min_data 1 -policy_path "output/vqa_policy_512_1024_answer_smallvocab/model.pt"
python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_size_smallvocab_vilbert/baseline" -model "lstm" -update_every 128 -agent "REINFORCE" -K_epochs 20 -eps_clip 0.02 -lr 0.005 -optimizer "sgd" -word_emb_size 512 -hidden_size 1024 -num_episodes_train 30000 -debug "0,2000" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "vilbert" -reward_vocab "../vilbert-multi-task/config/bert_base_6layer_6conect.json" -reward_path "../vilbert-multi-task/save/VQA_bert_base_6layer_6conect-finetune_from_multi_task_model_task1/pytorch_19.bin" -num_episodes_test 100 -mask_answers 1 -fusion "average" -condition_answer "after_fusion" -device_id 2 -min_data 1 -policy_path "output/vqa_policy_512_1024_answer_smallvocab/model.pt"
echo "-----------PPO--------------------"
python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_size_smallvocab_vilbert/baseline" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.0001 -optimizer "sgd" -word_emb_size 512 -hidden_size 1024 -num_episodes_train 30000 -debug "0,2000" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "vilbert" -reward_vocab "output/vilbert_vqav2/bert_base_6layer_6conect.json" -reward_path "output/vilbert_vqav2/model.bin" -num_episodes_test 100 -mask_answers 1 -fusion "average" -condition_answer "after_fusion" -device_id 0 -min_data 1 -policy_path "output/vqa_policy_512_1024_answer_smallvocab/model.pt"
python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_size_smallvocab_vilbert/baseline" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.005 -optimizer "sgd" -word_emb_size 512 -hidden_size 1024 -num_episodes_train 30000 -debug "0,2000" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "vilbert" -reward_vocab "output/vilbert_vqav2/bert_base_6layer_6conect.json" -reward_path "output/vilbert_vqav2/model.bin" -num_episodes_test 100 -mask_answers 1 -fusion "average" -condition_answer "after_fusion" -device_id 2 -min_data 1 -policy_path "output/vqa_policy_512_1024_answer_smallvocab/model.pt"