#!/usr/bin/env bash
echo "---- lr = 0.0001 -----------------------"
python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_debug_pretrained_baseline/sgd_warmup" -model "lstm" -update_every 128 -agent "REINFORCE" -optimizer "rmsprop" -lr 0.0001 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 10000 -debug "0,2000" -lm_path "gpt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -fusion "average" -condition_answer "after_fusion" -policy_path "output/vqa_policy_512_1024_answer_smallvocab/model.pt" -device_id 1 -min_data 1
python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_debug_pretrained_baseline/sgd_warmup" -model "lstm" -update_every 128 -agent "REINFORCE" -optimizer "rmsprop" -lr 0.0001 -opt_schedule "cyclic" -word_emb_size 512 -hidden_size 1024 -num_episodes_train 10000 -debug "0,2000" -lm_path "gpt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -fusion "average" -condition_answer "after_fusion" -policy_path "output/vqa_policy_512_1024_answer_smallvocab/model.pt" -device_id 1 -min_data 1
python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_debug_pretrained_baseline/sgd_warmup" -model "lstm" -update_every 128 -agent "REINFORCE" -optimizer "rmsprop" -lr 0.0001 -opt_schedule "cyclic_multi" -word_emb_size 512 -hidden_size 1024 -num_episodes_train 10000 -debug "0,2000" -lm_path "gpt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -fusion "average" -condition_answer "after_fusion" -policy_path "output/vqa_policy_512_1024_answer_smallvocab/model.pt" -device_id 1 -min_data 1
python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_debug_pretrained_baseline/sgd_warmup" -model "lstm" -update_every 128 -agent "REINFORCE" -optimizer "rmsprop" -lr 0.0001 -opt_schedule "cyclic" -div_factor 10 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 10000 -debug "0,2000" -lm_path "gpt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -fusion "average" -condition_answer "after_fusion" -policy_path "output/vqa_policy_512_1024_answer_smallvocab/model.pt" -device_id 1 -min_data 1
python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_debug_pretrained_baseline/sgd_warmup" -model "lstm" -update_every 128 -agent "REINFORCE" -optimizer "rmsprop" -lr 0.0001 -opt_schedule "cyclic_multi" -div_factor 10 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 10000 -debug "0,2000" -lm_path "gpt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -fusion "average" -condition_answer "after_fusion" -policy_path "output/vqa_policy_512_1024_answer_smallvocab/model.pt" -device_id 1 -min_data 1
python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_debug_pretrained_baseline/sgd_warmup" -model "lstm" -update_every 128 -agent "REINFORCE" -optimizer "rmsprop" -lr 0.0001 -opt_schedule "cyclic" -div_factor 50 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 10000 -debug "0,2000" -lm_path "gpt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -fusion "average" -condition_answer "after_fusion" -policy_path "output/vqa_policy_512_1024_answer_smallvocab/model.pt" -device_id 1 -min_data 1
python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_debug_pretrained_baseline/sgd_warmup" -model "lstm" -update_every 128 -agent "REINFORCE" -optimizer "rmsprop" -lr 0.0001 -opt_schedule "cyclic_multi" -div_factor 50 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 10000 -debug "0,2000" -lm_path "gpt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -fusion "average" -condition_answer "after_fusion" -policy_path "output/vqa_policy_512_1024_answer_smallvocab/model.pt" -device_id 1 -min_data 1
python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_debug_pretrained_baseline/sgd_warmup" -model "lstm" -update_every 128 -agent "REINFORCE" -optimizer "rmsprop" -lr 0.0001 -opt_schedule "WR" -word_emb_size 512 -hidden_size 1024 -num_episodes_train 10000 -debug "0,2000" -lm_path "gpt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -fusion "average" -condition_answer "after_fusion" -policy_path "output/vqa_policy_512_1024_answer_smallvocab/model.pt" -device_id 1 -min_data 1
echo "---- lr = 0.00005-----------------------"
python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_debug_pretrained_baseline/sgd_warmup" -model "lstm" -update_every 128 -agent "REINFORCE" -optimizer "rmsprop" -lr 0.00005 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 10000 -debug "0,2000" -lm_path "gpt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -fusion "average" -condition_answer "after_fusion" -policy_path "output/vqa_policy_512_1024_answer_smallvocab/model.pt" -device_id 1 -min_data 1
python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_debug_pretrained_baseline/sgd_warmup" -model "lstm" -update_every 128 -agent "REINFORCE" -optimizer "rmsprop" -lr 0.00005 -opt_schedule "cyclic" -word_emb_size 512 -hidden_size 1024 -num_episodes_train 10000 -debug "0,2000" -lm_path "gpt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -fusion "average" -condition_answer "after_fusion" -policy_path "output/vqa_policy_512_1024_answer_smallvocab/model.pt" -device_id 1 -min_data 1
python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_debug_pretrained_baseline/sgd_warmup" -model "lstm" -update_every 128 -agent "REINFORCE" -optimizer "rmsprop" -lr 0.00005 -opt_schedule "cyclic_multi" -word_emb_size 512 -hidden_size 1024 -num_episodes_train 10000 -debug "0,2000" -lm_path "gpt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -fusion "average" -condition_answer "after_fusion" -policy_path "output/vqa_policy_512_1024_answer_smallvocab/model.pt" -device_id 1 -min_data 1
python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_debug_pretrained_baseline/sgd_warmup" -model "lstm" -update_every 128 -agent "REINFORCE" -optimizer "rmsprop" -lr 0.00005 -opt_schedule "cyclic" -div_factor 10 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 10000 -debug "0,2000" -lm_path "gpt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -fusion "average" -condition_answer "after_fusion" -policy_path "output/vqa_policy_512_1024_answer_smallvocab/model.pt" -device_id 1 -min_data 1
python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_debug_pretrained_baseline/sgd_warmup" -model "lstm" -update_every 128 -agent "REINFORCE" -optimizer "rmsprop" -lr 0.00005 -opt_schedule "cyclic_multi" -div_factor 10 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 10000 -debug "0,2000" -lm_path "gpt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -fusion "average" -condition_answer "after_fusion" -policy_path "output/vqa_policy_512_1024_answer_smallvocab/model.pt" -device_id 1 -min_data 1
python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_debug_pretrained_baseline/sgd_warmup" -model "lstm" -update_every 128 -agent "REINFORCE" -optimizer "rmsprop" -lr 0.00005 -opt_schedule "cyclic" -div_factor 50 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 10000 -debug "0,2000" -lm_path "gpt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -fusion "average" -condition_answer "after_fusion" -policy_path "output/vqa_policy_512_1024_answer_smallvocab/model.pt" -device_id 1 -min_data 1
python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_debug_pretrained_baseline/sgd_warmup" -model "lstm" -update_every 128 -agent "REINFORCE" -optimizer "rmsprop" -lr 0.00005 -opt_schedule "cyclic_multi" -div_factor 50 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 10000 -debug "0,2000" -lm_path "gpt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -fusion "average" -condition_answer "after_fusion" -policy_path "output/vqa_policy_512_1024_answer_smallvocab/model.pt" -device_id 1 -min_data 1
python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_debug_pretrained_baseline/sgd_warmup" -model "lstm" -update_every 128 -agent "REINFORCE" -optimizer "rmsprop" -lr 0.00005 -opt_schedule "WR" -word_emb_size 512 -hidden_size 1024 -num_episodes_train 10000 -debug "0,2000" -lm_path "gpt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -fusion "average" -condition_answer "after_fusion" -policy_path "output/vqa_policy_512_1024_answer_smallvocab/model.pt" -device_id 1 -min_data 1