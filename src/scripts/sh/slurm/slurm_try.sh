#!/bin/bash
#SBATCH --job-name=rl-nlp-debug-slurm
#SBATCH --ntasks=16
#SBATCH --time=50:00:00
#SBATCH --mem=8g
#SBATCH --gpus=4


module load miniconda
conda activate rl-nlp

python src/scripts/run.py -max_len 20 -data_path "data" -out_path "output/RL/debug" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 0 -debug "0,20000" -lm_path "output/lm_ext/model.pt" -reward "vqa" -reward_path "output/vqa_model_film/model.pt" -condition_answer "after_fusion" -num_episodes_test 5000 -reward_vocab "data/closure_vocab.json" -mask_answers 1 -truncate_mode "proba_thr" -p_th 0.1 -grad_clip 1 -eval_modes "sampling_ranking_lm"