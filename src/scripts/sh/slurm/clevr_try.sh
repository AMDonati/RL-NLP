#!/bin/bash
#SBATCH --job-name=clevrfeatures
#SBATCH --qos=qos_gpu-t3
#SBATCH -C v100-32g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8g
#SBATCH --output=clevrfeatures%j.out
#SBATCH --error=clevrfeatures%j.err
#SBATCH --time=03:00:00

export TMPDIR=$JOBSCRATCH
module purge
module load  pytorch-gpu/py3/1.7.1
conda activate rl-nlp

export PYTHONPATH=src:${PYTHONPATH}

set -x
python src/scripts/run.py -max_len 20 -data_path "data" -out_path "output/RL/debug" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 10 -debug "0,20000" -lm_path "output/lm_model/model.pt" -reward "vqa" -reward_path "output/vqa_model_film/model.pt" -condition_answer "after_fusion" -num_episodes_test 10 -reward_vocab "data/closure_vocab.json" -mask_answers 1 -truncate_mode "proba_thr" -p_th 0.1 -grad_clip 1 -eval_modes "sampling" "greedy"