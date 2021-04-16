#!/bin/bash
#SBATCH --job-name=clevr-pth0.1
#SBATCH --qos=qos_gpu-t3
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8g
#SBATCH --output=slurm_out/clevr/pth0.1-%j.out
#SBATCH --error=slurm_out/clevr/pth0.1-%j.err
#SBATCH --time=20:00:00

export TMPDIR=$JOBSCRATCH
module purge
module load  pytorch-gpu/py3/1.7.1
conda activate rl-nlp-2

export PYTHONPATH=src:${PYTHONPATH}

OUTPUT_PATH="output/RL/CLEVR_lm-ext"

set -x
srun python -u src/scripts/run.py -max_len 20 -data_path "data" -out_path $OUTPUT_PATH -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 50000 -debug "0,20000" -lm_path "output/lm_ext/model.pt" -reward "vqa" -reward_path "output/vqa_model_film/model.pt" -condition_answer "after_fusion" -num_episodes_test 5000 -reward_vocab "data/closure_vocab.json" -mask_answers 1 -truncate_mode "proba_thr" -p_th 0.1 -grad_clip 1

