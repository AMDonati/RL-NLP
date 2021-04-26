#!/bin/bash
#SBATCH --job-name=T-pth-temp
#SBATCH --qos=qos_gpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8g
#SBATCH --output=slurm_out/vqa/T-pth-temp-%j.out
#SBATCH --error=slurm_out/vqa/T-pth-temp-%j.err
#SBATCH --time=20:00:00


export TMPDIR=$JOBSCRATCH
module purge
module load  pytorch-gpu/py3/1.7.1
conda activate rl-nlp-2
export PYTHONPATH=src:${PYTHONPATH}


MODEL_PATH="output/RL/VQA_vilbert_small/proba_thr0.005_temp1.5_tmin1.0"
NUM_EPISODES_TEST=20000

set -x
srun python -u src/scripts/test.py -models_path $MODEL_PATH -num_episodes_test $NUM_EPISODES_TEST -test_metrics "return" "oracle" "dialog" "bleu" "meteor" "cider" "ppl_dialog_lm" "size_valid_actions" "sum_probs" "selfbleu" "dialogimage" "language_score" "kurtosis" "peakiness"