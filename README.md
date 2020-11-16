# RL-NLP

## CLEVR Datasets: 
* [CLEVR original dataset page](https://cs.stanford.edu/people/jcjohns/clevr/)
* [CLEVR preprocessing github](https://github.com/facebookresearch/clevr-iep/blob/master/TRAINING.md)
* [CLOSURE dataset github page](https://github.com/rizar/CLOSURE)

## Downloading the dataset
The CLEVR Dataset can be downloaded [here](https://cs.stanford.edu/people/jcjohns/clevr/).   
The full dataset (questions + images) or only the questions part can be downloaded. 
To avoid memory issues on local machines, the first 21 images of the train/val/test datasets are available for download [here](https://drive.google.com/drive/folders/1OEy8Dfq2mO-vAiL9hFO1E_HbqC0wX4WB?usp=sharing).  
To download the dataset directly via the shell, you can run the following commands: 
`mkdir data` 
`wget https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip -O data/CLEVR_v1.0.zip`  
`unzip data/CLEVR_v1.0.zip -d data`

## Requirements 
* You can create a conda environment called rl-nlp: `conda create -n rl-nlp`
* And activate it: `conda activate rl-nlp`
* The required library can be installed via the file requirements.txt: `pip install -r requirements.txt`
* The code relies on the CLOSURE github: you need to install it with: `python -m pip install git+https://github.com/gqkc/CLOSURE.git --upgrade`
* And on the VILBERT multi-task github: `python -m pip install git+https://github.com/gqkc/vilbert-multi-task.git --upgrade`

## Data preprocessing
### CLEVR
* To run all the scripts from the origin repo (RL-NLP), run first the following command line: `export PYTHONPATH=src:${PYTHONPATH}`

#### Preprocessing the dataset questions
To preprocess the questions of the three datasets, run the scripts src/sh/preprocess_questions or the 3 following command lines (in this order): 

* `python src/preprocessing/preprocess_questions.py -data_path "data/CLEVR_v1.0/questions/CLEVR_train_questions.json" \`
`-out_vocab_path "data/vocab.json" -out_h5_path "data/train_questions.h5" -min_token_count 1`

* `python src/preprocessing/preprocess_questions.py -data_path "data/CLEVR_v1.0/questions/CLEVR_val_questions.json" \`
`-out_vocab_path "data/vocab.json" -out_h5_path "data/val_questions.h5" -min_token_count 1`

* `python src/preprocessing/preprocess_questions.py -data_path "data/CLEVR_v1.0/questions/CLEVR_test_questions.json" \`
`-out_vocab_path "data/vocab.json" -out_h5_path "data/test_questions.h5" -min_token_count 1`

#### Extracting the image features
To extract the image features, run the script src/sh/extract_features.py or the 3 following command lines (batch size arg must be tuned depending on memory availability): 
* `python src/preprocessing/extract_features.py \`
  `--input_image_dir data/CLEVR_v1.0/images/train \`
  `--output_h5_file data/train_features.h5 --batch_size 128`

* `python src/preprocessing/extract_features.py \`
  `--input_image_dir data/CLEVR_v1.0/images/val \`
  `--output_h5_file data/val_features.h5 --batch_size 128`

* `python src/preprocessing/extract_features.py \`
  `--input_image_dir data/CLEVR_v1.0/images/test \`
  `--output_h5_file data/test_features.h5 --batch_size 128`

### VQA
1. First, extract the vocab:
* `python src/data_provider/vqa_dataset.py -data_path "data/vqa-v2" -features_path "data/vqa-v2/coco_trainval.lmdb"
> This creates a file "vocab.json" with the vocab.
2. Once the vocab is extracted, you can create the preprocessed pkl file for each dataset:
* `python src/data_provider/vqa_dataset.py -data_path "data/vqa-v2" -features_path "data/vqa-v2/coco_trainval.lmdb" -vocab_path "data/vqa-v2/cache/vocab.json" -split "train" -test 1

## Training the models 
#### Link to the pre-trained models 
1. Language Model .pt file [here](https://drive.google.com/drive/folders/1zRT4EF8xNmilzZMYysyhCj73oQKvBLsX?usp=sharing). 
2. Levenshtein Task: 
  * Pretrained Policy .pt file (word_emb_size = 32, hidden_size = 64) [here](https://drive.google.com/file/d/1m_pXVQwQ41jgDUwuBvRHJ1U-GLqKRd3N/view?usp=sharing). 
3. VQA task: 
  * Pretrained VQA model (FiLM version [here](https://drive.google.com/file/d/15HiUyfcXcJyGdZEs-knb9EQEFGfyg4cj/view?usp=sharing). 
  * Pretrained Policy [here](https://drive.google.com/file/d/1m_pXVQwQ41jgDUwuBvRHJ1U-GLqKRd3N/view?usp=sharing)
### Training the Language Model on the Dataset of Questions
#### CLEVR
`python src/train/launch_train.py -task "lm" -dataset "clevr" -model "lstm" -num_layers 1 -emb_size 512  \`
`-hidden_size 512 -p_drop 0.1 -lr 0.001 -data_path "data" \`
`-out_path "output" -bs 512 -ep 20 -num_workers 6`
#### VQA
`python src/train/launch_train.py -task "lm" -dataset "vqa" -model "lstm" -num_layers 1 -emb_size 512  \`
`-hidden_size 512 -p_drop 0.1 -lr 0.001 -data_path "data/vqa-v2" -features_path "data/vqa-v2/coco_trainval.lmdb" \`
`-out_path "output" -bs 512 -ep 50 -num_workers 6`

### Pre-training of the Policy with Supervised Learning 
#### CLEVR
`python src/train/launch_train.py -task "policy" -dataset "clevr" -data_path "data" -out_path "output/policy_pre_training" -word_emb_size 32 -hidden_size 64 \
 -bs 512 -ep 50 -num_workers 0 -max_samples 21`  
 N.B: When training only on a CPU, the max_samples args is required to train only on a subset of the dataset. 
 
### Training the RL Agent 
* See examples in src/scripts/sh.
* The folder "debug" allows to run small experiments on each of the algo for the 2 CLEVR tasks (Levenshtein & VQA rewards). 

#### logging on tensorboard to display results: 
* `cd output/2000_img_len_20"`
* `tensorboard --logdir=experiments/train`
