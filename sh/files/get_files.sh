#!/usr/bin/env bash
DATA_ID=1AVZXRzmKBxVH6Ul9ZviSWPVdj_kwU3yX
OUTPUT_ID=13K2sv7K2zK4MeFChH2-w_MlOnKLfwxh7
FULL_CLEVR=$1
FULL_VQAV2=$2
BATCH_SIZE=10

gdown --id $DATA_ID --output data/data.zip
unzip data/data.zip -d data/
rm data/data.zip

gdown --id $OUTPUT_ID --output output/output.zip
unzip output/output.zip -d output/
rm output/output.zip

# getting the full datasets
if [ $FULL_CLEVR = "full" ]
then
  rm -r data/CLEVR_v1.0
  wget https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip -O data/CLEVR_v1.0.zip
  unzip data/CLEVR_v1.0.zip -d data
  rm data/CLEVR_v1.0.zip
  BATCH_SIZE=64
fi

if [ $FULL_VQAV2 = "full" ]#TODO Test this
then
  wget https://dl.fbaipublicfiles.com/vilbert-multi-task/datasets.tar.gz
  tar xf datasets.tar.gz
  mv datasets/coco/features_100/COCO_test_resnext152_faster_rcnn_genome.lmdb data/vqa-v2/coco_test.lmdb
  mv datasets/coco/features_100/COCO_trainval_resnext152_faster_rcnn_genome.lmdb data/vqa-v2/coco_trainval.lmdb
  rm datasets.tar.gz
  rm -r datasets
fi


python src/preprocessing/preprocess_questions.py -data_path "data/CLEVR_v1.0/questions/CLEVR_train_questions.json"  -out_vocab_path "data/vocab.json" -out_h5_path "data/train_questions.h5" -min_token_count 1

python src/preprocessing/preprocess_questions.py -data_path "data/CLEVR_v1.0/questions/CLEVR_val_questions.json" -out_vocab_path "data/vocab.json" -out_h5_path "data/val_questions.h5" -min_token_count 1

python src/preprocessing/preprocess_questions.py -data_path "data/CLEVR_v1.0/questions/CLEVR_test_questions.json" -out_vocab_path "data/vocab.json" -out_h5_path "data/test_questions.h5" -min_token_count 1


python src/preprocessing/extract_features.py --input_image_dir data/CLEVR_v1.0/images/train --output_h5_file data/train_features.h5 --batch_size $BATCH_SIZE

python src/preprocessing/extract_features.py --input_image_dir data/CLEVR_v1.0/images/val --output_h5_file data/val_features.h5 --batch_size $BATCH_SIZE

python src/preprocessing/extract_features.py --input_image_dir data/CLEVR_v1.0/images/test  --output_h5_file data/test_features.h5 --batch_size $BATCH_SIZE
