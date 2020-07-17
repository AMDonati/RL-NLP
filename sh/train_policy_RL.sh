python src/train/train_Policy_RL.py -out_path "output/policy_pre_training/SL_32_64/RL" \
 -data_path "data" -max_len 15 -lr 0.001 -max_samples 200 -pre_train True -model_path "output/SL_32_64/model.pt"


 # on local machine - pre_train:
 python src/train/train_Policy_RL.py -out_path "output/SL_32_64_2/RL/with_trunc" -data_path "data" -max_len 15 -lr 0.001 -pre_train True -model_path "output/SL_32_64_2/model.pt" -trunc True

 # on local machine - trunc:
 python src/train/train_Policy_RL.py -out_path "output/RL/trunc_32_64_k_10" -data_path "data" -max_len 15 -lr 0.001 -pre_train False

python src/train/train_Policy_RL.py -out_path "output/RL/trunc_32_64_k_1_img_0" -data_path "data" -max_len 15 -lr 0.001 -pre_train False