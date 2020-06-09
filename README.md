# RL-NLP

### CLVR Datasets: 
* [CLVR original dataset page](https://cs.stanford.edu/people/jcjohns/clevr/)
* [CLVR preprocessing github](https://github.com/facebookresearch/clevr-iep/blob/master/TRAINING.md)
* [CLOSURE dataset github page](https://github.com/rizar/CLOSURE)


#### Example of use 
`cd src/scripts/`

`run.py -data_path your_data_path -out_path your_output_path -max_len 7 -logger_level DEBUG -num_episodes_train 4000 -log_interval 1 -reward "levenshtein_" -model lstm_word -update_timestep 48 -K_epochs 10 -entropy_coeff 0.01 -eps_clip 0.02 -pretrain 0 -debug 1 -num_episodes_test 100 -agent PPO`
