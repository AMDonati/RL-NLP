import os
import torch
import json
import numpy as np
from data_provider.QuestionsDataset import QuestionsDataset

def generate_top_k_words(model, test_dataset, device, seq_len, samples, k):
  test_q, _ = test_dataset.get_questions()
  test_sample = test_q[:seq_len, samples].long().to(device) # (S, num_samples)
  # forward pass:
  model.eval()
  hidden = model.init_hidden(len(samples))
  with torch.no_grad():
    output, hidden = model(test_sample, hidden)  # output (S*num_samples, num_words)
    output = output.view(-1, len(samples), output.size(-1))  # (S, num_samples, num_words)
    log_pred = output[-1, :, :]  # taking last pred of the seq
    top_k, top_i = torch.topk(log_pred, k, dim=-1)  # (num_samples, k)
    top_i = top_i.data.numpy()
  # decode input data and top k idx.
  list_top_words, decoded_input_text = [], []
  for index in range(len(samples)):
    input_idx = list(test_sample[:, index].data.numpy())
    top_k_idx = list(top_i[index, :])
    top_k_tokens = test_dataset.idx2word(seq_idx=top_k_idx, delim=',')
    input_tokens = test_dataset.idx2word(seq_idx=input_idx, delim=' ')
    list_top_words.append(top_k_tokens)
    decoded_input_text.append(input_tokens)
  dict_top_words = dict(zip(decoded_input_text, list_top_words))
  return dict_top_words

if __name__ == '__main__':

  temp_path = '../../data/CLEVR_v1.0/temp/5000_2000_samples'
  temp_dataset = QuestionsDataset(h5_questions_path=os.path.join(temp_path, 'test_questions.h5'),
                                  vocab_path=os.path.join(temp_path, 'vocab.json'))
  sample_index = list(np.random.randint(0, len(temp_dataset), size=20))
  dict_top_words = generate_top_k_words(test_dataset=temp_dataset, seq_len=5, samples=sample_index, k=10)
  print('top words', dict_top_words)
  # sample_top_words['sampled index'] = sample_index
  # json_top_words = os.path.join(args.out_path, 'sampled_top_words.json')
  with open(json_top_words, mode='w') as f:
    json.dump(sample_top_words, f)
