import os
import torch
import json
from preprocessing.text_functions import decode

def generate_top_k_words(model, test_dataset, device, seq_len, samples, k):
  #TODO: simplify and correct this function.
  test_q, _ = test_dataset.get_questions()
  vocab = test_dataset.get_vocab()
  idx_to_token = dict(zip(list(vocab.values()), list(vocab.keys())))
  test_sample = test_q[:seq_len, samples].long().to(device) # (S, num_samples) #TODO: add a to device here.

  decoded_input_text = []
  for index in range(len(samples)):
    text_i = list(test_sample[:, index].data.numpy())  # (S)
    decode_seq = decode(seq_idx=text_i, idx_to_token=idx_to_token, stop_at_end=False, delim=' ')
    decoded_input_text.append(decode_seq)
  # forward pass:
  model.eval()
  hidden = model.init_hidden(len(samples))
  with torch.no_grad():
    output, hidden = model(test_sample, hidden)  # output (S*num_samples, num_words)
    output = output.view(-1, len(samples), output.size(-1))  # (S, num_samples, num_words)
    log_pred = output[-1, :, :]  # taking last pred of the seq
    top_k, top_i = torch.topk(log_pred, k, dim=-1)  # (num_samples, k)
    top_i = top_i.data.numpy()

  list_top_words = []
  for index in range(len(samples)):
    seq_idx = list(top_i[index, :])
    token_idx = decode(seq_idx=seq_idx, idx_to_token=idx_to_token, stop_at_end=False, delim=',')
    list_top_words.append(token_idx)

  dict_top_words = dict(zip(decoded_input_text, list_top_words))
  return dict_top_words

if __name__ == '__main__':


  sample_index = list(np.random.randint(0, len(test_dataset), size=20))
  sample_top_words = generate_top_k_words(test_dataset=test_dataset, seq_len=10, samples=sample_index, k=10)
  print('top words', sample_top_words)
  # sample_top_words['sampled index'] = sample_index
  json_top_words = os.path.join(args.out_path, 'sampled_top_words.json')
  with open(json_top_words, mode='w') as f:
    json.dump(sample_top_words, f)
