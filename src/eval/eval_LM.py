'''Generate words from the pre-trained language model
Generare top-k words given a specific sequence length.
Inspired from: https://github.com/pytorch/examples/blob/master/word_language_model/generate.py
'''


import argparse
import os
import torch

#  trick for boolean parser args.
def str2bool(v):
  if isinstance(v, bool):
    return v
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':

  parser = argparse.ArgumentParser()

  parser.add_argument("-data_path", type=str, required=True, help="path for data")
  parser.add_argument("-model_path", type=str, required=True, help="path for saved model")
  parser.add_argument("-out_path", type=str, required=True, default='../../output')
  parser.add_argument("-words", type=int, default=100, help="num words to generate")
  parser.add_argument("-seed", type=int, default=123, help="seed for reproducibility")
  parser.add_argument('-num_workers', type=int, required=True, default=0, help="num workers for DataLoader")
  parser.add_argument('-cuda', type=str2bool, required=True, default=False, help='use cuda')
  parser.add_argument('-temperature', type=float, default=1.0, help='temperature - higher will increase diversity')

  args = parser.parse_args()

  device = torch.device("cpu" if not args.cuda else "cuda")


  with open(args.model_path, 'rb') as f:
    model = torch.load(f).to(device)
  model.eval()

  #TODO: replace this by the refactorized QuestionsDataset.
  corpus = data.Corpus(args.data)
  num_tokens = len(corpus.dictionary)

  out_file = os.path.join(args.output_path, 'generate_words.txt')
  log_interval = int(args.words / 10)


  ###############################################################################
  # generate words
  ###############################################################################

  # initialize hidden state
  hidden = model.init_hidden(batch_size=1)
  input = torch.randint(low=0, high=num_tokens, size=(1,1), dtype=torch.long).to(device)

  with open(out_file, 'w') as f:
    with torch.no_grad():
      for i in range(args.words):
        output, hidden = model(input, hidden) # output (1, num_tokens)
        word_weights = output.squeeze().div(args.temperature).exp().cpu() #TODO: add the option of no temperature sampling...
        word_idx = torch.multinomial(word_weights, num_samples=1)[0] #TODO: understand the 0.
        input.fill_(word_idx)

        word = corpus.dictionary.idx2word[word_idx] #TODO change here.

        f.write(word + ('\n' if i % 20 == 19 else ' '))

        if i % log_interval == 0:
          print('| Generated {}/{} words'.format(i, args.words))


