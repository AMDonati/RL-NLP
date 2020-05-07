'''Generate words from the pre-trained language model
Generare top-k words given a specific sequence length.
Inspired from: https://github.com/pytorch/examples/blob/master/word_language_model/generate.py
'''
import argparse
import os
import torch
from data_provider.QuestionsDataset import QuestionsDataset
from torch.utils.data import DataLoader
from utils.utils_train import create_logger

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

def eval_overconfidence(model, test_loader, device, threshold=0.5):
    N = test_loader.batch_size
    total, correct, over_conf = 0., 0., 0.
    model.eval()
    hidden = model.init_hidden(batch_size=N)
    with torch.no_grad():
      for inputs, targets in test_loader:
        seq_len = inputs.size(1)
        inputs, targets = inputs.t().long().to(device), targets.t().long().to(device)
        output, _ = model(inputs, hidden)  # (N*seq_len, num_tokens)
        output = output.view(seq_len, N, -1)  # (N, seq_len, num_tokens)
        log_prob, word_preds = torch.max(output.data, -1)
        total += targets.size(0) * targets.size(1)
        bool_correct = word_preds == targets
        correct += bool_correct.sum().item()
        over_conf += ((log_prob.exp() * bool_correct) >= threshold).sum().item()

    return correct / total, over_conf / total

if __name__ == '__main__':

  parser = argparse.ArgumentParser()

  parser.add_argument("-data_path", type=str, required=True, help="path for data")
  parser.add_argument("-model_path", type=str, required=True, help="path for saved model")
  parser.add_argument("-out_path", type=str, required=True, default='../../output')
  parser.add_argument("-words", type=int, default=100, help="num words to generate")
  #parser.add_argument("-seed", type=int, default=123, help="seed for reproducibility")
  parser.add_argument("-oc_th", type=float, default=0.5, help="proba threshold for overconfidence function")
  parser.add_argument('-temperature', type=float, default=1.0, help='temperature - higher will increase diversity')
  parser.add_argument('-num_workers', type=int, default=0, help="num workers for DataLoader")

  args = parser.parse_args()

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  with open(args.model_path, 'rb') as f:
    model = torch.load(f, map_location=device).to(device)
  model.eval()
  model.bidirectional = False
  test_dataset = QuestionsDataset(h5_questions_path=os.path.join(args.data_path, 'test_questions.h5'),
                                  vocab_path=os.path.join(args.data_path, 'vocab.json'))
  num_tokens = test_dataset.vocab_len

  #if not torch.cuda.is_available():
    #print("selecting a subset of the whole test dataset made of the first 10,000 samples")
    #test_dataset = Subset(QuestionsDataset, indices=[i for i in range(10000)])
  test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), drop_last=True, num_workers=args.num_workers)


  out_file = os.path.join(args.out_path, 'generate_words_temp_{}.txt'.format(args.temperature))
  out_file_log = os.path.join(args.out_path, 'eval_log.log')
  logger = create_logger(out_file_log)
  log_interval = int(args.words / 10)

  ###############################################################################
  # Evaluate overconfidence on test set
  #############################################################################

  accuracy, over_confidence = eval_overconfidence(model=model, test_loader=test_loader, device=device)
  logger.info('test accuracy:{}'.format(accuracy))
  logger.info('overconfidence rate for thr = {}: {}'.format(0.5, over_confidence))

  ###############################################################################
  # generate words
  ###############################################################################

  # initialize hidden state
  hidden = model.init_hidden(batch_size=1)
  input = torch.randint(low=0, high=num_tokens, size=(1,1), dtype=torch.long).to(device)

  with open(out_file, 'w') as f:
    input_word = test_dataset.idx2word([input[0].item()], delim='')
    f.write(input_word + '\n')
    with torch.no_grad():
      for i in range(args.words):
        output, hidden = model(input, hidden) # output (1, num_tokens)
        if args.temperature is not None:
          word_weights = output.squeeze().div(args.temperature).exp().cpu()
          word_idx = torch.multinomial(word_weights, num_samples=1)[0] # [0] to have a scalar tensor.
        else:
          word_idx = output.squeeze().argmax()
        input.fill_(word_idx)

        word = test_dataset.idx2word(seq_idx=[word_idx.item()], delim='')

        f.write(word + ('\n' if i % 20 == 19 else ' '))

        if i % log_interval == 0:
          print('| Generated {}/{} words'.format(i, args.words))

  #############################################################################################
  # look at the top-k words for a given sequence of input words.
###############################################################################################

