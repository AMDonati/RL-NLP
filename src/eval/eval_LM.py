'''Generate words from the pre-trained language model
Generare top-k words given a specific sequence length.
Inspired from: https://github.com/pytorch/examples/blob/master/word_language_model/generate.py
'''
import argparse
import os, json
import torch
import numpy as np
from data_provider.QuestionsDataset import QuestionsDataset
from torch.utils.data import DataLoader
from utils.utils_train import create_logger
from eval.eval_functions import generate_top_k_words

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
    total, correct, over_conf, over_conf_correct = 0., 0., 0., 0.
    model.eval()
    hidden = model.init_hidden(batch_size=N)
    with torch.no_grad():
      for inputs, targets in test_loader:
        seq_len = inputs.size(1)
        inputs, targets = inputs.t().long().to(device), targets.t().long().to(device)
        output, _ = model(inputs, hidden)  # (N*seq_len, num_tokens)
        output = output.view(seq_len, N, -1)  # (N, seq_len, num_tokens)
        log_prob, word_preds = torch.max(output.data, -1) # (N, seq_len)
        total += targets.size(0) * targets.size(1)
        bool_correct = word_preds == targets
        correct_words = word_preds * bool_correct
        correct += bool_correct.sum().item()
        over_conf += (log_prob.exp() >= threshold).sum().item()
        over_conf_correct += ((log_prob.exp() * bool_correct) >= threshold).sum().item()

    return correct / total, over_conf / total, over_conf_correct / correct, correct_words

if __name__ == '__main__':

  parser = argparse.ArgumentParser()

  parser.add_argument("-data_path", type=str, required=True, help="path for data")
  parser.add_argument("-model_path", type=str, required=True, help="path for saved model")
  parser.add_argument("-out_path", type=str, required=True, default='../../output')
  parser.add_argument("-words", type=int, default=200, help="num words to generate")
  #parser.add_argument("-seed", type=int, default=123, help="seed for reproducibility")
  parser.add_argument("-oc_th", type=float, default=0.1, help="proba threshold for overconfidence function")
  parser.add_argument('-temperature', type=float, help='temperature - higher will increase diversity')
  parser.add_argument("-top_k", type=int, default=10, help="num of top-k words to generate from input sequence.")
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

  test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), drop_last=True, num_workers=args.num_workers)

  out_file = os.path.join(args.out_path, 'generate_words_temp_{}.txt'.format(args.temperature))
  out_file_top_k_words = os.path.join(args.out_path, 'generate_top_k_words_k_{}.json'.format(args.top_k))
  out_file_log = os.path.join(args.out_path, 'eval_log.log')
  logger = create_logger(out_file_log)
  log_interval = int(args.words / 10)

  ###############################################################################
  # Evaluate overconfidence on test set
  #############################################################################
  logger.info('evaluating accuracy, overconfidence, and correct words on first {} samples of test dataset...'.format(len(test_dataset)))
  accuracy, over_confidence, over_confidence_correct, correct_words = eval_overconfidence(model=model,
                                                                                          test_loader=test_loader,
                                                                                          device=device,
                                                                                          threshold=args.oc_th)
  correct_words = correct_words.view(-1).data.numpy()
  idx_correct = np.where(correct_words !=0)
  correct_words = list(correct_words[list(idx_correct)])
  #correct_words = correct_words.remove(0)
  unique_correct_words = list(set(correct_words))
  decoded_correct_words = test_dataset.idx2word(unique_correct_words, delim=',')
  logger.info('test accuracy:{}'.format(accuracy))
  logger.info('overconfidence rate for thr = {}: {}'.format(args.oc_th, over_confidence))
  logger.info('overconfidence rate for correct preds for thr= {}: {}'.format(args.oc_th, over_confidence_correct))
  logger.info('list of correct words predicted: {}'.format(decoded_correct_words))

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
  seq_len = 5
  sample_index = list(np.random.randint(0, len(test_dataset), size=50))
  logger.info("looking at top {} words for 50 samples of the test dataset".format(args.top_k))
  dict_top_words = generate_top_k_words(model=model,
                                        test_dataset=test_dataset,
                                        samples=sample_index,
                                        device=device,
                                        seq_len=seq_len,
                                        k=args.top_k)
  str_index = ','.join([str(i) for i in sample_index])
  dict_top_words = {'sampled_index': str_index,
                    'top_k_words': dict_top_words}
  with open(out_file_top_k_words, mode='w') as f:
    json.dump(dict_top_words, f, indent=4)
  logger.info("done with saving top-k words")

  # with open(out_file_top_k_words, 'w') as f:
  #   f.write('sampled indexes:' + sample_index + '\n')
  #   for (input, top_k) in zip(list(dict_top_words.keys()), list(dict_top_words.values())):
  #     f.write(input + ' ' + ':' + ' ' + top_k + '\n')
