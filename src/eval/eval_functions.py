import os
import torch
import json
import numpy as np
from data_provider.QuestionsDataset import QuestionsDataset


def generate_top_k_words(model, test_dataset, device, seq_len, samples, k):
    test_q, _ = test_dataset.get_questions()
    test_sample = test_q[samples, :seq_len].to(device)  # (S, num_samples)
    # forward pass:
    model.eval()
    with torch.no_grad():
        output, hidden = model(test_sample)  # output (S*num_samples, num_words)
        output = output.view(len(samples), -1, output.size(-1))  # (S, num_samples, num_words)
        log_pred = output[:, -1, :]  # taking last pred of the seq
        top_k, top_i = torch.topk(log_pred, k, dim=-1)  # (num_samples, k)
        top_i = top_i.data.numpy()
    # decode input data and top k idx.
    list_top_words, decoded_input_text = [], []
    for index in range(len(samples)):
        input_idx = list(test_sample[index, :].data.numpy())
        top_k_idx = list(top_i[index, :])
        top_k_tokens = test_dataset.idx2word(seq_idx=top_k_idx, delim=',')
        input_tokens = test_dataset.idx2word(seq_idx=input_idx, delim=' ')
        list_top_words.append(top_k_tokens)
        decoded_input_text.append(input_tokens)
    dict_top_words = dict(zip(decoded_input_text, list_top_words))

    return dict_top_words


def eval_overconfidence(model, test_loader, device, threshold=[0.5]):
    N = test_loader.batch_size
    total, correct, over_conf, over_conf_correct = 0., 0., 0., 0.
    model.eval()

    with torch.no_grad():
        for inputs, targets in test_loader:
            seq_len = inputs.size(1)
            inputs, targets = inputs.to(device), targets.to(device)
            output, _ = model(inputs)  # (N*seq_len, num_tokens)
            output = output.view(N, seq_len, -1)  # (N, seq_len, num_tokens)
            log_prob, word_preds = torch.max(output.data, -1)  # (N, seq_len)
            total += targets.size(0) * targets.size(1)
            bool_correct = word_preds == targets
            correct_words = word_preds * bool_correct
            correct += bool_correct.sum().item()

        over_confs, over_confs_correct = [], []
        for thr in threshold:
            over_conf = (log_prob.exp() >= thr).sum().item()
            over_conf_correct = ((log_prob.exp() * bool_correct) >= thr).sum().item()
            over_confs.append(over_conf / total)
            over_confs_correct.append(over_conf_correct / over_conf)
        over_confs = dict(zip(threshold, over_confs))
        over_confs_correct = dict(zip(threshold, over_confs_correct))

    return correct / total, over_confs, over_confs_correct, correct_words


if __name__ == '__main__':
    temp_path = '../../data/CLEVR_v1.0/temp/5000_2000_samples'
    temp_dataset = QuestionsDataset(h5_questions_path=os.path.join(temp_path, 'test_questions.h5'),
                                    vocab_path=os.path.join(temp_path, 'vocab.json'))
    sample_index = list(np.random.randint(0, len(temp_dataset), size=20))
    dict_top_words = generate_top_k_words(test_dataset=temp_dataset, seq_len=5, samples=sample_index, k=10)
    print('top words', dict_top_words)
