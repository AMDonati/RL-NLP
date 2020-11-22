# https://towardsdatascience.com/perplexity-intuition-and-derivation-105dd481c8f3
# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
import json
import math
import os
import datetime
import torch
from torch.utils.data import DataLoader
from train.train_functions import train_one_epoch_policy, train_one_epoch, train_one_epoch_vqa, evaluate_vqa, evaluate, evaluate_policy
from utils.utils_train import create_logger, write_to_csv


class SLAlgo:
    def __init__(self, model, train_dataset, val_dataset, test_dataset, args):
        self.model = model
        self.dataset_name = args.dataset
        self.train_dataset, self.val_dataset, self.test_dataset = train_dataset, val_dataset, test_dataset
        self.train_generator = DataLoader(dataset=train_dataset, batch_size=args.bs, drop_last=True,
                                          num_workers=args.num_workers)
        self.val_generator = DataLoader(dataset=val_dataset, batch_size=args.bs, drop_last=True,
                                        num_workers=args.num_workers)
        self.batch_size = args.bs
        self.lr = args.lr
        self.optimizer = torch.optim.Adam(params=model.parameters(), lr=self.lr)
        PAD_IDX = train_dataset.vocab_questions["<PAD>"]
        self.criterion = torch.nn.NLLLoss(ignore_index=PAD_IDX)
        self.EPOCHS = args.ep
        self.grad_clip = args.grad_clip
        self.print_interval = args.print_interval
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.create_out_path(args)
        self.train_function, self.eval_function = self.get_algo_functions(args)
        self.task = args.task
        self.check_batch()

    def create_out_path(self, args):
        out_path = '{}_{}_{}_layers_{}_emb_{}_hidden_{}_pdrop_{}_gradclip_{}_bs_{}_lr_{}'.format(args.dataset, args.task, args.model,
                                                                                              args.num_layers,
                                                                                              args.emb_size,
                                                                                              args.hidden_size,
                                                                                              args.p_drop,
                                                                                              args.grad_clip,
                                                                                              args.bs, args.lr)
        if args.task == 'policy':
            out_path = out_path + '_cond-answer_{}'.format(args.condition_answer)
        self.out_path = os.path.join(args.out_path, out_path, "{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)
        out_file_log = os.path.join(self.out_path, 'training_log.log')
        self.logger = create_logger(out_file_log)
        self.out_csv = os.path.join(self.out_path, 'train_history.csv')
        self.model_path = os.path.join(self.out_path, 'model.pt')
        self.logger.info("hparams: {}".format(vars(args)))
        self.logger.info('train dataset length: {}'.format(self.train_dataset.__len__()))
        if self.dataset_name == "vqa":
            self.logger.info("number of filtered entries:{}".format(len(self.train_dataset.filtered_entries)))
        self.logger.info('number of tokens: {}'.format(self.train_dataset.len_vocab))
        self._save_hparams(args, self.out_path)

    def _save_hparams(self, args, out_path):
        dict_hparams = vars(args)
        dict_hparams = {key: str(value) for key, value in dict_hparams.items()}
        config_path = os.path.join(out_path, "config.json")
        with open(config_path, 'w') as fp:
            json.dump(dict_hparams, fp, sort_keys=True, indent=4)

    def check_batch(self):
        if self.task == 'policy':
            (temp_inp, temp_tar), answers, img = self.train_dataset.__getitem__(0)
            if isinstance(img, tuple):
                feats = img[0]
            else:
                feats = img
            self.logger.info('img features shape:{}'.format(feats.shape))
        elif self.task == 'lm':
            if self.dataset_name == "clevr":
                temp_inp, temp_tar = self.train_dataset.__getitem__(0)
            elif self.dataset_name == "vqa":
                (temp_inp, temp_tar), _, _ = self.train_dataset.__getitem__(0)

        self.logger.info("input shape: {}".format(temp_inp.shape))
        self.logger.info("target shape: {}".format(temp_tar.shape))
        self.logger.info("input:{}".format(temp_inp))
        self.logger.info("target: {}".format(temp_tar))


    def get_algo_functions(self, args):
        if args.task == "lm":
            train_function = train_one_epoch_vqa if args.dataset == "vqa" else train_one_epoch
            eval_function = evaluate_vqa if args.dataset == "vqa" else evaluate
        elif args.task == "policy":
            train_function = train_one_epoch_policy
            eval_function = evaluate_policy
        return train_function, eval_function

    def get_answers_img_features(self, dataset, index):
        if self.dataset_name == "clevr":
            img_idx = dataset.img_idxs[index]
            img_feats, _, answers = dataset.get_data_from_img_idx(img_idx)
            answer = answers[0].view(1)
        elif self.dataset_name == "vqa":
            entry = dataset.filtered_entries[index]
            img_feats, _, _ = dataset.get_img_data(entry)
            answer, _ = dataset.get_answer_data(entry)
        return img_feats, answer

    def train(self):
        self.logger.info("start training...")
        train_loss_history, train_ppl_history, val_loss_history, val_ppl_history = [], [], [], []
        best_val_loss = None
        for epoch in range(self.EPOCHS):
            self.logger.info('epoch {}/{}'.format(epoch + 1, self.EPOCHS))
            train_loss, elapsed = self.train_function(model=self.model,
                                                      train_generator=self.train_generator,
                                                      optimizer=self.optimizer,
                                                      criterion=self.criterion,
                                                      device=self.device,
                                                      grad_clip=self.grad_clip,
                                                      print_interval=self.print_interval)
            self.logger.info('train loss {:5.3f} - train perplexity {:8.3f}'.format(train_loss, math.exp(train_loss)))
            self.logger.info('time for one epoch...{:5.2f}'.format(elapsed))
            val_loss = self.eval_function(model=self.model, val_generator=self.val_generator, criterion=self.criterion,
                                          device=self.device)
            self.logger.info('val loss: {:5.3f} - val perplexity: {:8.3f}'.format(val_loss, math.exp(val_loss)))

            # saving loss and metrics information.
            train_loss_history.append(train_loss)
            train_ppl_history.append(math.exp(train_loss))
            val_loss_history.append(val_loss)
            val_ppl_history.append(math.exp(val_loss))
            self.logger.info('-' * 89)

            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(self.model_path, 'wb') as f:
                    torch.save(self.model, f)
                best_val_loss = val_loss

        self.logger.info("saving loss and metrics information...")
        hist_keys = ['train_loss', 'train_ppl', 'val_loss', 'val_ppl']
        hist_dict = dict(zip(hist_keys, [train_loss_history, train_ppl_history, val_loss_history, val_ppl_history]))
        write_to_csv(self.out_csv, hist_dict)

    def _generate_text(self, input, temperatures=[None, 0.5, 1, 2], words=50, img_feats=None, index_img=None, answer=None):
        for temp in temperatures:
            self.logger.info("generating text with temperature: {}".format(temp))
            answer_ = answer[0].cpu().item() if answer is not None else answer
            out_file_generate = os.path.join(self.out_path, 'generate_words_temp_{}_img_{}_answer_{}.txt'.format(temp, index_img, answer_))
            with open(out_file_generate, 'w') as f:
                with torch.no_grad():
                    for i in range(words):
                        if img_feats is None:
                            output, _ = self.model(input)  # output (1, num_tokens)
                        else:
                            output, _ = self.model(state_text=input, state_img=img_feats,
                                                   state_answer=answer)  # output = logits (1, num_tokens)
                        if temp is not None:
                            word_weights = output.squeeze().div(temp).exp()  # (exp(1/temp * log_sofmax)) = (p_i^(1/T))
                            word_weights = word_weights / word_weights.sum(dim=-1).cpu()
                            word_idx = torch.multinomial(word_weights, num_samples=1)[0]  # [0] to have a scalar tensor.
                        else:
                            word_idx = output.squeeze().argmax()
                        input.fill_(word_idx)
                        if self.task == "lm" and self.dataset_name == "clevr":
                            word = self.val_dataset.idx2word([word_idx.item()])
                        else:
                            word = self.val_dataset.question_tokenizer.decode([word_idx.item()])

                        f.write(word + ('\n' if i % 20 == 19 else ' '))
                        if i % 10 == 0:
                            print('| Generated {}/{} words'.format(i, words))
                    f.close()

    def generate_text(self, temperatures=[None, 0.5, 1, 2], words=50):
        input = self.test_dataset.vocab_questions["<SOS>"]
        input = torch.LongTensor([input]).view(1, 1).to(self.device)
        if self.task == "lm":
            self._generate_text(input, temperatures=temperatures, words=words)
        elif self.task == "policy":
            indexes = list(range(3))
            for index in indexes:
                print("Generating text condtioned on img: {}".format(index))
                img_feats, answer = self.get_answers_img_features(dataset=self.val_dataset, index=index)
                img_feats = img_feats.unsqueeze(0)
                self._generate_text(input, temperatures=temperatures, words=words, img_feats=img_feats, index_img=index, answer=answer)
