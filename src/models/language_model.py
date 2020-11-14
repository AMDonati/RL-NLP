import torch.nn.functional as F
import torch
import numpy as np

class LanguageModel:
    def __init__(self, pretrained_lm, dataset, tokenizer=None, prefix_tokenizer=""):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer
        self.language_model = pretrained_lm.to(self.device)
        self.dataset = dataset
        self.prefix_tokenizer = prefix_tokenizer
        self.dataset_to_lm_trad = {value: self.tokenizer.encode(text=prefix_tokenizer + key)[0] for
                                   key, value in self.dataset.vocab_questions.items() if
                                   len(self.tokenizer.encode(text=prefix_tokenizer + key)) == 1}

    def forward(self, state):
        pass

    def encode(self, **kwargs):
        return self.tokenizer.encode(**kwargs)

    def decode(self, **kwargs):
        return self.tokenizer.decode(**kwargs)


class ClevrLanguageModel(LanguageModel):
    def __init__(self, pretrained_lm, dataset, tokenizer=None):
        LanguageModel.__init__(self, pretrained_lm, dataset, tokenizer)

    def forward(self, state_text):
        seq_len = state_text.size(1)
        log_probas, logits = self.language_model(state_text.to(self.device))
        logits = logits.view(len(state_text), seq_len, -1)
        logits = logits[:, -1, :]
        last_log_probas = log_probas.view(len(state_text), seq_len, -1)
        last_log_probas = last_log_probas[:, -1, :]
        return last_log_probas, logits, log_probas.view(len(state_text), seq_len, -1)


class GenericLanguageModel(LanguageModel):
    def __init__(self, pretrained_lm, dataset, tokenizer=None, prefix_tokenizer=" ", init_text=None, custom_init=0):
        LanguageModel.__init__(self, pretrained_lm, dataset, tokenizer, prefix_tokenizer=prefix_tokenizer)
        self.tokenizer = tokenizer
        self.name = "generic"
        self.dataset_to_lm_trad = {value: self.tokenizer.encoder[key] for
                                 key, value in self.dataset.vocab_questions.items() if
                                 key in self.tokenizer.encoder.keys()}

        self.bos_token = self.tokenizer.bos_token
        self.get_init_text(init_text, custom_init)

    def get_init_text(self, init_text, custom_init, seed=1234):
        if custom_init > 0:
            np.random.seed(seed)
            idxs = np.random.randint(0,len(self.dataset.remaining_entries),size=custom_init)
            samples = np.array(self.dataset.remaining_entries)[list(set(idxs))]
            example_questions = [s["question"] for s in samples]
            example_questions_text = " ".join(example_questions)
            self.init_text = init_text + example_questions_text
            self.init_text_short = init_text + " ".join(example_questions[:min(len(example_questions),2)]) + "..."
        else:
            self.init_text = init_text
            self.init_text_short = init_text
        if self.init_text is not None:
            print("init text for GPT-2 pre-conditioning...", self.init_text)


    def forward(self, state_text):
        text = self.dataset.question_tokenizer.decode(state_text.cpu().numpy().ravel(), stop_at_end=True)
        if self.init_text is not None:
            text = self.init_text + text
        if text == "":
            text = self.tokenizer.bos_token
        input_ids = self.tokenizer.encode(text, return_tensors="pt")
        origin_logits_lm = self.language_model(input_ids.to(self.device))[0]
        origin_log_probs_lm = F.log_softmax(origin_logits_lm, dim=-1)
        logits = (-torch.ones(len(self.dataset.vocab_questions)) * 1e32).to(self.device)
        logits[list(self.dataset_to_lm_trad.keys())] = origin_logits_lm[:, -1, :][0][
            list(self.dataset_to_lm_trad.values())]
        logits = logits.unsqueeze(dim=0)
        log_probas = F.log_softmax(logits, dim=-1)
        return log_probas, logits, origin_log_probs_lm.view(input_ids.size(0), input_ids.size(1), -1)


if __name__ == '__main__':
    from transformers import AutoModelWithLMHead, GPT2Tokenizer, BertTokenizer
    from data_provider.vqa_dataset import *
    from data_provider.vqa_tokenizer import VQATokenizer
    print("test of generic language model...")
    vqa_data_path = '../../data/vqa-v2'

    init_string = "The question is:"
    init_string = "Please generate a question. Here are a few examples:"

    lm_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    reward_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    features_h5path = os.path.join(vqa_data_path, "coco_trainval.lmdb")
    images_feature_reader = ImageFeaturesH5Reader(features_h5path, False)

    question_tokenizer = VQATokenizer(lm_tokenizer=lm_tokenizer)
    train_dataset = VQADataset(split="minval", dataroot=vqa_data_path,
                               question_tokenizer=question_tokenizer,
                               image_features_reader=images_feature_reader,
                               reward_tokenizer=reward_tokenizer, clean_datasets=True, max_seq_length=23,
                               num_images=20, vocab_path=os.path.join(vqa_data_path, 'cache/vocab.json'),
                               filter_entries=True)
    lm_model = AutoModelWithLMHead.from_pretrained("gpt2")
    pretrained_lm = GenericLanguageModel(pretrained_lm=lm_model, dataset=train_dataset,
                                         tokenizer=lm_tokenizer, init_text=init_string, custom_init=2)

    print("Test of Language Model forward pass...")
    state_text = torch.tensor([[4,5,6]])
    log_probs, logits, _ = pretrained_lm.forward(state_text)
    print("log_probs", log_probs.shape)

    #print("length intersection vocab", len(pretrained_lm.clevr_to_lm_trad))
    #print("total length vocab", train_dataset.len_vocab)  # should be intersection vocab - 4 (special tokens).