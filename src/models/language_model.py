import torch.nn.functional as F
import torch

class LanguageModel:
    def __init__(self, pretrained_lm, clevr_dataset):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.language_model = pretrained_lm.to(self.device)
        self.dataset = clevr_dataset

    def forward(self, state):
        pass


class ClevrLanguageModel(LanguageModel):
    def __init__(self, pretrained_lm, clevr_dataset, tokenizer=None):
        LanguageModel.__init__(self, pretrained_lm, clevr_dataset)
        self.name = "trained_on_dataset"

    def forward(self, state_text):
        seq_len = state_text.size(1)
        log_probas, logits = self.language_model(state_text.to(self.device))
        logits = logits.view(len(state_text), seq_len, -1)
        logits = logits[:, -1, :]
        log_probas = log_probas.view(len(state_text), seq_len, -1)
        log_probas = log_probas[:, -1, :]
        return log_probas, logits, log_probas


class GenericLanguageModel(LanguageModel):
    def __init__(self, pretrained_lm, clevr_dataset, tokenizer=None):
        LanguageModel.__init__(self, pretrained_lm, clevr_dataset)
        self.tokenizer = tokenizer
        self.name = "generic"
        self.clevr_to_lm_trad = {value: self.tokenizer.encoder[key] for
                                 key, value in self.dataset.vocab_questions.items() if
                                 key in self.tokenizer.encoder.keys()}
        self.bos_token = self.tokenizer.bos_token
        self.bos_token = "."

    def forward(self, state_text):
        #text = self.tokenizer.bos_token + " " + self.dataset.idx2word(state_text.cpu().numpy().ravel(),
                                                                      #stop_at_end=True)
        text = self.tokenizer.bos_token + self.dataset.idx2word(state_text.cpu().numpy().ravel(),
                                                                      stop_at_end=True)
        input_ids = self.tokenizer.encode(text, return_tensors="pt")
        origin_logits_lm = self.language_model(input_ids.to(self.device))[0][:, -1, :]
        origin_log_probs_lm = F.log_softmax(origin_logits_lm, dim=-1)
        logits = (-torch.ones(self.dataset.len_vocab) * 1e32).to(self.device)
        logits[list(self.clevr_to_lm_trad.keys())] = origin_logits_lm[0][list(self.clevr_to_lm_trad.values())]
        logits = logits.unsqueeze(dim=0)
        log_probas = F.log_softmax(logits, dim=-1)
        return log_probas, logits, origin_log_probs_lm

if __name__ == '__main__':
    from transformers import AutoModelWithLMHead, AutoTokenizer, BertTokenizer
    from data_provider.vqa_dataset import *
    print("test of generic language model...")
    vqa_data_path = '../../data/vqa-v2'
    lm_model = AutoModelWithLMHead.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    reward_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    features_h5path = "../../data/vqa-v2/reduced_coco_train.lmdb"
    images_feature_reader = ImageFeaturesH5Reader(features_h5path, False)
    SPECIAL_TOKENS = {
        '<PAD>': 0,
        '<SOS>': 1,
        '<EOS>': 2,
        '<UNK>': 3,
    }
    vqa_dataset = VQADataset(task="1_gpt", split="minval", dataroot=vqa_data_path, lm_tokenizer=tokenizer,
                             image_features_reader=images_feature_reader,
                             reward_tokenizer=reward_tokenizer, special_tokens=SPECIAL_TOKENS, clean_datasets=True,
                             max_seq_length=16, num_images=20)
    pretrained_lm = GenericLanguageModel(pretrained_lm=lm_model, clevr_dataset=vqa_dataset,
                                         tokenizer=tokenizer)
    print("length intersection vocab", len(pretrained_lm.clevr_to_lm_trad))
    print("total length vocab", vqa_dataset.len_vocab) # should be intersection vocab - 4 (special tokens).

