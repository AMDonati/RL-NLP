import torch
import torch.nn.functional as F


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
        self.name = "clevr"

    def forward(self, state_text):
        seq_len = state_text.size(1)
        log_probas, logits = self.language_model(state_text.to(self.device))
        logits = logits.view(len(state_text), seq_len, -1)
        logits = logits[:, -1, :]
        log_probas = log_probas.view(len(state_text), seq_len, -1)
        log_probas = log_probas[:, -1, :]
        return log_probas, logits

    def get_logits_all_sequence(self, state_text):
        seq_len = state_text.size(1)
        log_probas, logits = self.language_model(state_text.to(self.device))
        logits = logits.view(len(state_text), seq_len, -1)
        log_probas = log_probas.view(len(state_text), seq_len, -1)
        return log_probas.squeeze(dim=0), logits.squeeze(dim=0)


class GenericLanguageModel(LanguageModel):
    def __init__(self, pretrained_lm, clevr_dataset, tokenizer=None):
        LanguageModel.__init__(self, pretrained_lm, clevr_dataset)
        self.tokenizer = tokenizer
        self.clevr_to_lm_trad = {value: self.tokenizer.encode(" " + key)[0] for
                                 key, value in self.dataset.vocab_questions.items() if
                                 len(self.tokenizer.encode(" " + key)) == 1}
        self.name = 'generic'

    def forward(self, state_text):
        text = self.tokenizer.bos_token+" " + self.dataset.idx2word(state_text.cpu().numpy().ravel(), stop_at_end=True)
        input_ids = self.tokenizer.encode(text, return_tensors="pt")
        logits_lm = self.language_model(input_ids.to(self.device))[0][:, -1, :]
        logits = (-torch.ones(len(self.dataset.vocab_questions)) * 1e32).to(self.device)
        logits[list(self.clevr_to_lm_trad.keys())] = logits_lm[0][list(self.clevr_to_lm_trad.values())]
        logits = logits.unsqueeze(dim=0)
        log_probas = F.log_softmax(logits, dim=-1)
        return log_probas, logits

    def get_logits_all_sequence(self, state_text):
        #TODO: to complete, to be able to get the ppl from GPT...
        text = self.tokenizer.bos_token + " " + self.dataset.idx2word(state_text.cpu().numpy().ravel(),
                                                                      stop_at_end=True)
        input_ids = self.tokenizer.encode(text, return_tensors="pt")
        logits_lm = self.language_model(input_ids)[0]
        logits = -torch.ones(len(self.dataset.vocab_questions)) * 1e32
        logits[list(self.clevr_to_lm_trad.keys())] = logits_lm[0][list(self.clevr_to_lm_trad.values())]
        logits = logits.unsqueeze(dim=0)
        log_probas = F.log_softmax(logits, dim=-1)
        return log_probas, logits
