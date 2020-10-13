import torch
import torch.nn.functional as F

from models.LM_networks import LSTMModel


class LanguageModel:
    def __init__(self, pretrained_lm):
        self.language_model = pretrained_lm
        self.forward_func = self.forward_clevr if self.language_model.__class__ in [LSTMModel] else self.forward_generic

    def forward(self, state):
        return self.forward_func(state)


    def forward_clevr(self, state):
        seq_len = state.text.size(1)
        log_probas, logits = self.language_model(state.text.to(self.device))
        logits = logits.view(len(state.text), seq_len, -1)
        logits = logits[:, -1, :]
        log_probas = log_probas.view(len(state.text), seq_len, -1)
        log_probas = log_probas[:, -1, :]
        return log_probas, logits

    def forward_generic(self, state):
        text = "bos " + self.dataset.idx2word(state.text.cpu().numpy().ravel(), stop_at_end=True)
        input_ids = self.tokenizer.encode(text, return_tensors="pt")
        logits_lm = self.language_model(input_ids)[0][:, -1, :]
        logits = -torch.ones(len(self.lm_to_clevr_trad)) * 1e32
        logits[list(self.clevr_to_lm_trad.values())] = logits_lm[0][list(self.clevr_to_lm_trad.keys())]
        # filtered_next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=50, top_p=1.0)
        # probs = F.softmax(filtered_next_token_logits, dim=-1)
        # next_token = torch.multinomial(probs, num_samples=1)
        logits = logits.unsqueeze(dim=0)
        log_probas = F.log_softmax(logits, dim=-1)
        return log_probas, logits
