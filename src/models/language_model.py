import torch
import torch.nn.functional as F
from transformers import BertGenerationConfig, BertGenerationEncoder, BertGenerationDecoder, BertGenerationTokenizer


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
    def __init__(self, pretrained_lm, clevr_dataset, tokenizer=None):
        LanguageModel.__init__(self, pretrained_lm, clevr_dataset, tokenizer)

    def forward(self, state_text):
        seq_len = state_text.size(1)
        log_probas, logits = self.language_model(state_text.to(self.device))
        logits = logits.view(len(state_text), seq_len, -1)
        logits = logits[:, -1, :]
        log_probas = log_probas.view(len(state_text), seq_len, -1)
        log_probas = log_probas[:, -1, :]
        return log_probas, logits, log_probas


class GenericLanguageModel(LanguageModel):
    def __init__(self, pretrained_lm, dataset, tokenizer=None, prefix_tokenizer=" "):
        LanguageModel.__init__(self, pretrained_lm, dataset, tokenizer, prefix_tokenizer=" ")

    def forward(self, state_text):
        text = self.tokenizer.bos_token + " " + self.dataset.question_tokenizer.decode(
            text=state_text.cpu().numpy().ravel(), stop_at_end=True)
        input_ids = self.tokenizer.encode(text, return_tensors="pt")
        origin_logits_lm = self.language_model(input_ids.to(self.device))[0][:, -1, :]
        origin_log_probs_lm = F.log_softmax(origin_logits_lm, dim=-1)
        logits = (-torch.ones(len(self.dataset.vocab_questions)) * 1e32).to(self.device)
        logits[list(self.dataset_to_lm_trad.keys())] = origin_logits_lm[0][list(self.dataset_to_lm_trad.values())]
        logits = logits.unsqueeze(dim=0)
        log_probas = F.log_softmax(logits, dim=-1)
        return log_probas, logits, origin_log_probs_lm


class BertGeneration(LanguageModel):
    '''
    https://huggingface.co/transformers/model_doc/bertgeneration.html
    From: https://arxiv.org/pdf/1907.12461.pdf
    '''

    def __init__(self, pretrained_lm, clevr_dataset, tokenizer):
        # ENCODER PART:
        tokenizer = BertGenerationTokenizer.from_pretrained('google/bert_for_seq_generation_L-24_bbc_encoder')
        model = BertGenerationEncoder.from_pretrained('google/bert_for_seq_generation_L-24_bbc_encoder',
                                                      return_dict=True)

        inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state

        # DECODER PART:
        tokenizer = BertGenerationTokenizer.from_pretrained('google/bert_for_seq_generation_L-24_bbc_encoder')
        config = BertGenerationConfig.from_pretrained("google/bert_for_seq_generation_L-24_bbc_encoder")
        config.is_decoder = True
        model = BertGenerationDecoder.from_pretrained('google/bert_for_seq_generation_L-24_bbc_encoder',
                                                      config=config, return_dict=True)
        inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        outputs = model(**inputs)
        prediction_logits = outputs.logits
