
class VQATokenizer:
    def __init__(self, lm_tokenizer):
        self.allow_unk = False
        self.lm_tokenizer = lm_tokenizer
        self.special_tokens = {
    '<PAD>': 0,
    '<SOS>': 1,
    '<EOS>': 2,
    '<UNK>': 3,
}

    def decode(self, text, ignored=['<PAD>'], decode_answers=True, stop_at_end=True):
        lm_question_idx = self.translate_for_lm(text)
        question_decoded = self.lm_tokenizer.decode(lm_question_idx)
        return question_decoded

    def encode(self, text):
        lm_question_idx = self.lm_tokenizer.encode(text)
        question_idx = [self.lm_to_dataset_trad[idx] for idx in lm_question_idx]
        return question_idx

    def translate_for_lm(self, question_idx):
        lm_question_idx = [self.dataset_to_lm_trad[idx] for idx in question_idx if idx not in self.special_tokens.values()]  # question_idx should not include special tokens.
        return lm_question_idx

    def set_vocab(self, vocab):
        self.vocab = vocab
        self.idx_to_token = dict(zip(list(vocab.values()), list(vocab.keys())))
        self.dataset_to_lm_trad = {val: self.lm_tokenizer.encoder[key] for key, val in self.vocab.items() if
                                   key in self.lm_tokenizer.encoder.keys()}
        self.dataset_to_lm_trad[self.special_tokens['<PAD>']] = self.lm_tokenizer.pad_token_id
        self.dataset_to_lm_trad[self.special_tokens['<SOS>']] = self.lm_tokenizer.bos_token_id
        self.dataset_to_lm_trad[self.special_tokens['<EOS>']] = self.lm_tokenizer.eos_token_id
        self.dataset_to_lm_trad[self.special_tokens['<UNK>']] = self.lm_tokenizer.unk_token_id
        self.lm_to_dataset_trad = {v: k for k, v in self.dataset_to_lm_trad.items()}
        assert len(self.dataset_to_lm_trad)==len(vocab), "error when setting vocab"