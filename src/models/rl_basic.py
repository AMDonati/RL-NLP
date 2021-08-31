import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torchcontrib import nn as contrib_nn
from transformers import AutoModelWithLMHead, AutoTokenizer, GPT2Model

from RL_toolbox.truncation import mask_truncature, mask_inf_truncature
from models.language_model import ClevrLanguageModel
from typing import Optional


class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim, answer_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.answer_att = nn.Linear(answer_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden, answer_embedding=None):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att_sum = att1 + att2.unsqueeze(1)
        if answer_embedding != None:
            att3 = self.answer_att(answer_embedding)
            att_sum += att3
        att = self.full_att(self.relu(att_sum)).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class PolicyLSTMBatch(nn.Module):

    def __init__(self, num_tokens, word_emb_size, hidden_size,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), num_layers=1, num_filters=3,
                 kernel_size=1, stride=5, fusion="cat", env=None,
                 condition_answer="none", attention_dim=512):
        super(PolicyLSTMBatch, self).__init__()
        self.device = device
        self.condition_answer = condition_answer
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.word_embedding = nn.Embedding(num_tokens, word_emb_size, padding_idx=0)
        self.lstm = nn.LSTM(word_emb_size, self.hidden_size, batch_first=True)
        truncature = {"masked": mask_truncature, "masked_inf": mask_inf_truncature}
        self.truncate = truncature["masked_inf"]
        self.answer_embedding = nn.Embedding(env.dataset.len_vocab_answer, word_emb_size)
        self.fusion = fusion
        self.num_filters = word_emb_size if num_filters is None else num_filters
        self.stride = stride
        self.kernel_size = kernel_size
        self.word_emb_size = word_emb_size
        self.attention_dim = attention_dim
        h_out = int((14 + 2 * 0 - 1 * (self.kernel_size - 1) - 1) / self.stride + 1)
        self.conv = nn.Conv2d(in_channels=1024, out_channels=self.num_filters, kernel_size=self.kernel_size,
                              stride=self.stride)
        if self.fusion == "none":
            self.fusion_dim = self.hidden_size
        elif self.fusion == "film":
            self.gammabeta = nn.Linear(self.hidden_size, 2 * self.num_filters)
            self.film = contrib_nn.FiLM()
            self.fusion_dim = self.num_filters * h_out ** 2
        elif self.fusion == "sat":
            self.attention = Attention(101, self.hidden_size, attention_dim, word_emb_size)
            self.init_h = nn.Linear(101, hidden_size)  # linear layer to find initial hidden state of LSTMCell
            self.init_c = nn.Linear(101, hidden_size)  # linear layer to find initial cell state of LSTMCell
            self.last_states = None
            self.f_beta = nn.Linear(hidden_size, 101)  # linear layer to create a sigmoid-activated gate
            self.sigmoid = nn.Sigmoid()
            self.fc = nn.Linear(hidden_size, self.num_tokens)
            self.decode_step = nn.LSTMCell(word_emb_size + 101, hidden_size, bias=True)
            self.fusion_dim = hidden_size
        elif self.fusion == "average":
            self.projection = nn.Linear(2048, hidden_size)
            self.avg_pooling = nn.AvgPool1d(kernel_size=101)
            self.fusion_dim = 2 * hidden_size
        else:
            self.fusion_dim = self.num_filters * h_out ** 2 + self.hidden_size

        if self.condition_answer in ["after_fusion", "attention"]:
            self.fusion_dim += word_emb_size

        self.action_head = nn.Linear(self.fusion_dim, num_tokens)
        self.value_head = nn.Linear(self.fusion_dim, 1)

    def init_hidden_state(self, state):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        (text, img, answer) = state
        h, c = torch.zeros((img.size(0), self.hidden_size)).to(self.device), torch.zeros(
            (img.size(0), 1, self.hidden_size)).to(self.device)
        if self.fusion == "sat":
            encoder_out = img.transpose(2, 1).to(self.device)
            mean_encoder_out = encoder_out.mean(dim=1)
            h = self.init_h(mean_encoder_out.to(self.device))  # (batch_size, decoder_dim)
            c = self.init_c(mean_encoder_out.to(self.device))
        return h, c

    def filter_logits(self, logits):
        return logits

    def forward(self, state_text, state_img, state_answer=None, valid_actions=None, logits_lm=0, alpha=0., ht=None,
                ct=None):
        embed_text, ct = self._get_embed_text(state_text, state_answer, state_img, ht, ct)
        state_answer = state_answer if state_answer is None else state_answer.to(self.device)
        img_feat = state_img.to(self.device)  # shape (1, 1024, 14, 14) vs (1,101,2048)
        img_feat_ = img_feat if self.fusion in ["average", "none", "sat"] else F.relu(
            self.conv(img_feat))  # shape (1,3,7,7)
        embedding = self.process_fusion(embed_text, img_feat_, img_feat, state_answer)
        logits = self.action_head(embedding)  # (B,S,num_tokens)
        logits_filtered = self.filter_logits(logits)
        value = self.value_head(embedding)
        # adding lm logits bonus
        logits_exploration = (1 - alpha) * logits_filtered + alpha * logits_lm
        policy_dist, policy_dist_truncated = self.get_policies(valid_actions, logits_exploration)
        return policy_dist, policy_dist_truncated, value, embed_text, ct

    def get_policies(self, valid_actions, logits_exploration):
        probs = F.softmax(logits_exploration, dim=-1)
        policy_dist = Categorical(probs)
        if valid_actions is not None:
            policy_dist_truncated = self.truncate(valid_actions, logits_exploration, device=self.device,
                                                  num_tokens=self.num_tokens)
        else:
            policy_dist_truncated = Categorical(F.softmax(logits_exploration, dim=-1))
        if torch.isnan(policy_dist_truncated.probs).any():
            print("policy dist truncated with nan")
        return policy_dist, policy_dist_truncated

    def process_fusion(self, embed_text, img_feat_, img_feat, answer):
        if self.fusion == "none":
            embedding = embed_text
        elif self.fusion == "film":
            gammabeta = self.gammabeta(embed_text).view(-1, 2, self.num_filters)
            gamma, beta = gammabeta[:, 0, :], gammabeta[:, 1, :]
            embedding = self.film(img_feat_, gamma, beta).view(img_feat.size(0), -1)
        elif self.fusion == "average":
            img_feat__ = self.projection(img_feat_)  # (1,101,64)
            img_feat__ = img_feat__.transpose(2, 1)
            img_feat__ = self.avg_pooling(img_feat__)  # (1,64,1)
            img_feat__ = img_feat__.squeeze(dim=-1)
            embedding = torch.cat((img_feat__, embed_text), dim=-1)  # (B,S,hidden_size).
        elif self.fusion == "sat":
            embedding = embed_text
        else:
            img_feat__ = img_feat_.view(img_feat.size(0), -1)
            embedding = torch.cat((img_feat__, embed_text), dim=-1)  # (B,S,hidden_size).
        if self.condition_answer in ["after_fusion", "attention"] and answer is not None:
            embedding = torch.cat([embedding, self.answer_embedding(answer.view(-1))], dim=1)
        return embedding

    def _get_embed_text(self, text, answer, img, h, c):
        lens = (text != 0).sum(dim=1).type(torch.int64).cpu()
        pad_embed = self.word_embedding(text.to(self.device))
        if self.fusion == "sat":
            last_word_embedding = pad_embed[torch.arange(pad_embed.size(0)), lens - 1]
            img_transposed = img.transpose(2, 1).to(self.device)
            # h, c = self.init_hidden_state(img_transposed) if pad_embed.size(1) == 1 else (ht, ct)
            answer_embedding = None
            if self.condition_answer == "attention":
                answer_embedding = self.answer_embedding(answer.view(text.size(0), 1).to(self.device))

            attention_weighted_encoding, alpha = self.attention(img_transposed, h.to(self.device), answer_embedding)
            gate = self.sigmoid(self.f_beta(h))
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(torch.cat([last_word_embedding, attention_weighted_encoding], dim=1), (h, c))
            # self.last_states = (h, c)
            return h, c

        if self.condition_answer == "before_lstm" and answer is not None:
            pad_embed = torch.cat([pad_embed, self.answer_embedding(answer.view(text.size(0), 1)).to(self.device)],
                                  dim=1)
        pad_embed_pack = pack_padded_sequence(pad_embed, lens, batch_first=True, enforce_sorted=False)
        packed_output, (ht, ct) = self.lstm(pad_embed_pack)
        return ht[-1], ct


class PolicyGPTBatch(PolicyLSTMBatch):
    def __init__(self, num_tokens, word_emb_size, hidden_size,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), num_layers=1, num_filters=3,
                 kernel_size=1, stride=5, fusion="cat", env=None,
                 condition_answer="none", attention_dim=512):
        # hidden size to 768 for transformers
        self.hidden_size_transformers = 768
        super().__init__(num_tokens, word_emb_size, hidden_size,
                         device=device, num_layers=num_layers,
                         num_filters=num_filters, kernel_size=kernel_size,
                         stride=stride, fusion=fusion, env=env, condition_answer=condition_answer,
                         attention_dim=attention_dim)
        self.lm_model = AutoModelWithLMHead.from_pretrained("cache/gpt-2")
        self.tokenizer = AutoTokenizer.from_pretrained("cache/gpt-2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.dataset_tokenizer = env.dataset.question_tokenizer
        self.fusion_dim = self.hidden_size_transformers + hidden_size
        # self.fusion_dim = hidden_size_transformers
        if self.condition_answer in ["after_fusion", "attention"]:
            self.fusion_dim += word_emb_size
        self.action_head = nn.Linear(self.fusion_dim, num_tokens)
        self.value_head = nn.Linear(self.fusion_dim, 1)
        lm_weights_vocab = self.lm_model.lm_head.weight[list(env.dataset.lm_to_dataset_trad.keys())]
        self.text_fc = torch.cat((torch.zeros((4, 768)), lm_weights_vocab))
        self.action_head.weight.data = torch.zeros_like(self.action_head.weight.data)
        self.action_head.weight.data.t()[self.hidden_size:-self.word_emb_size] = self.text_fc.t()
        self.env = env
        start_input_encoded = torch.tensor([self.tokenizer.bos_token_id])
        self.start_input_for_gpt = self.tokenizer.decode(start_input_encoded)
        self.init_text = f"Here are a few examples:{self.get_init_text(10)}"
        self.init_batch = self.tokenizer([self.init_text], padding=True, truncation=True, return_tensors="pt")
        past_key_values = self.lm_model(**self.init_batch, output_hidden_states=True, use_cache=True).past_key_values
        self.init_past_key_values = [[kv[0].detach(), kv[1].detach()] for kv in past_key_values]

    def _get_embed_text(self, text, answer, img, h, c):
        batch_sentences = ["?" + self.dataset_tokenizer.decode(x.cpu().numpy().ravel(),
                                                               stop_at_end=True) for x in text]
        batch_size = len(batch_sentences)
        # if self.init_text is not None:
        #    batch_sentences = self.init_text + batch_sentences
        batch = self.tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
        init_past_key_values = [[k.repeat_interleave(batch_size, 0).detach().to(self.device),
                                 v.repeat_interleave(batch_size, 0).detach().to(self.device)] for (k, v) in
                                self.init_past_key_values]
        init_batch = self.init_batch.attention_mask.repeat((batch_size, 1))
        attention_mask = torch.cat((init_batch, batch.attention_mask), dim=-1).to(self.device)
        input_ids = batch.input_ids.to(self.device)
        # check if input_ids is empty to avoid the runtime error in the forward pass
        # TODO understand why it happens
        if batch["input_ids"].nelement() == 0:
            batch = self.tokenizer([self.start_input_for_gpt], padding=True, truncation=True, return_tensors="pt")
        outputs = self.lm_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True,
                                past_key_values=init_past_key_values)
        lengths = batch["attention_mask"].sum(dim=-1)
        index = (lengths - 1)
        embed_text = outputs["hidden_states"][-1][torch.arange(index.size(0)), index]
        return embed_text, torch.zeros_like(embed_text.unsqueeze(dim=1))

    def init_hidden_state(self, state):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :return: hidden state, cell state
        """
        (text, img, answer) = state
        h, c = torch.zeros((img.size(0), self.hidden_size_transformers)).to(self.device), torch.zeros(
            (img.size(0), 1, self.hidden_size_transformers)).to(self.device)
        if self.fusion == "sat":
            encoder_out = img.transpose(2, 1).to(self.device)
            mean_encoder_out = encoder_out.mean(dim=1)
            h = self.init_h(mean_encoder_out.to(self.device))  # (batch_size, decoder_dim)
            c = self.init_c(mean_encoder_out.to(self.device))
        return h, c

    def filter_logits(self, logits):
        logits[:, torch.arange(0, 4)] = torch.tensor(-1e32).to(self.device)
        return logits

    def get_init_text(self, custom_init=10):
        idxs = np.random.randint(0, len(self.env.dataset.remaining_entries), size=custom_init)
        samples = np.array(self.env.dataset.remaining_entries)[list(set(idxs))]
        example_questions = [s["question"] for s in samples]
        return " ".join(example_questions)


class PolicyGPTBatch_No_Cond(PolicyGPTBatch):
    def __init__(self, num_tokens, word_emb_size, hidden_size,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), num_layers=1, num_filters=3,
                 kernel_size=1, stride=5, fusion="cat", env=None,
                 condition_answer="none", attention_dim=512):
        super().__init__(num_tokens, word_emb_size, hidden_size,
                         device=device, num_layers=num_layers,
                         num_filters=num_filters, kernel_size=kernel_size,
                         stride=stride, fusion=fusion, env=env, condition_answer=condition_answer,
                         attention_dim=attention_dim)
        self.lm_model = AutoModelWithLMHead.from_pretrained("cache/gpt-2")
        self.tokenizer = AutoTokenizer.from_pretrained("cache/gpt-2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.dataset_tokenizer = env.dataset.question_tokenizer
        self.fusion_dim = self.hidden_size_transformers + hidden_size
        # self.fusion_dim = hidden_size_transformers
        if self.condition_answer in ["after_fusion", "attention"]:
            self.fusion_dim += word_emb_size
        self.action_head = nn.Linear(self.fusion_dim, num_tokens)
        self.value_head = nn.Linear(self.fusion_dim, 1)
        lm_weights_vocab = self.lm_model.lm_head.weight[list(env.dataset.lm_to_dataset_trad.keys())]
        self.text_fc = torch.cat((torch.zeros((4, 768)), lm_weights_vocab))
        self.action_head.weight.data = torch.zeros_like(self.action_head.weight.data)
        self.action_head.weight.data.t()[self.hidden_size:-self.word_emb_size] = self.text_fc.t()
        self.env = env

    def _get_embed_text(self, text, answer, img, h, c):
        batch_sentences = [self.tokenizer.bos_token + self.dataset_tokenizer.decode(x.cpu().numpy().ravel(),
                                                                                    stop_at_end=True) for x in text]
        batch = self.tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
        # check if input_ids is empty to avoid the runtime error in the forward pass
        # TODO understand why it happens
        if batch["input_ids"].nelement() == 0:
            batch = self.tokenizer([self.tokenizer.pad_token], padding=True, truncation=True, return_tensors="pt")
        outputs = self.lm_model(**batch, output_hidden_states=True)
        if len(batch_sentences):
            v, i = torch.sort(outputs.logits.squeeze(), descending=True)
            #print(self.tokenizer.batch_decode(i[:10]))
        lengths = batch["attention_mask"].sum(dim=-1)
        index = (lengths - 1)
        # self.lm_model.generate()
        embed_text = outputs["hidden_states"][-1][torch.arange(index.size(0)), index]
        return embed_text, torch.zeros_like(embed_text.unsqueeze(dim=1))

    def _prepare_input_ids_for_generation(
            self, bos_token_id: Optional[int]) -> torch.LongTensor:
        if bos_token_id is None:
            raise ValueError("`bos_token_id` has to be defined when no `input_ids` are provided.")
        return torch.ones((1, 1), dtype=torch.long, device=self.device) * bos_token_id

    def _prepare_attention_mask_for_generation(
            self, input_ids: torch.Tensor, pad_token_id: int, eos_token_id: int
    ) -> torch.LongTensor:
        is_pad_token_in_inputs_ids = (pad_token_id is not None) and (pad_token_id in input_ids)
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (
                (eos_token_id is not None) and (pad_token_id != eos_token_id)
        )
        if is_pad_token_in_inputs_ids and is_pad_token_not_equal_to_eos_token_id:
            return input_ids.ne(pad_token_id).long()
        return input_ids.new_ones(input_ids.shape, dtype=torch.long)


class PolicyCLOSUREBatch(PolicyLSTMBatch):
    def __init__(self, num_tokens, word_emb_size, hidden_size,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), num_layers=1, num_filters=3,
                 kernel_size=1, stride=5, fusion="cat", env=None,
                 condition_answer="none", attention_dim=512):
        super().__init__(num_tokens, word_emb_size, hidden_size,
                         device=device, num_layers=num_layers,
                         num_filters=num_filters, kernel_size=kernel_size,
                         stride=stride, fusion=fusion, env=env, condition_answer=condition_answer,
                         attention_dim=attention_dim)
        lm_path = "output/lm_ext/model.pt"
        lm_model = torch.load(lm_path, map_location=torch.device(device))
        lm_model.train()
        self.lm_model = ClevrLanguageModel(pretrained_lm=lm_model, dataset=env.dataset,
                                           tokenizer=env.dataset.question_tokenizer, device=device,
                                           lm_path=lm_path)
        self.hidden_size_closure = self.lm_model.language_model.lstm.hidden_size
        h_out = int((14 + 2 * 0 - 1 * (self.kernel_size - 1) - 1) / self.stride + 1)
        self.dataset_tokenizer = env.dataset.question_tokenizer
        img_dim = self.num_filters * h_out ** 2
        self.fusion_dim = img_dim + self.hidden_size_closure
        if self.condition_answer in ["after_fusion", "attention"]:
            self.fusion_dim += word_emb_size
        self.action_head = nn.Linear(self.fusion_dim, num_tokens)
        self.value_head = nn.Linear(self.fusion_dim, 1)

        self.unk_idx = env.dataset.vocab_questions["<UNK>"]
        lm_weights_vocab = self.lm_model.language_model.fc.weight  # [list(env.dataset.lm_to_dataset_trad.keys())]
        # lm_weights_vocab[self.unk_idx] = 0.
        self.action_head.weight.data = torch.zeros_like(self.action_head.weight.data)
        self.action_head.weight.data.t()[img_dim:-self.word_emb_size] = lm_weights_vocab.t()

    def _get_embed_text(self, text, answer, img, h, c):
        lens = (text != 0).sum(dim=1).type(torch.int64).cpu()
        emb = self.lm_model.language_model.embedding(text.to(self.device))
        emb = self.lm_model.language_model.dropout(emb)
        output, (ht, ct) = self.lm_model.language_model.lstm(emb)
        output_ = self.lm_model.language_model.dropout(output)
        index = (lens - 1)
        embed_text = output_[torch.arange(index.size(0)), index, :]
        return embed_text, ct

    def init_hidden_state(self, state):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :return: hidden state, cell state
        """
        (text, img, answer) = state
        h, c = torch.zeros((img.size(0), self.hidden_size_closure)).to(self.device), torch.zeros(
            (img.size(0), 1, self.hidden_size_closure)).to(self.device)
        return h, c

    def filter_logits(self, logits):
        logits[:, self.unk_idx] = torch.tensor(-1e32).to(self.device)
        return logits


class PolicyLSTMBatch_SL(nn.Module):
    def __init__(self, num_tokens, word_emb_size, hidden_size,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), num_layers=1, num_filters=3,
                 kernel_size=1, stride=5, fusion="cat", condition_answer="none", num_tokens_answer=32,
                 attention_dim=512):
        super(PolicyLSTMBatch_SL, self).__init__()

        self.device = device
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_filters = word_emb_size if num_filters is None else num_filters
        self.stride = stride
        self.kernel_size = kernel_size
        self.fusion = fusion
        self.condition_answer = condition_answer

        self.word_embedding = nn.Embedding(num_tokens, word_emb_size, padding_idx=0)
        self.lstm = nn.LSTM(word_emb_size, self.hidden_size, batch_first=True)
        self.answer_embedding = nn.Embedding(num_tokens_answer, word_emb_size)
        self.conv = nn.Conv2d(in_channels=1024, out_channels=self.num_filters, kernel_size=self.kernel_size,
                              stride=self.stride)
        self.attention_dim = attention_dim

        h_out = int((14 + 2 * 0 - 1 * (self.kernel_size - 1) - 1) / self.stride + 1)

        if self.fusion == "film":
            self.gammabeta = nn.Linear(self.hidden_size, 2 * self.num_filters)
            self.film = contrib_nn.FiLM()
            self.fusion_dim = self.num_filters * h_out ** 2
        elif self.fusion == "average":
            self.projection = nn.Linear(2048, hidden_size)
            self.avg_pooling = nn.AvgPool1d(kernel_size=101)
            self.fusion_dim = 2 * hidden_size
        elif self.fusion == "sat":
            self.attention = Attention(101, self.hidden_size, attention_dim, word_emb_size)
            self.init_h = nn.Linear(101, hidden_size)  # linear layer to find initial hidden state of LSTMCell
            self.init_c = nn.Linear(101, hidden_size)  # linear layer to find initial cell state of LSTMCell
            self.last_states = None
            self.f_beta = nn.Linear(hidden_size, 101)  # linear layer to create a sigmoid-activated gate
            self.sigmoid = nn.Sigmoid()
            self.fc = nn.Linear(hidden_size, self.num_tokens)
            self.decode_step = nn.LSTMCell(word_emb_size + 101, hidden_size, bias=True)
            self.fusion_dim = hidden_size
        else:
            self.fusion_dim = self.num_filters * h_out ** 2 + self.hidden_size

        if self.condition_answer in ["after_fusion", "attention"]:
            self.fusion_dim += word_emb_size

        self.action_head = nn.Linear(self.fusion_dim, num_tokens)

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out.to(self.device))  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out.to(self.device))
        return h, c

    def _get_embed_text(self, text, img, answer):
        lens = (text != 0).sum(dim=1)
        pad_embed = self.word_embedding(text)
        if self.fusion == "sat":
            output = torch.zeros(text.size(0), text.size(1), self.hidden_size).to(self.device)

            caption_lengths, sort_ind = lens.sort(dim=0, descending=True)
            img = img[sort_ind]
            answer = answer[sort_ind]
            text = text[sort_ind]
            pad_embed = pad_embed[sort_ind]

            img_transposed = img.transpose(2, 1).to(self.device)
            decode_lengths = caption_lengths.tolist()
            h, c = self.init_hidden_state(img_transposed)
            for t in range(max(decode_lengths)):
                batch_size_t = sum([l > t for l in decode_lengths])
                answer_embedding = None
                if self.condition_answer == "attention":
                    answer_embedding = self.answer_embedding(answer[:batch_size_t].unsqueeze(dim=1)).to(self.device)

                attention_weighted_encoding, alpha = self.attention(img_transposed[:batch_size_t],
                                                                    h[:batch_size_t].to(self.device),
                                                                    answer_embedding)
                gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
                attention_weighted_encoding = gate * attention_weighted_encoding
                h, c = self.decode_step(torch.cat([pad_embed[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                                        (h[:batch_size_t], c[:batch_size_t]))
                output[:batch_size_t, t, :] = h
            packed_output = pack_padded_sequence(output, caption_lengths, batch_first=True)
            pad_output, input_sizes = pad_packed_sequence(packed_output, batch_first=True, total_length=text.size(1))
            outputs = pad_output[sort_ind.argsort()]
            return outputs

        pad_embed_pack = pack_padded_sequence(pad_embed, lens, batch_first=True, enforce_sorted=False)
        packed_output, (ht, ct) = self.lstm(pad_embed_pack)
        output, input_sizes = pad_packed_sequence(packed_output, batch_first=True, total_length=text.size(1))
        return output

    def process_fusion(self, embed_text, img_feat_, img_feat, answer, seq_len):
        if self.fusion == "film":
            gammabeta = self.gammabeta(embed_text).view(embed_text.size(0), embed_text.size(1), 2, self.num_filters)
            gamma, beta = gammabeta[:, :, 0, :], gammabeta[:, :, 1, :]
            img_feat__ = img_feat_.unsqueeze(1).repeat(1, seq_len, 1, 1, 1).view(-1, img_feat_.size(1),
                                                                                 img_feat_.size(2),
                                                                                 img_feat_.size(2))
            embedding = self.film(img_feat__, gamma.view(-1, gamma.size(2)), beta.view(-1, beta.size(2)))
            embedding = embedding.view(embed_text.size(0), embed_text.size(1), -1)
        elif self.fusion == "average":
            img_feat__ = img_feat_.transpose(2, 1)  # (B, 2048, 101)
            img_feat__ = self.avg_pooling(img_feat__).squeeze(-1)  # (B,2048)
            img_feat__ = self.projection(img_feat__)  # (B,hidden_size)
            img_feat__ = img_feat__.unsqueeze(1).repeat(1, seq_len, 1)
            embedding = torch.cat((img_feat__, embed_text), dim=-1)  # (B,S,hidden_size).
        elif self.fusion == "sat":
            embedding = embed_text
        else:
            img_feat__ = img_feat_.view(img_feat.size(0), -1).unsqueeze(1).repeat(1, seq_len, 1)
            embedding = torch.cat((img_feat__, embed_text), dim=-1)  # (B,S,hidden_size).

        if self.condition_answer in ["after_fusion", "attention"] and answer is not None:
            repeated_answer = self.answer_embedding(answer).unsqueeze(1).repeat(1, seq_len, 1)
            embedding = torch.cat([embedding, repeated_answer], dim=2)

        return embedding

    def forward(self, state_text, state_img, state_answer):
        embed_text = self._get_embed_text(state_text, state_img, state_answer)
        seq_len = embed_text.size(1)
        img_feat = state_img.to(self.device)
        img_feat_ = img_feat if self.fusion in ["average", "none", "sat"] else F.relu(self.conv(img_feat))
        embedding = self.process_fusion(embed_text=embed_text, img_feat_=img_feat_, img_feat=img_feat,
                                        answer=state_answer, seq_len=seq_len)
        logits = self.action_head(embedding)  # (B,S,num_tokens)
        logits = logits.view(-1, self.num_tokens)  # (S*B, num_tokens)
        value = None
        return logits, value


if __name__ == '__main__':
    train_features_path = '../../data/train_features.h5'
    hf = h5py.File(train_features_path, 'r')
    img_feat = hf.get('features')
    img_feat = np.array(img_feat)
    print('features shape', img_feat.shape)  # shape (num_samples, 1024, 14, 14).

    img_feat = torch.tensor(img_feat, dtype=torch.float32)
    img_feat_RL = img_feat[0].unsqueeze(0)
    seq_len = 10
    num_tokens = 85
    word_emb_size = 64
    hidden_size = 128
    dummy_text_input = torch.ones(img_feat.size(0), seq_len, dtype=torch.long)
    dummy_text_input_RL = torch.ones(img_feat_RL.size(0), seq_len, dtype=torch.long)

    # RL mode.
    model = PolicyLSTMBatch(num_tokens=num_tokens, word_emb_size=word_emb_size, hidden_size=hidden_size)
    policy_dist, policy_dist_truncated, value = model(dummy_text_input_RL, img_feat_RL, valid_actions=[0, 4, 8, 10])
    model = PolicyLSTMBatch(num_tokens=num_tokens, word_emb_size=word_emb_size, hidden_size=hidden_size,
                            truncate_mode="masked_inf")
    policy_dist, policy_dist_truncated, value = model(dummy_text_input_RL, img_feat_RL, valid_actions=[0, 4, 8, 10])

    # SL mode.
    model = PolicyLSTMBatch_SL(num_tokens=num_tokens, word_emb_size=word_emb_size, hidden_size=hidden_size)
    logits, value = model(dummy_text_input, img_feat)
    print(logits.shape)
