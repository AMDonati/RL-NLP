# code inspired from: https://github.com/pytorch/examples/blob/master/word_language_model/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.LayerNormLSTM import LayerNormLSTM


class GRUModel(nn.Module):

    def __init__(self, num_tokens, emb_size, hidden_size, num_layers=1, p_drop=0, bidirectional=False):
        super(GRUModel, self).__init__()
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.p_drop = p_drop
        self.dropout = nn.Dropout(p_drop)
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(num_tokens, emb_size)
        self.gru = nn.GRU(input_size=emb_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          dropout=p_drop,
                          bidirectional=bidirectional,
                          batch_first=True)
        if bidirectional:
            in_features = 2 * hidden_size
        else:
            in_features = hidden_size
        self.fc = nn.Linear(in_features=in_features,
                            out_features=num_tokens)

    def forward(self, input):
        emb = self.embedding(input)  # (B, seq_len, emb_size)
        emb = self.dropout(emb)
        output, hidden = self.gru(
            emb)  # output (B,seq_len,hidden_size*num_directions) , hidden: (num_layers * num_directions, B, hidden_size)
        output = self.dropout(output)
        dec_output = self.fc(output)  # (B, S, num_tokens)
        dec_output = dec_output.view(-1, self.num_tokens)  # (S * B, num_tokens) # resizing for the NLL Loss.
        log_probas = F.log_softmax(dec_output,
                                   dim=1)  # when outputting the log_probas, use torch.nn.NLLLoss and not torch.nn.CrossEntropy.

        return log_probas, hidden


class LSTMModel(nn.Module):
    def __init__(self, num_tokens, emb_size, hidden_size, num_layers=1, p_drop=0, bidirectional=False):
        super(LSTMModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_tokens = num_tokens
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.p_drop = p_drop
        self.bidirectional = bidirectional
        self.dropout = nn.Dropout(p_drop)
        self.embedding = nn.Embedding(num_tokens, emb_size)
        self.lstm = nn.LSTM(input_size=emb_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=p_drop,
                            bidirectional=bidirectional,
                            batch_first=True)
        if bidirectional:
            in_features = 2 * hidden_size
        else:
            in_features = hidden_size
        self.fc = nn.Linear(in_features=in_features, out_features=num_tokens)

    def forward(self, input):
        emb = self.embedding(input.to(self.device))  # (B, seq_len, emb_size)
        emb = self.dropout(emb)
        output, hidden = self.lstm(
            emb)  # output (B, seq_len, hidden_size*num_dimension) # hidden: (num_layers * num_directions, B, hidden_size)
        output = self.dropout(output)
        dec_output = self.fc(output)  # (S,B,num_tokens)
        dec_output = dec_output.view(-1, self.num_tokens)  # (S*B, num_tokens)
        log_probas = F.log_softmax(dec_output, dim=-1)

        return log_probas, hidden


class LayerNormLSTMModel(nn.Module):
    def __init__(self, num_tokens, emb_size, hidden_size, num_layers=1, p_drop=0):
        super(LayerNormLSTMModel, self).__init__()
        self.num_tokens = num_tokens
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.p_drop = p_drop
        self.dropout = nn.Dropout(p_drop)
        self.embedding = nn.Embedding(num_tokens, emb_size)
        self.ln_lstm = LayerNormLSTM(input_size=emb_size, hidden_size=hidden_size, num_layers=num_layers, p_drop=p_drop)
        self.fc = nn.Linear(in_features=hidden_size, out_features=num_tokens)

    def forward(self, input):
        emb = self.embedding(input)
        emb = self.dropout(emb)
        output, hidden = self.ln_lstm(
            input=emb)  # output (S,B,hidden_size), hidden = (h,c): (num_layers, B, hidden_size)
        output = self.dropout(output)
        dec_output = self.fc(output)  # (S,B,num_tokens)
        dec_output = dec_output.view(-1, self.num_tokens)  # (S*B, num_tokens) # resizing for the loss.
        log_probas = F.log_softmax(dec_output, dim=1)

        return log_probas, hidden


if __name__ == '__main__':
    batch_size = 8
    emb_size = 512
    hidden_size = 128
    num_tokens = 85
    seq_len = 5
    device = torch.device("cpu")
    inputs = torch.ones(batch_size, seq_len, dtype=torch.long).to(device)

    # ----------- Test of GRU Model ------------------------------------------------------------------------------------------------------------------------
    model = GRUModel(num_tokens=num_tokens, emb_size=emb_size, hidden_size=hidden_size, bidirectional=True)
    output, hidden = model(inputs)
    print('output', output.shape)
    print('hidden', hidden.shape)
    output = output.view(batch_size, seq_len, num_tokens)
    print('output reshaped', output.shape)

    # ------------ Test of LSTM Model ---------------------------------------------------------------------------------------------------------------------------

    model = LSTMModel(num_tokens=num_tokens, emb_size=emb_size, hidden_size=hidden_size, bidirectional=True)
    output, (h, c) = model(inputs)
    print('output', output.shape)
    print('hidden state', h.shape)
    print('cell state', c.shape)

    # ----------- Test of LSTM with LayerNorm Model -------------------------------------------------------------------------------------------------------------

    model = LayerNormLSTMModel(num_tokens=num_tokens, emb_size=emb_size, hidden_size=hidden_size, num_layers=2,
                               p_drop=1)
    output, (h, c) = model(inputs)
    print('output', output.shape)
    print('hidden state', h.shape)
    print('cell state', c.shape)
