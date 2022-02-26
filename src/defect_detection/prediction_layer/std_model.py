import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class biLSTM(nn.Module):
    def __init__(self, vocab_size, input_dim, hidden_dim, n_labels, n_layers):
        super(biLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, input_dim)
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=n_layers,
                            batch_first=True, 
                            bidirectional=True)

        self.fc = nn.Sequential(nn.GELU(), nn.Linear(hidden_dim * 2, n_labels))

    def forward(self, input_ids, labels=None):
        embed = self.embedding(input_ids)
        outputs, (hidden, _) = self.lstm(embed)
        hidden = hidden.permute(1, 0, 2)
        hidden = torch.cat((hidden[:, -1, :], hidden[:, -2, :]), dim=1)
        logits = self.fc(hidden)
        prob = F.sigmoid(logits)
        if labels is not None:
            loss = self.criterion(prob, labels)
            return loss, prob
        else:
            return prob


class biGRU(nn.Module):
    def __init__(self, vocab_size, input_dim, hidden_dim, n_labels, n_layers):
        super(biGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, input_dim)
        self.gru = nn.GRU(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=n_layers,
                            batch_first=True, 
                            bidirectional=True)

        self.fc = nn.Sequential(nn.GELU(), nn.Linear(hidden_dim * 2, n_labels))

    def forward(self, input_ids, labels=None):
        embed = self.embedding(input_ids)
        _, hidden = self.gru(embed)
        hidden = hidden.permute(1, 0, 2)
        hidden = torch.cat((hidden[:, -1, :], hidden[:, -2, :]), dim=1)
        logits = self.fc(hidden)
        prob = F.sigmoid(logits)

        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0] + 1e-10)*labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            loss = -loss.mean()
            return loss, prob
        else:
            return prob


class TextCNN(nn.Module):
    def __init__(self, vocab_size, input_dim, hidden_dim, n_labels):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, input_dim)
        self.conv1 = nn.Conv2d(1, hidden_dim, (3, input_dim))
        self.conv2 = nn.Conv2d(1, hidden_dim, (4, input_dim))
        self.conv3 = nn.Conv2d(1, hidden_dim, (5, input_dim))
        self.fc = nn.Linear(hidden_dim * 3, n_labels)

    def forward(self, x, labels=None):
        embed = self.embedding(x).unsqueeze(1)
        c1 = torch.relu(self.conv1(embed).squeeze(3))
        p1 = torch.max_pool1d(c1, c1.size()[2]).squeeze(2)
        c2 = torch.relu(self.conv2(embed).squeeze(3))
        p2 = torch.max_pool1d(c2, c2.size()[2]).squeeze(2)
        c3 = torch.relu(self.conv3(embed).squeeze(3))
        p3 = torch.max_pool1d(c3, c3.size()[2]).squeeze(2)
        logits = self.fc(torch.cat((p1, p2, p3), 1))
        prob = F.sigmoid(logits)

        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0] + 1e-10)*labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            loss = -loss.mean()
            return loss, prob
        else:
            return prob


class TransformerModel(nn.Module):

    def __init__(self, vocab_size, d_model: int, nhead: int, d_hid: int, n_labels, n_layers: int,  dropout=0.5):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        self.encoder = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.gelu = nn.GELU()
        self.decoder = nn.Linear(d_model*2, n_labels)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, labels=None):
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        # output = output.permute(1, 0, 2)
        output = torch.cat((output[:, -1, :], output[:, -2, :]), dim=1)
        output = self.gelu(output)
        output = self.decoder(output)
        prob = F.sigmoid(output)

        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0] + 1e-10)*labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            loss = -loss.mean()
            return loss, prob
        else:
            return prob

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout=0.5, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class DistilledLoss(nn.Module):
    def __init__(self, a: int=0.5):
        super(DistilledLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.a = a

    def forward(self, std_logits, bert_logits, labels):
        labels = labels.float()
        loss = torch.log(std_logits[:, 0] + 1e-10)*labels + torch.log((1 - std_logits)[:, 0] + 1e-10) * (1 - labels)
        loss = -loss.mean()
        return self.a * loss + (1. - self.a) * self.mse(std_logits, bert_logits)
