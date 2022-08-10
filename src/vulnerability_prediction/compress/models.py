"""Only Model, biLSTM and distill_loss are used in experiments."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss


class Model(nn.Module):
    def __init__(self, encoder):
        super(Model, self).__init__()
        self.encoder = encoder

    def forward(self, input_ids=None, labels=None):
        logits = self.encoder(input_ids, attention_mask=input_ids.ne(1))[0]
        prob = F.sigmoid(logits)
        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0] + 1e-10)*labels + \
                torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            loss = -loss.mean()
            return loss, prob
        else:
            return logits


class LSTM(nn.Module):
    def __init__(self, vocab_size, input_dim, hidden_dim, n_labels, n_layers):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, input_dim)
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=n_layers,
                            batch_first=True,
                            bidirectional=False,
                            dropout=0.2)
        self.fc = nn.Linear(hidden_dim, n_labels)

    def forward(self, input_ids, labels=None):
        embed = self.embedding(input_ids)
        outputs, (hidden, _) = self.lstm(embed)
        hidden = hidden.permute(1, 0, 2)
        hidden = hidden[:, -1, :]
        # hidden = torch.cat((hidden[:, -1, :], hidden[:, -2, :]), dim=1)
        logits = self.fc(hidden)
        prob = F.softmax(logits)

        if labels is not None:
            labels = labels.long()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, (1 - labels))
            return loss, prob
        else:
            return prob


class biLSTM(nn.Module):
    def __init__(self, vocab_size, input_dim, hidden_dim, n_labels, n_layers):
        super(biLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, input_dim)
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=n_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=0.2)
        self.dense = nn.Linear(hidden_dim * 2, 200)
        self.fc = nn.Linear(200, n_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, labels=None):
        embed = self.embedding(input_ids)
        outputs, (hidden, _) = self.lstm(embed)
        hidden = hidden.permute(1, 0, 2)
        hidden = torch.cat((hidden[:, -1, :], hidden[:, -2, :]), dim=1)
        x = F.relu(self.dense(hidden))
        x = self.dropout(x)
        logits = self.fc(x)
        prob = F.softmax(logits)

        if labels is not None:
            labels = labels.long()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob


class GRU(nn.Module):
    def __init__(self, vocab_size, input_dim, hidden_dim, n_labels, n_layers):
        super(GRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, input_dim)
        self.gru = nn.GRU(input_size=input_dim,
                          hidden_size=hidden_dim,
                          num_layers=n_layers,
                          batch_first=True,
                          bidirectional=False,
                          dropout=0.2)
        self.fc = nn.Linear(hidden_dim, n_labels)

    def forward(self, input_ids, labels=None):
        embed = self.embedding(input_ids)
        _, hidden = self.gru(embed)
        hidden = hidden.permute(1, 0, 2)
        hidden = hidden[:, -1, :]
        # hidden = torch.cat((hidden[:, -1, :], hidden[:, -2, :]), dim=1)
        logits = self.fc(hidden)
        prob = F.softmax(logits)

        if labels is not None:
            labels = labels.long()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, (1 - labels))
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
                          bidirectional=True,
                          dropout=0.2)
        self.fc = nn.Linear(hidden_dim * 2, n_labels)

    def forward(self, input_ids, labels=None):
        embed = self.embedding(input_ids)
        _, hidden = self.gru(embed)
        hidden = hidden.permute(1, 0, 2)
        hidden = torch.cat((hidden[:, -1, :], hidden[:, -2, :]), dim=1)
        logits = self.fc(hidden)
        prob = F.softmax(logits)

        if labels is not None:
            labels = labels.long()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, (1 - labels))
            return loss, prob
        else:
            return prob


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, vocab_size=5000):
        super().__init__()

        pe = torch.zeros(vocab_size, d_model)
        position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, input_dim, hidden_dim, n_labels, n_layers):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, input_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=8, dim_feedforward=hidden_dim, dropout=0.2)
        self.pos_encoder = PositionalEncoding(
            d_model=input_dim, vocab_size=vocab_size)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, n_layers)
        self.fc = nn.Linear(input_dim, n_labels)
        self.input_dim = input_dim

    def forward(self, input_ids, labels=None):
        embed = self.embedding(input_ids) * math.sqrt(self.input_dim)
        embed = self.pos_encoder(embed)
        hidden = self.transformer_encoder(embed)
        hidden = hidden[:, 0, :]
        logits = self.fc(hidden)
        prob = F.softmax(logits)

        if labels is not None:
            labels = labels.long()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, (1 - labels))
            return loss, prob
        else:
            return prob


def loss_func(preds, labels, knowledge):
    labels = labels.long()
    knowledge = knowledge.long()

    loss = 0.5 * F.cross_entropy(preds, 1-labels) + \
        0.5 * F.cross_entropy(preds, 1-knowledge)

    return loss


def mix_loss_func(preds, labels, knowledge):
    labels = labels.long()
    knowledge = knowledge.long()

    loss = 0
    for p, l, k in zip(preds, labels, knowledge):
        p = p.view(1, 2)
        if l == -1.0:
            loss += F.cross_entropy(p, (1-k).view(1))
        else:
            loss += 0.5 * \
                F.cross_entropy(p, (1-l).view(1)) + 0.5 * \
                F.cross_entropy(p, (1-k).view(1))

    loss = loss/labels.size(0)
    return loss


def distill_loss(logits, knowledge, temperature=10.0):

    loss = F.kl_div(F.log_softmax(logits/temperature), F.softmax(knowledge /
                    temperature), reduction="batchmean") * (temperature**2)
    # Equivalent to cross_entropy for soft labels, from https://github.com/huggingface/transformers/blob/50792dbdcccd64f61483ec535ff23ee2e4f9e18d/examples/distillation/distiller.py#L330

    return loss


def mse_loss(logits, knowledge):
    kd_criterion = nn.MSELoss()
    loss = kd_criterion(logits, knowledge)

    return loss
