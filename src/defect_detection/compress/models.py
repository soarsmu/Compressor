import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss

class CodeBERT(nn.Module):
    def __init__(self, encoder):
        super(CodeBERT, self).__init__()
        self.encoder = encoder
        
    def forward(self, input_ids=None, labels=None): 
        logits = self.encoder(input_ids, attention_mask=input_ids.ne(1))[0]
        prob = F.sigmoid(logits)
        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0] + 1e-10)*labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            loss = -loss.mean()
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
                            bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, n_labels)

    def forward(self, input_ids, labels=None):
        embed = self.embedding(input_ids)
        outputs, (hidden, _) = self.lstm(embed)
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


class biGRU(nn.Module):
    def __init__(self, vocab_size, input_dim, hidden_dim, n_labels, n_layers):
        super(biGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, input_dim)
        self.gru = nn.GRU(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=n_layers,
                            batch_first=True, 
                            bidirectional=True)
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


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()

    def forward(self, input_ids):
        pass


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

    def forward(self, input_ids):
        pass


def loss_func(preds, labels, knowledge):
    labels = labels.long()
    knowledge = knowledge.long()

    loss = 0.5 * F.cross_entropy(preds, 1-labels) + 0.5 * F.cross_entropy(preds, 1-knowledge)

    return loss