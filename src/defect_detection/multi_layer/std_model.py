import torch
import math
from torch import nn
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

class Model(nn.Module):
    def __init__(self, encoder):
        super(Model, self).__init__()
        self.encoder = encoder
        
    def forward(self, input_ids=None, labels=None):
        output = self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1))[0]
        logits = self.encoder(input_ids, attention_mask=input_ids.ne(1))[0]
        prob = F.sigmoid(logits)
        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0] + 1e-10)*labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            loss = -loss.mean()
            return loss, prob
        else:
            return prob

def distillation_loss(y, labels, teacher_scores, T, alpha, reduction_kd='mean', reduction_nll='mean'):
 
    if teacher_scores is not None:
        d_loss = nn.KLDivLoss(reduction=reduction_kd)(F.log_softmax(y / T, dim=1),
                                                      F.softmax(teacher_scores / T, dim=1)) * T * T
    else:
        assert alpha == 0, 'alpha cannot be {} when teacher scores are not provided'.format(alpha)
        d_loss = 0.0
    nll_loss = F.cross_entropy(y, labels, reduction=reduction_nll)

    tol_loss = alpha * d_loss + (1.0 - alpha) * nll_loss
    
    return tol_loss, d_loss, nll_loss


def patience_loss(teacher_patience, student_patience, normalized_patience=False):
    
    if normalized_patience:
        teacher_patience = F.normalize(teacher_patience, p=2, dim=2)
        student_patience = F.normalize(student_patience, p=2, dim=2)
    return F.mse_loss(teacher_patience.float(), student_patience.float()).half()


class Roberta(nn.Module):
    def __init__(self, encoder):
        super(Roberta, self).__init__()
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