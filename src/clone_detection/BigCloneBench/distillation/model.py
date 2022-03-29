import torch
import torch.nn as nn
import torch.nn.functional as F

class RobertaClassificationHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features):
        x = features[:, 0, :]
        x = x.reshape(-1, x.size(-1)*2)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Roberta(nn.Module):
    def __init__(self, encoder, config, args):
        super(Roberta, self).__init__()
        self.encoder = encoder
        self.config = config
        self.classifier = RobertaClassificationHead(config)
        self.args = args

    def forward(self, input_ids=None):
        input_ids = input_ids.view(-1, self.args.block_size)
        outputs = self.encoder(input_ids=input_ids,
                               attention_mask=input_ids.ne(1))[0]
        logits = self.classifier(outputs)
        return logits


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

    def forward(self, input_ids):
        embed = self.embedding(input_ids)
        outputs, (hidden, _) = self.lstm(embed)
        hidden = hidden.permute(1, 0, 2)
        hidden = torch.cat((hidden[:, -1, :], hidden[:, -2, :]), dim=1)
        logits = self.fc(hidden)
        # prob = F.sigmoid(logits)
        return logits

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

    def forward(self, input_ids):
        embed = self.embedding(input_ids)
        _, hidden = self.gru(embed)
        hidden = hidden.permute(1, 0, 2)
        hidden = torch.cat((hidden[:, -1, :], hidden[:, -2, :]), dim=1)
        logits = self.fc(hidden)
        # prob = F.sigmoid(logits)
        return logits


def ce_loss_func(std_logits, tea_logits, labels, alpha=0.9, temperature=2.0):
    labels = labels.long()

    loss = F.cross_entropy(std_logits, labels)

    ce_loss = F.kl_div(F.log_softmax(std_logits/temperature), F.softmax(tea_logits/temperature), reduction="batchmean") * (temperature**2)
    # Equivalent to cross_entropy for soft labels, from https://github.com/huggingface/transformers/blob/50792dbdcccd64f61483ec535ff23ee2e4f9e18d/examples/distillation/distiller.py#L330

    return alpha * loss + (1. - alpha) * ce_loss

def mse_loss_func(std_logits, tea_logits, labels, alpha=0.9, normalized=True):
    labels = labels.long()

    loss = F.cross_entropy(std_logits, labels)
    
    if normalized:
        mse_loss = F.mse_loss(F.normalize(std_logits, p=2, dim=1), F.normalize(tea_logits, p=2, dim=1), reduction="sum")
    else:
        mse_loss = F.mse_loss(std_logits, tea_logits, reduction="sum")
    mse_loss /= std_logits.size(0)

    return alpha * loss + (1. - alpha) * mse_loss