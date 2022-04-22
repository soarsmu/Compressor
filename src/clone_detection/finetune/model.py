import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss


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


class Model(nn.Module):
    def __init__(self, encoder, config, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.classifier = RobertaClassificationHead(config)
        self.args = args

    def forward(self, input_ids=None, labels=None):
        input_ids = input_ids.view(-1, self.args.block_size)
        outputs = self.encoder(input_ids=input_ids,
                               attention_mask=input_ids.ne(1))[0]
        logits = self.classifier(outputs)
        prob = F.softmax(logits)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob


class biLSTM(nn.Module):
    def __init__(self):
        super(biLSTM, self).__init__()
        self.Embedding = nn.Embedding(50265, 200)
        self.lstm = nn.LSTM(input_size=200, hidden_size=512,
                            num_layers=1, batch_first=True, dropout=0, bidirectional=True)
        # self.linear = nn.Linear(in_features=256, out_features=2)
        self.fc1 = nn.Linear(512*8, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x, hidden=None):
        x = x.view(-1, 200)
        x = self.Embedding(x)
        lstm_out, hidden = self.lstm(x, hidden)     # LSTM 的返回很多
        lstm_out = lstm_out[:, 0, :]
        lstm_out = lstm_out.reshape(-1, lstm_out.size(-1)*4)
        # print(lstm_out.size())
        out = self.fc1(lstm_out)
        activated_t = F.relu(out)
        linear_out = self.fc2(activated_t)
        # linear_out = torch.max(linear_out, dim=1)[0]

        return linear_out