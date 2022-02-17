import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, encoder):
        super(Model, self).__init__()
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
    def __init__(self):
        super(biLSTM, self).__init__()
        self.Embedding = nn.Embedding(50265, 200)
        self.lstm = nn.LSTM(input_size=200, hidden_size=512,
                            num_layers=1, batch_first=True, dropout=0, bidirectional=True)
        # self.linear = nn.Linear(in_features=256, out_features=2)
        self.fc1 = nn.Linear(512*2, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x, hidden=None):
        x = self.Embedding(x)
        lstm_out, hidden = self.lstm(x, hidden)     # LSTM 的返回很多
        out = self.fc1(lstm_out)
        activated_t = F.relu(out)
        linear_out = self.fc2(activated_t)
        linear_out = torch.max(linear_out, dim=1)[0]

        return linear_out


class CNN(nn.Module):
        def __init__(self, x_dim=50265, e_dim=200, h_dim=512, o_dim=2):
            super(CNN, self).__init__()
            self.emb = nn.Embedding(x_dim, e_dim, padding_idx=0)
            self.dropout = nn.Dropout(0.2)
            self.conv1 = nn.Conv2d(1, h_dim, (3, e_dim))
            self.conv2 = nn.Conv2d(1, h_dim, (4, e_dim))
            self.conv3 = nn.Conv2d(1, h_dim, (5, e_dim))
            self.fc = nn.Linear(h_dim * 3, o_dim)
            self.softmax = nn.Softmax(dim=1)
            self.log_softmax = nn.LogSoftmax(dim=1)

        def forward(self, x):
            embed = self.dropout(self.emb(x)).unsqueeze(1)
            c1 = torch.relu(self.conv1(embed).squeeze(3))
            p1 = torch.max_pool1d(c1, c1.size()[2]).squeeze(2)
            c2 = torch.relu(self.conv2(embed).squeeze(3))
            p2 = torch.max_pool1d(c2, c2.size()[2]).squeeze(2)
            c3 = torch.relu(self.conv3(embed).squeeze(3))
            p3 = torch.max_pool1d(c3, c3.size()[2]).squeeze(2)
            pool = self.dropout(torch.cat((p1, p2, p3), 1))
            hidden = self.fc(pool)
            return self.log_softmax(hidden)