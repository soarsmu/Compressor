import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, encoder, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.args = args

    def forward(self, input_ids=None, labels=None): 
        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1))[0]
        logits = outputs
        prob = F.sigmoid(logits)
        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0] + 1e-10)*labels.reshape(16,1) + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels.reshape(16,1))
            loss = -loss.mean()
            return loss, prob
        else:
            return prob
