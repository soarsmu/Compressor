import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import numpy as np


class Roberta(nn.Module):
    def __init__(self, encoder):
        super(Roberta, self).__init__()
        self.encoder = encoder
        
    def forward(self, input_ids=None): 
        logits = self.encoder(input_ids, attention_mask=input_ids.ne(1))[0]

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

        self.fc = nn.Linear(hidden_dim * 2, n_labels)

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

        self.fc = nn.Linear(hidden_dim * 2, n_labels)

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

    loss = F.cross_entropy(std_logits, (1 - labels))

    ce_loss = F.kl_div(F.log_softmax(std_logits/temperature), F.softmax(tea_logits/temperature), reduction="batchmean") * (temperature**2)
    # Equivalent to cross_entropy for soft labels, from https://github.com/huggingface/transformers/blob/50792dbdcccd64f61483ec535ff23ee2e4f9e18d/examples/distillation/distiller.py#L330

    return alpha * loss + (1. - alpha) * ce_loss

def mse_loss_func(std_logits, tea_logits, labels, alpha=0.9, normalized=True):
    labels = labels.long()

    loss = F.cross_entropy(std_logits, (1 - labels))
    
    if normalized:
        mse_loss = F.mse_loss(F.normalize(std_logits, p=2, dim=1), F.normalize(tea_logits, p=2, dim=1), reduction="sum")
    else:
        mse_loss = F.mse_loss(std_logits, tea_logits, reduction="sum")
    mse_loss /= std_logits.size(0)

    return alpha * loss + (1. - alpha) * mse_loss

def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)

def log_prob(z, mu, logvar):
    var = torch.exp(logvar)
    logp = - (z-mu)**2 / (2*var) - torch.log(2*np.pi*var) / 2
    return logp.sum(dim=1)

def loss_kl(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(mu)


class TextModel(nn.Module):
    """Container module with word embedding and projection layers"""

    def __init__(self, tokenizer, args, initrange=0.1):
        super().__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.embed = nn.Embedding(args.vocab_size, args.input_dim)
        self.proj = nn.Linear(args.hidden_dim, args.vocab_size)

        self.embed.weight.data.uniform_(-initrange, initrange)
        self.proj.bias.data.zero_()
        self.proj.weight.data.uniform_(-initrange, initrange)


class DAE(TextModel):
    """Denoising Auto-Encoder"""

    def __init__(self, tokenizer, args):
        super().__init__(tokenizer, args)
        self.drop = nn.Dropout(args.dropout)
        self.E = nn.LSTM(args.input_dim, args.hidden_dim, args.n_layers,
            dropout=args.dropout if args.n_layers > 1 else 0, bidirectional=True)
        self.G = nn.LSTM(args.input_dim, args.hidden_dim, args.n_layers,
            dropout=args.dropout if args.n_layers > 1 else 0)
        self.h2mu = nn.Linear(args.hidden_dim*2, args.dim_z)
        self.h2logvar = nn.Linear(args.hidden_dim*2, args.dim_z)
        self.z2emb = nn.Linear(args.dim_z, args.input_dim)
        self.opt = optim.Adam(self.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

    def flatten(self):
        self.E.flatten_parameters()
        self.G.flatten_parameters()

    def encode(self, input):
        input = self.drop(self.embed(input))
        _, (h, _) = self.E(input)
        h = torch.cat([h[-2], h[-1]], 1)
        return self.h2mu(h), self.h2logvar(h)

    def decode(self, z, input, hidden=None):
        input = self.drop(self.embed(input)) + self.z2emb(z)
        output, hidden = self.G(input, hidden)
        output = self.drop(output)
        logits = self.proj(output.view(-1, output.size(-1)))
        return logits.view(output.size(0), output.size(1), -1), hidden

    def generate(self, z, max_len, alg):
        assert alg in ['greedy' , 'sample' , 'top5']
        sents = []
        input = torch.zeros(1, len(z), dtype=torch.long, device=z.device).fill_(self.tokenizer.token_to_id("<s>"))
        hidden = None
        for l in range(max_len):
            sents.append(input)
            logits, hidden = self.decode(z, input, hidden)
            if alg == 'greedy':
                input = logits.argmax(dim=-1)
            elif alg == 'sample':
                input = torch.multinomial(logits.squeeze(dim=0).exp(), num_samples=1).t()
            elif alg == 'top5':
                not_top5_indices=logits.topk(logits.shape[-1]-5,dim=2,largest=False).indices
                logits_exp=logits.exp()
                logits_exp[:,:,not_top5_indices]=0.
                input = torch.multinomial(logits_exp.squeeze(dim=0), num_samples=1).t()
        return torch.cat(sents)

    def forward(self, input, is_train=False):
        _input = input
        mu, logvar = self.encode(_input)
        z = reparameterize(mu, logvar)
        logits, _ = self.decode(z, input)
        return mu, logvar, z, logits

    def loss_rec(self, logits, targets):
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
            ignore_index=self.tokenizer.token_to_id("<pad>"), reduction='none').view(targets.size())
        return loss.sum(dim=0)

    def loss(self, losses):
        return losses['rec']

    def autoenc(self, inputs, targets, is_train=False):
        _, _, _, logits = self(inputs, is_train)
        return {'rec': self.loss_rec(logits, targets).mean()}

    def step(self, losses):
        self.opt.zero_grad()
        losses['loss'].backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        #nn.utils.clip_grad_norm_(self.parameters(), clip)
        self.opt.step()

    def nll_is(self, inputs, targets, m):
        """compute negative log-likelihood by importance sampling:
           p(x;theta) = E_{q(z|x;phi)}[p(z)p(x|z;theta)/q(z|x;phi)]
        """
        mu, logvar = self.encode(inputs)
        tmp = []
        for _ in range(m):
            z = reparameterize(mu, logvar)
            logits, _ = self.decode(z, inputs)
            v = log_prob(z, torch.zeros_like(z), torch.zeros_like(z)) - \
                self.loss_rec(logits, targets) - log_prob(z, mu, logvar)
            tmp.append(v.unsqueeze(-1))
        ll_is = torch.logsumexp(torch.cat(tmp, 1), 1) - np.log(m)
        return -ll_is
   