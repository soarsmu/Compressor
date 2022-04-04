import os
import torch
import torch.nn as nn
import logging
import argparse
import warnings
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from torchinfo import summary

from sklearn.metrics import recall_score, precision_score, f1_score
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup, RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def teacher_predict(model, args, loader):
    model.eval()
    model.load_state_dict(torch.load("../checkpoint/model.bin"))
    model.to(args.device)
    logits = []
    with torch.no_grad():
        bar = tqdm(loader, total=len(loader))
        for batch in bar:
            inputs = batch[2].to(args.device)
            logit = model(inputs)
            logits.append(logit)
    return logits

def student_train(T_model, S_model, args, train_loader, test_loader):
    try:
        logger.info("Loading Teacher Model's Logits from {}".format("./logits/train_logits_"+ str(args.vocab_size) + ".bin"))
        t_train_logits = torch.load("./logits/train_logits_"+ str(args.vocab_size) + ".bin")
    except:
        logger.info("Creating Teacher Model's Logits.")
        t_train_logits = teacher_predict(T_model, args, train_loader)
        os.makedirs("./logits", exist_ok=True)
        torch.save(t_train_logits, "./logits/train_logits_"+ str(args.vocab_size) + ".bin")

    total_params = sum(p.numel() for p in S_model.parameters())
    logger.info(f'{total_params:,} total parameters.')
    logger.info(f'{total_params*4/1e6} MB model size')

    # summary(S_model, (1, 400), dtypes=[torch.long], verbose=2,
    # col_width=16,
    # col_names=["kernel_size", "output_size", "num_params", "mult_adds"],
    # row_settings=["var_names"],)
    # exit()
    num_steps = len(train_loader) * args.epochs
    
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in S_model.named_parameters(
        ) if not any(nd in n for nd in no_decay)]}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_steps*0.1,
                                                num_training_steps=num_steps)
    dev_best_acc = 0

    for epoch in range(args.epochs):
        S_model.train()
        tr_num = 0
        train_loss = 0

        logger.info('Epoch [{}/{}]'.format(epoch + 1, args.epochs))
        bar = tqdm(train_loader, total=len(train_loader))
        bar.set_description("Train")
        for step, batch in enumerate(bar):
            texts = batch[0].to(args.device)    
            label = batch[1].to(args.device)

            s_logits = S_model(texts)

            if args.loss_func == "ce":
                loss = ce_loss_func(s_logits, t_train_logits[step], label, args.alpha, args.temperature)
            elif args.loss_func == "mse":
                loss = mse_loss_func(s_logits, t_train_logits[step], label, args.alpha, args.normalized)
            loss.backward()
            train_loss += loss.item()
            tr_num += 1

            scheduler.step()
            optimizer.step()
            optimizer.zero_grad()

        dev_results = student_evaluate(args, S_model, test_loader)
        dev_acc = dev_results["eval_acc"]
        if dev_acc >= dev_best_acc:
            dev_best_acc = dev_acc
            # os.makedirs("./best/" + args.std_model + "/" + str(args.size) + "/" + str(args.alpha), exist_ok=True)
            output_dir = os.path.join(args.model_dir, "best")
            os.makedirs(output_dir, exist_ok=True)
            torch.save(S_model.state_dict(), os.path.join(output_dir, "model.bin"))
            logger.info("New best model found and saved.")
        else:
            output_dir = os.path.join(args.model_dir, "recent")
            os.makedirs(output_dir, exist_ok=True)
            torch.save(S_model.state_dict(), os.path.join(output_dir, "model.bin"))
        
        logger.info("Train Loss: {0}, Val Acc: {1}, Val Precision: {2}, Val Recall: {3}, Val F1: {4}".format(train_loss/tr_num, dev_results["eval_acc"], dev_results["eval_precision"], dev_results["eval_recall"], dev_results["eval_f1"]))


def student_evaluate(args, S_model, test_loader):
    S_model.eval()
    predict_all = []
    labels_all = []

    with torch.no_grad():
        bar = tqdm(test_loader, total=len(test_loader))
        bar.set_description("Evaluation")
        for batch in bar:
            texts = batch[0].to(args.device)        
            label = batch[1].to(args.device)
            logits = S_model(texts)
            prob = F.softmax(logits)

            predict_all.append(prob.cpu().numpy())
            labels_all.append(label.cpu().numpy())

    predict_all = np.concatenate(predict_all, 0)
    labels_all = np.concatenate(labels_all, 0)

    preds = predict_all[:, 0] > 0.5
    recall = recall_score(labels_all, preds)
    precision = precision_score(labels_all, preds)
    f1 = f1_score(labels_all, preds)
    results = {
        "eval_acc": np.mean(labels_all==preds),
        "eval_precision": float(precision),
        "eval_recall": float(recall),
        "eval_f1": float(f1)
    }

    return results

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

def train_and_score(genes, dataset):
    print(genes.geneparam.items())
    S_model = biGRU(genes.geneparam["vocab_size"], genes.geneparam["input_dim"], genes.geneparam["hidden_dim"], 2, genes.geneparam["n_layers"])
    S_model.to("cuda")
    total_params = sum(p.numel() for p in S_model.parameters())
    logger.info(f'{total_params:,} total parameters.')
    logger.info(f'{total_params*4/1e6} MB model size')

    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    tokenizer.do_lower_case = True
    train_dataset = DistilledDataset(400, tokenizer, genes.geneparam["vocab_size"], dataset)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=16)

    # summary(S_model, (1, 400), dtypes=[torch.long], verbose=2,
    # col_width=16,
    # col_names=["kernel_size", "output_size", "num_params", "mult_adds"],
    # row_settings=["var_names"],)
    # exit()
    num_steps = len(train_dataloader) * 1
    
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in S_model.named_parameters(
        ) if not any(nd in n for nd in no_decay)]}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=genes.geneparam["lr"], eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_steps*0.1,
                                                num_training_steps=num_steps)

    for epoch in range(1):
        S_model.train()
        tr_num = 0
        train_loss = 0

        logger.info('Epoch [{}/{}]'.format(epoch + 1, 15))
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        bar.set_description("Train")
        for step, batch in enumerate(bar):
            texts = batch[0].to("cuda")    
            label = batch[1].to("cuda")

            s_logits = S_model(texts)

            loss = loss_func(s_logits, label, genes.geneparam["alpha"], genes.geneparam["temperature"])
            loss.backward()
            train_loss += loss.item()
            tr_num += 1

            scheduler.step()
            optimizer.step()
            optimizer.zero_grad()

    return train_loss/tr_num

import os
import json
import torch
import random
import logging
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset
from tokenizers import ByteLevelBPETokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__name__)


class DistilledDataset(Dataset):
    def __init__(self, block_size, teacher_tokenizer, vocab_size=10000, file_path=None):
        postfix = file_path.split('/')[-1].split('.')[0]
        self.examples = []
        logger.info("Creating features from file at %s ", file_path)

        folder = '/'.join(file_path.split('/')[:-1])
        cache_file_path = os.path.join(folder, 'cached_{}.bin'.format(postfix+"_dis_"+str(vocab_size)))

        try:
            self.examples = torch.load(cache_file_path)
            logger.info("Loading features from cached file %s", cache_file_path)
        except:
            data = []
            with open(file_path) as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            data = data[:1000]
            if os.path.exists("./tokenizer_"+str(vocab_size)):
                logger.info("Loading vocabulary from file %s", "./tokenizer_"+str(vocab_size))
                tokenizer = ByteLevelBPETokenizer.from_file("./tokenizer_"+str(vocab_size)+"/vocab.json", "./tokenizer_"+str(vocab_size)+"/merges.txt")
            else:
                logger.info("Creating vocabulary to file %s", "./tokenizer_"+str(vocab_size))
                tokenizer = ByteLevelBPETokenizer(lowercase=True)
                texts = [" ".join(d["func"].split()) for d in data]
                tokenizer.train_from_iterator(texts, vocab_size=vocab_size, show_progress=False, special_tokens=["<s>", "<pad>", "</s>", "<unk>"])
                os.makedirs("./tokenizer_"+str(vocab_size), exist_ok=True)
                tokenizer.save_model("./tokenizer_"+str(vocab_size))

            logger.info("Creating features to %s", cache_file_path)
            for d in tqdm(data):
                code = " ".join(d["func"].split())
                source_ids = tokenizer.encode(code).ids[:block_size-2]
                source_ids = [tokenizer.token_to_id("<s>")]+source_ids+[tokenizer.token_to_id("</s>")]
                padding_length = block_size - len(source_ids)
                source_ids += [tokenizer.token_to_id("<pad>")] * padding_length
                self.examples.append((InputFeatures(code, source_ids, d["target"]), convert_examples_to_features(d, teacher_tokenizer)))

            torch.save(self.examples, cache_file_path)
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i][0].input_ids), torch.tensor(self.examples[i][0].label), torch.tensor(self.examples[i][1].input_ids)


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYHTONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class InputFeatures(object):

    def __init__(self,
                 input_tokens,
                 input_ids,
                 label
                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label = label


def convert_examples_to_features(data, tokenizer):
    code = " ".join(data["func"].split())
    code_tokens = tokenizer.tokenize(code)[:400-2]
    source_tokens = [tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = 400 - len(source_ids)
    source_ids += [tokenizer.pad_token_id]*padding_length
    return InputFeatures(source_tokens, source_ids, data["target"])


import torch
import torch.nn as nn
import torch.nn.functional as F


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

def loss_func(std_logits, labels, alpha=0.9, temperature=2.0):
    labels = labels.long()

    loss = F.cross_entropy(std_logits, (1 - labels))

    # Equivalent to cross_entropy for soft labels, from https://github.com/huggingface/transformers/blob/50792dbdcccd64f61483ec535ff23ee2e4f9e18d/examples/distillation/distiller.py#L330

    return loss


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
