import os
import sys
import copy
import torch
import random
import logging
import hashlib
import warnings
import argparse
import numpy as np

from tqdm import tqdm
from thop import profile
from torchinfo import summary
from utils import GATextDataset, TextDataset

from predefined.models import biLSTM, biGRU, CodeBERT
from sklearn.metrics import recall_score, precision_score, f1_score
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup, RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer

warnings.filterwarnings("ignore")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

logger = logging.getLogger(__name__)

def train(model, lr, train_loader, eval_loader, teacher_preds):
    num_steps = len(train_loader) * 10
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_steps*0.1,
                                                num_training_steps=num_steps)
    dev_best_acc = 0

    for epoch in range(10):
        model.train()
        tr_num = 0
        train_loss = 0

        logger.info('Epoch [{}/{}]'.format(epoch + 1, 10))
        bar = tqdm(train_loader, total=len(train_loader))
        bar.set_description("Train")
        for step, batch in enumerate(bar):
            texts = batch[0].to("cuda")
            labels = batch[1].to("cuda")
            loss, _ = model(texts, labels)

            loss.backward()
            train_loss += loss.item()
            tr_num += 1

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        dev_results = evaluate(model, eval_loader, teacher_preds)
        dev_acc = dev_results["agreements"]
        if dev_acc >= dev_best_acc:
            dev_best_acc = dev_acc

    return dev_best_acc

def evaluate(model, test_loader, teacher_preds):
    model.eval()
    predict_all = []
    labels_all = []

    with torch.no_grad():
        bar = tqdm(test_loader, total=len(test_loader))
        bar.set_description("Evaluation")
        for batch in bar:
            texts = batch[0].to("cuda")        
            label = batch[1].to("cuda")
            prob = model(texts)

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
        "eval_f1": float(f1),
        "agreements": np.sum(teacher_preds==preds)
    }
    logger.info(results)
    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--population_size", default=10, type=int, required=True)
    parser.add_argument("--generation_size", default=20, type=int, required=True)
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--model_dir", default="./", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=42,
                        help="random seed for initialization")

    args = parser.parse_args()
    search_space = {
        "model_arch": ["biGRU", "biLSTM"],
        "encoding": ["token", "subtoken", "BPE"],
        "vocab_size": [*range(1000, 51000, 1000)],
        "input_dim": [*range(1, 769)],
        "hidden_dim": [*range(1, 769)],
        "n_layers": [*range(1, 13)],
        "lr": [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4]
    }

    logger.info("***Start GA search for %d generations and %d population***" %
          (args.generation_size, args.population_size))

    # {'encoding': 'subtoken', 'hidden_dim': 114, 'input_dim': 367, 'lr': 2e-05, 'model_arch': 'biLSTM', 'n_layers': 2, 'vocab_size': 36000}

    train_dataset = GATextDataset(args, 1000, "BPE", args.train_data_file, logger)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    eval_dataset = GATextDataset(args, 1000, "BPE", args.eval_data_file, logger)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=8, pin_memory=True)

    model = biGRU(1000, 208, 48, 2, 12)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'{total_params:,} total parameters.')
    logger.info(f'{total_params*4/1e6} MB model size')

    inputs = torch.randint(1000, (1, 400))
    flops, params = profile(model, (inputs, ), verbose=True)
    
    summary(model, (1, 400), dtypes=[torch.long], verbose=2,
    col_width=16,
    col_names=["kernel_size", "output_size", "num_params", "mult_adds"],
    row_settings=["var_names"],)
    logger.info(params)
    logger.info(flops/1e6)
    logger.info(params*4/1e6)
    torch.save(model.state_dict(), "./model.bin")
    exit()
    model.to("cuda")
    teacher_preds = np.load("teacher_preds.npy")
    agreements = train(model, 1e-3, train_dataloader, eval_dataloader, teacher_preds)



if __name__ == "__main__":
    main()


