import os
from unittest import result
import torch
import logging
import argparse
import warnings
import numpy as np
import torch.nn.functional as F
import collections
from tqdm import tqdm
from torchinfo import summary
from sklearn.cluster import KMeans
from utils import set_seed, DistilledDataset
    
from model import DAE, biLSTM, biGRU, Roberta, ce_loss_func, mse_loss_func
from sklearn.metrics import recall_score, precision_score, f1_score
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup, RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer
from tokenizers import ByteLevelBPETokenizer

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

class AverageMeter(object):
    def __init__(self):
        self.clear()

    def clear(self):
        self.cnt = 0
        self.sum = 0
        self.avg = 0

    def update(self, val, n=1):
        self.cnt += n
        self.sum += val * n
        self.avg = self.sum / self.cnt

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--model_dir", default="./", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--choice", default="best", type=str,
                        help="Model to test")
    parser.add_argument("--vocab_size", default=10000, type=int,
                        help="Vocabulary Size.")
    parser.add_argument("--input_dim", default=512, type=int,
                        help="Embedding Dim.")
    parser.add_argument("--hidden_dim", default=512, type=int,
                        help="Hidden dim of student model.")
    parser.add_argument("--n_layers", default=1, type=int,
                        help="Num of layers in student model.")
    parser.add_argument("--std_model", default="biLSTM", type=str, required=True,
                        help="Student Model Type.")
    parser.add_argument("--loss_func", default="ce", type=str,
                        help="Loss Function Type.")
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--alpha", default=0.5, type=float,
                        help="Weighting factor in loss fucntion.")
    parser.add_argument("--temperature", default=2.0, type=float,
                        help="Temperature factor in loss fucntion.")
    parser.add_argument("--normalized", default=True, type=bool,
                        help="Whether to normalize loss in mse.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=42,
                        help="random seed for initialization")

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.per_gpu_train_batch_size = args.train_batch_size//args.n_gpu
    args.per_gpu_eval_batch_size = args.eval_batch_size//args.n_gpu

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO)
    logger.info("Device: %s, n_gpu: %s", args.device, args.n_gpu)

    set_seed(args.seed)

    model_name = "microsoft/codebert-base"
    config = RobertaConfig.from_pretrained(model_name)
    config.num_labels = 2
    
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    tokenizer.do_lower_case = True

    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)


    train_dataset = DistilledDataset(args, tokenizer, args.vocab_size, args.train_data_file)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    eval_dataset = DistilledDataset(args, tokenizer, args.vocab_size, args.eval_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=8, pin_memory=True)
    tokenizer = ByteLevelBPETokenizer.from_file("./tokenizer_"+str(args.vocab_size)+"/vocab.json", "./tokenizer_"+str(args.vocab_size)+"/merges.txt")

    args.dropout = 0.5
    args.dim_z = 128
    model = DAE(tokenizer, args)
    model.to(args.device)

    num_steps = len(train_dataloader) * args.epochs
    best_val_loss = 300
    for epoch in range(args.epochs):
        model.train()
        meters = collections.defaultdict(lambda: AverageMeter())
        tr_num = 0
        train_loss = 0

        logger.info('Epoch [{}/{}]'.format(epoch + 1, args.epochs))
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        bar.set_description("Train")
        for step, batch in enumerate(bar):
            texts = batch[0].to(args.device)    
            label = batch[1].to(args.device)

            losses = model.autoenc(texts, texts, is_train=True)
            losses['loss'] = model.loss(losses)
            model.step(losses)
            for k, v in losses.items():
                meters[k].update(v.item())

            if (step + 1) % 1000 == 0:
                log_output = '| epoch {:3d} | {:5d}/{:5d} batches |'.format(
                    epoch + 1, step + 1, len(train_dataloader))
                for k, meter in meters.items():
                    log_output += ' {} {:.2f},'.format(k, meter.avg)
                    meter.clear()
                logging.info(log_output)

        dev_results = evaluate(model, eval_dataloader)
        print(dev_results['loss'].avg)
        if dev_results['loss'].avg < best_val_loss:
            best_val_loss =  dev_results['loss'].avg
            # print(best_val_loss)
            torch.save(model.state_dict(), "./model.bin")
        # dev_acc = dev_results["eval_acc"]
        # if dev_acc >= dev_best_acc:
        #     dev_best_acc = dev_acc
        #     # os.makedirs("./best/" + args.std_model + "/" + str(args.size) + "/" + str(args.alpha), exist_ok=True)
        #     output_dir = os.path.join(args.model_dir, "best")
        #     os.makedirs(output_dir, exist_ok=True)
        #     torch.save(model.state_dict(), os.path.join(output_dir, "model.bin"))
        #     logger.info("New best model found and saved.")
        # else:
        #     output_dir = os.path.join(args.model_dir, "recent")
        #     os.makedirs(output_dir, exist_ok=True)
        #     torch.save(model.state_dict(), os.path.join(output_dir, "model.bin"))
        
        # logger.info("Train Loss: {0}, Val Acc: {1}, Val Precision: {2}, Val Recall: {3}, Val F1: {4}".format(train_loss/tr_num, dev_results["eval_acc"], dev_results["eval_precision"], dev_results["eval_recall"], dev_results["eval_f1"]))

def evaluate(model, batches):
    model.eval()
    meters = collections.defaultdict(lambda: AverageMeter())
    with torch.no_grad():
        bar = tqdm(batches, total=len(batches))
        bar.set_description("Train")
        for step, batch in enumerate(bar):
            texts = batch[0].to("cuda")    
            label = batch[1].to("cuda")
            losses = model.autoenc(texts, texts)
            for k, v in losses.items():
                meters[k].update(v.item(), texts.size(1))
    loss = model.loss({k: meter.avg for k, meter in meters.items()})
    meters['loss'].update(loss)
    print(meters.items())
    return meters


if __name__ == "__main__":
    main()