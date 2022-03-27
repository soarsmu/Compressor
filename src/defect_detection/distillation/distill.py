from itertools import count
import os
import torch
import logging
import argparse
import warnings
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from utils import set_seed, DistilledDataset
from model import biLSTM, biGRU, Roberta, ce_loss_func, mse_loss_func
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

    args.model_name = "microsoft/codebert-base"
    config = RobertaConfig.from_pretrained(args.model_name)
    config.num_labels = 2
    
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    tokenizer.do_lower_case = True

    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)

    teacher_model = Roberta(RobertaForSequenceClassification.from_pretrained(args.model_name, config=config))
   
    n_labels = 2

    if args.std_model == "biLSTM":
        student_model = biLSTM(args.vocab_size, args.input_dim, args.hidden_dim, n_labels, args.n_layers)
    elif args.std_model == "biGRU":
        student_model = biGRU(args.vocab_size, args.input_dim, args.hidden_dim, n_labels, args.n_layers)
    elif args.std_model == "Roberta":
        std_config = RobertaConfig.from_pretrained(args.model_name)
        std_config.num_labels = n_labels
        std_config.hidden_size = args.hidden_dim
        std_config.max_position_embeddings = args.hidden_dim + 2
        std_config.vocab_size = args.vocab_size
        std_config.num_attention_heads = 8
        std_config.num_hidden_layers = args.n_layers
        student_model = Roberta(RobertaForSequenceClassification(std_config))

    if args.do_train:
        train_dataset = DistilledDataset(args, tokenizer, args.vocab_size, args.train_data_file)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    
    eval_dataset = DistilledDataset(args, tokenizer, args.vocab_size, args.eval_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=8, pin_memory=True)
    
    student_model.to(args.device)

    if args.do_train:
        student_train(teacher_model, student_model, args, train_dataloader, eval_dataloader)

    if args.do_eval:
        model_dir = os.path.join(args.model_dir, args.choice, "model.bin")
        student_model.load_state_dict(torch.load(model_dir))
        student_model.to(args.device)
        eval_res = student_evaluate(args, student_model, eval_dataloader)
        logger.info("Acc: {0}, Precision: {1}, Recall: {2}, F1: {3}".format(eval_res["eval_acc"], eval_res["eval_precision"], eval_res["eval_recall"], eval_res["eval_f1"]))


if __name__ == "__main__":
    main()
