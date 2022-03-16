import os
import torch
import logging
import argparse
import warnings
import numpy as np

from tqdm import tqdm
import torch.nn.functional as F
from model import biLSTM, biGRU, Roberta, loss_func
from utils import set_seed, DistilledDataset
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
            with torch.no_grad():
                logit = model(inputs)
                logits.append(logit)

    return logits

def student_train(T_model, S_model, args, train_loader, test_loader):
    try:
        t_train_logits = torch.load("./train_logits.bin")
        t_test_logits = torch.load("./test_logits.bin")
    except:
        t_train_logits = teacher_predict(T_model, args, train_loader)
        t_test_logits = teacher_predict(T_model, args, test_loader)
        torch.save(t_train_logits, "./train_logits.bin")
        torch.save(t_test_logits, "./test_logits.bin")

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
        for step, batch in enumerate(bar):
            texts = batch[0].to(args.device)    
            label = batch[1].to(args.device)

            s_logits = S_model(texts)

            loss = loss_func(s_logits, t_train_logits[step], label)
            loss.backward()
            train_loss += loss.item()
            tr_num += 1

            scheduler.step()
            optimizer.step()
            optimizer.zero_grad()

        dev_acc = student_evaluate(args, S_model, test_loader)
        if dev_acc >= dev_best_acc:
            dev_best_acc = dev_acc
            os.makedirs("./best/" + args.std_model, exist_ok=True)
            torch.save(S_model.state_dict(), os.path.join("./best/", args.std_model, "model.bin"))
        else:
            os.makedirs("./recent/" + args.std_model, exist_ok=True)
            torch.save(S_model.state_dict(), os.path.join("./recent/", args.std_model, "model.bin"))
        
        logger.info("Train Loss: {0}, Val Acc: {1}".format(train_loss/tr_num, dev_acc))


def student_evaluate(args, S_model, test_loader):
    S_model.eval()
    predict_all = []
    labels_all = []

    with torch.no_grad():
        bar = tqdm(test_loader, total=len(test_loader))
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
    eval_acc = np.mean(labels_all==preds)
    return eval_acc


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default="./", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--choice", default="best", type=str,
                        help="Model to test")
    parser.add_argument("--vocab_size", default="", type=int,
                        help="Vocabulary Size.")
    parser.add_argument("--std_model", default="biLSTM", type=str,
                        help="Student Model Type.")
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")

    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=0.5, type=float,
                        help="Max gradient norm.")
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
   
    input_dim = 256
    hidden_dim = 512
    n_labels = 2
    n_layers = 1

    if args.std_model == "biLSTM":
        student_model = biLSTM(args.vocab_size, input_dim, hidden_dim, n_labels, n_layers)
    elif args.std_model == "biGRU":
        student_model = biGRU(args.vocab_size, input_dim, hidden_dim, n_labels, n_layers)
    elif args.std_model == "Roberta":
        std_config = RobertaConfig.from_pretrained(args.model_name)
        std_config.num_labels = n_labels
        std_config.hidden_size = hidden_dim
        std_config.max_position_embeddings = hidden_dim + 2
        std_config.vocab_size = args.vocab_size
        std_config.num_attention_heads = 8
        std_config.num_hidden_layers = n_layers
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
        model_dir = "./" + args.choice + "/" + args.std_model + "/" + "model.bin"
        student_model.load_state_dict(torch.load(model_dir))
        student_model.to(args.device)
        eval_acc = student_evaluate(args, student_model, eval_dataloader)
        logger.info("Acc: {0}".format(eval_acc))


if __name__ == "__main__":
    main()
