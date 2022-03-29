import os
import torch
import logging
import argparse
import warnings
import numpy as np

from tqdm import tqdm
from utils import set_seed, TextDataset
from model import Roberta, biLSTM, biGRU, ce_loss_func, mse_loss_func

from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup, RobertaConfig, RobertaModel, RobertaTokenizer

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def teacher_predict(model, args, loader):
    model.eval()
    model.load_state_dict(torch.load("../checkpoint/model.bin"))
    model.to(args.device)
    logits = []
    bar = tqdm(loader, total=len(loader))
    with torch.no_grad():
        for batch in bar:
            code_inputs = batch[2].to(args.device)    
            nl_inputs = batch[3].to(args.device)
            logit, code_vec, nl_vec = model(code_inputs, nl_inputs)
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
    dev_best_mrr = 0

    for epoch in range(args.epochs):
        S_model.train()
        tr_num = 0
        train_loss = 0

        logger.info('Epoch [{}/{}]'.format(epoch + 1, args.epochs))
        bar = tqdm(train_loader, total=len(train_loader))
        bar.set_description("Train")
        for step, batch in enumerate(bar):
            texts_1 = batch[0].to(args.device)        
            texts_2 = batch[1].to(args.device)

            s_logits, code_vec, nl_vec = S_model(texts_1, texts_2)
            if args.loss_func == "ce":
                loss = ce_loss_func(s_logits, t_train_logits[step], args.alpha, args.temperature)
            elif args.loss_func == "mse":
                loss = mse_loss_func(s_logits, t_train_logits[step], args.alpha, args.normalized)
            loss.backward()
            train_loss += loss.item()
            tr_num += 1

            scheduler.step()
            optimizer.step()
            optimizer.zero_grad()

        dev_results = student_evaluate(args, S_model, test_loader)
        dev_mrr = dev_results["eval_mrr"]
        if dev_mrr >= dev_best_mrr:
            dev_best_mrr = dev_mrr
            # os.makedirs("./best/" + args.std_model + "/" + str(args.size) + "/" + str(args.alpha), exist_ok=True)
            output_dir = os.path.join(args.model_dir, "best")
            os.makedirs(output_dir, exist_ok=True)
            torch.save(S_model.state_dict(), os.path.join(output_dir, "model.bin"))
            logger.info("New best model found and saved.")
        else:
            output_dir = os.path.join(args.model_dir, "recent")
            os.makedirs(output_dir, exist_ok=True)
            torch.save(S_model.state_dict(), os.path.join(output_dir, "model.bin"))
        
        logger.info("Train Loss: {0}, Val MRR: {1}".format(train_loss/tr_num, dev_results["eval_mrr"]))
    

def student_evaluate(args, S_model, test_loader):
    S_model.eval()
    code_vecs = [] 
    nl_vecs = []
    
    with torch.no_grad():
        bar = tqdm(test_loader, total=len(test_loader))
        bar.set_description("Evaluation")
        for batch in bar:
            code_inputs = batch[0].to(args.device)    
            nl_inputs = batch[1].to(args.device)
            with torch.no_grad():
                _, code_vec, nl_vec = S_model(code_inputs, nl_inputs)
                code_vecs.append(code_vec.cpu().numpy())
                nl_vecs.append(nl_vec.cpu().numpy())

    code_vecs = np.concatenate(code_vecs, 0)
    nl_vecs = np.concatenate(nl_vecs, 0)

    scores = np.matmul(nl_vecs, code_vecs.T)
    ranks = []
    for i in range(len(scores)):
        score = scores[i, i]
        rank = 1
        for j in range(len(scores)):
            if i != j and scores[i, j] >= score:
                rank += 1
        ranks.append(1/rank)    
       
    result = {
        "eval_mrr": float(np.mean(ranks))
    }

    return result


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

    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    tokenizer.do_lower_case = True

    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)

    teacher_model = Roberta(RobertaModel.from_pretrained(args.model_name, config=config), config, args)
    
    n_labels = 2

    if args.std_model == "biLSTM":
        student_model = biLSTM(args.vocab_size, args.input_dim, args.hidden_dim, n_labels, args.n_layers)
    elif args.std_model == "biGRU":
        student_model = biGRU(args.vocab_size, args.input_dim, args.hidden_dim, n_labels, args.n_layers)
    elif args.std_model == "Roberta":
        std_config = RobertaConfig.from_pretrained(args.model_name)
        std_config.num_labels = n_labels
        std_config.hidden_size = args.hidden_dim
        # std_config.max_position_embeddings = args.hidden_dim + 2
        std_config.vocab_size = args.vocab_size
        std_config.num_attention_heads = 8
        std_config.num_hidden_layers = args.n_layers
        student_model = Roberta(RobertaModel(std_config), std_config, args)

    if args.do_train:
        train_dataset = TextDataset(tokenizer, args, args.train_data_file)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    
    eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)
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
        logger.info("MRR: {0}".format(eval_res["eval_mrr"]))


if __name__ == "__main__":
    main()
