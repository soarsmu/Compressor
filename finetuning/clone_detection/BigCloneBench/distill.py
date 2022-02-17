import os
import time
import torch
import logging
import argparse
import warnings
import numpy as np

from tqdm import tqdm
from sklearn import metrics
from gpu_mem_track import MemTracker
from datetime import timedelta
from model import Model, biLSTM
from utils import set_seed, load_and_cache_examples
from sklearn.metrics import recall_score, precision_score, f1_score
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AdamW, get_linear_schedule_with_warmup, RobertaConfig, RobertaModel, RobertaTokenizer, BertTokenizer

tracker = MemTracker()
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

def get_time_dif(start_time):

    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

# 损失函数
def get_loss(t_logits, s_logits, label, a, T):
    loss1 = torch.nn.CrossEntropyLoss()
    loss2 = torch.nn.MSELoss()
    loss = a * loss1(s_logits, label) + T * loss2(t_logits, s_logits)
    # print(loss1(s_logits, label),loss2(t_logits, s_logits))
    return loss


def teacher_predict(model, args, loader):
    model.eval()
    checkpoint_prefix = "checkpoint/model.bin"
    output_dir = os.path.join(args.output_dir, "{}".format(checkpoint_prefix))
    model.load_state_dict(torch.load(output_dir))
    model.to(args.device)
    t_logits = []
    bar = tqdm(loader, total=len(loader))
    for batch in bar:
        inputs = batch[0].to(args.device)
        with torch.no_grad():
            logit = model(inputs)
        t_logits.append(logit)
    return t_logits


def student_train(T_model, S_model, args, train_loader, test_loader):
    t_train_logits = teacher_predict(T_model, args, train_loader)
    t_test_logits = teacher_predict(T_model, args, test_loader)
    total_params = sum(p.numel() for p in S_model.parameters())
    print(f'{total_params:,} total parameters.')
    # optimizer = torch.optim.SGD(S_model.parameters(), lr=0.01, momentum=0.9)
    optimizer = torch.optim.Adam(
            S_model.parameters(),
            lr=0.001,
            weight_decay=0,
            betas=(0.9, 0.999), eps=1e-9
        )
    num_steps = len(train_loader) * 5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_steps, eta_min=0.0)
    total_batch = 0
    tra_best_loss = float('inf')
    dev_best_loss = float('inf')
    S_model.train()
    start_time = time.time()
    for epoch in range(10):
        print('Epoch [{}/{}]'.format(epoch + 1, 10))
        bar = tqdm(train_loader, total=len(train_loader))
        for step, batch in enumerate(bar):
            texts = batch[0].to(args.device)        
            label = batch[1].to(args.device)
            optimizer.zero_grad()
            s_logits = S_model(texts)
            loss = get_loss(t_train_logits[step], s_logits, label.long(), 1, 2)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if total_batch % 1500 == 0:
                cur_pred = torch.squeeze(s_logits, dim=1)
                train_acc = metrics.accuracy_score(label.cpu().numpy(), torch.max(cur_pred, 1)[1].cpu().numpy())
                dev_loss, dev_acc = student_evaluate(S_model, args, t_test_logits, test_loader)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(S_model.state_dict(), "./model.bin")
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                S_model.train()
            total_batch += 1
    print(student_evaluate(S_model, args, t_test_logits, test_loader))
    
def student_evaluate(S_model, args, t_logits, test_loader):
    S_model.eval()
    predict_all = []
    labels_all = []
    loss_total = 0
    with torch.no_grad():
        bar = tqdm(test_loader, total=len(test_loader))
        for step, batch in enumerate(bar):
            texts = batch[0].to(args.device)
            label = batch[1].to(args.device)
            s_logits = S_model(texts)
            loss = get_loss(t_logits[step], s_logits, label.long(), 1, 2)
            loss_total += loss

            cur_pred = torch.squeeze(s_logits, dim=1)
            predic = torch.max(cur_pred, 1)[1].cpu().numpy()
            label = label.data.cpu().numpy()
            labels_all = np.append(labels_all, label)
            predict_all = np.append(predict_all, predic)
    acc = metrics.accuracy_score(labels_all, predict_all)
    return loss_total/len(test_loader), acc

def student_evaluate_v2(S_model, args, test_loader):
    S_model.eval()
    predict_all = []
    labels_all = []
    loss_total = 0
    # with torch.no_grad():
    bar = tqdm(test_loader, total=len(test_loader))
    for step, batch in enumerate(bar):
        texts = batch[0].to(args.device)   
        label = batch[1].to(args.device)
        # torch.cuda.empty_cache()
        check_memory()
        s_logits = S_model(texts)
        # torch.cuda.empty_cache()
        check_memory()
        cur_pred = torch.squeeze(s_logits, dim=1)
        predic = torch.max(cur_pred, 1)[1].cpu().numpy()
        label = label.data.cpu().numpy()
        labels_all = np.append(labels_all, label)
        predict_all = np.append(predict_all, predic)
    acc = metrics.accuracy_score(labels_all, predict_all)
    return loss_total/len(test_loader), acc

def train(args, model, tokenizer):

    train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    args.max_steps = args.epoch*len(train_dataloader)
    args.save_steps = len(train_dataloader)
    args.warmup_steps = len(train_dataloader)
    args.logging_steps = len(train_dataloader)
    args.num_train_epochs = args.epoch
    model.to(args.device)

    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters(
        ) if not any(nd in n for nd in no_decay)]}
    ]

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step = 0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_f1 = 0
    model.zero_grad()

    for idx in range(0, int(args.num_train_epochs)):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            inputs = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            model.train()
            loss, _ = model(inputs, labels)

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(train_loss/tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                avg_loss = round(np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging_loss = tr_loss
                    tr_nb = global_step

                if args.save_steps > 0 and global_step % args.save_steps == 0:

                    if args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer, eval_when_training=True)

                    logger.info("  "+"*"*20)
                    logger.info("  Current F1:%s", round(results["eval_f1"], 4))
                    logger.info("  Best F1:%s", round(best_f1, 4))
                    logger.info("  "+"*"*20)

                    if results["eval_f1"] >= best_f1:
                        best_f1 = results["eval_f1"]

                        checkpoint_prefix = 'checkpoint'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)

                        output_dir = os.path.join(output_dir, '{}'.format('model.bin'))
                        torch.save(model.module.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)
                    else:
                        logger.info("Model checkpoint are not saved")


def check_memory():
    logger.info('GPU memory: %.1f' % (torch.cuda.memory_allocated() // 1024 ** 2))

def evaluate(args, model, tokenizer, eval_when_training=False):

    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=args.eval_batch_size, num_workers=8, pin_memory=True)

    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    logits = []
    labels = []

    bar = tqdm(eval_dataloader, total=len(eval_dataloader))
    for batch in bar:
        bar.set_description("evaluation")
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        with torch.no_grad():
            _, logit = model(inputs, label)
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)

    y_preds = logits[:, 1] > 0.5
    recall = recall_score(labels, y_preds)
    precision = precision_score(labels, y_preds)
    f1 = f1_score(labels, y_preds)
    result = {
        "eval_recall": float(recall),
        "eval_acc": float(precision),
        "eval_f1": float(f1)
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


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
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epoch', type=int, default=42,
                        help="random seed for initialization")

    args = parser.parse_args()

    args.device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    # args.device = torch.device("cpu")
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
    tokenizer_s = BertTokenizer.from_pretrained("bert-base-uncased")

    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)

    model = Model(RobertaModel.from_pretrained(args.model_name, config=config), config, args)
    
    model.eval()
    checkpoint_prefix = "checkpoint/model.bin"
    output_dir = os.path.join(args.output_dir, "{}".format(checkpoint_prefix))
    model.load_state_dict(torch.load(output_dir))
    model.to(args.device)

    print(model)
    exit()

    student_model = biLSTM()
    
    
    train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)
    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=16)
    
    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=32, num_workers=8, pin_memory=True)

    # student_model.load_state_dict(torch.load("./model.bin"))
    student_model.to(args.device)
    # torch.cuda.empty_cache()
    # check_memory()
    # for step, batch in enumerate(eval_dataloader):
    #     texts = batch[0].to(args.device)
    # with profile(activities=[ProfilerActivity.CUDA],
    #     profile_memory=True, record_shapes=True) as prof:
    # # with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
    #     with torch.no_grad():
    #         student_model(texts)
            
    # print(prof.key_averages().table())

    # print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
    
    # student_train(model, student_model, args, train_dataloader, eval_dataloader)
    # logger.info("Training/evaluation parameters %s", args)

    # if args.do_train:
    #     train(args, model, tokenizer)

    # if args.do_eval:
    #     checkpoint_prefix = "checkpoint/model.bin"
    #     output_dir = os.path.join(
    #         args.output_dir, "{}".format(checkpoint_prefix))
    #     model.load_state_dict(torch.load(output_dir))
    #     model.to(args.device)
    #     evaluate(args, model, tokenizer)
    
    student_train(model, student_model, args, train_dataloader, eval_dataloader)




if __name__ == "__main__":
    main()
