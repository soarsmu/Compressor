import os
import torch
import logging
import argparse
import warnings
import numpy as np

import torch.nn as nn
from tqdm import tqdm
from model import Seq2Seq
from bleu import compute_bleu
from utils import set_seed, read_examples, convert_examples_to_features
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, TensorDataset
from transformers import AdamW, get_linear_schedule_with_warmup, RobertaConfig, RobertaModel, RobertaTokenizer


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def train(args, model, tokenizer):

    train_examples = read_examples(args.train_data_file)
    train_features = convert_examples_to_features(
        train_examples, tokenizer, args)
    all_source_ids = torch.tensor(
        [f.source_ids for f in train_features], dtype=torch.long)
    all_source_mask = torch.tensor(
        [f.source_mask for f in train_features], dtype=torch.long)
    all_target_ids = torch.tensor(
        [f.target_ids for f in train_features], dtype=torch.long)
    all_target_mask = torch.tensor(
        [f.target_mask for f in train_features], dtype=torch.long)
    train_dataset = TensorDataset(
        all_source_ids, all_source_mask, all_target_ids, all_target_mask)

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
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps*0.1,
                                                num_training_steps=args.max_steps)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Num epoch = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    nb_tr_examples, nb_tr_steps, tr_loss, global_step, best_ppl = 0, 0, 0, 0, 10000

    for idx in range(0, int(args.num_train_epochs)):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        for batch in bar:

            batch = tuple(t.to(args.device) for t in batch)
            source_ids, source_mask, target_ids, target_mask = batch
            model.train()
            loss, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                               target_ids=target_ids, target_mask=target_mask)

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            tr_loss += loss.item()
            tr_num += 1
            bar.set_description("epoch {} loss {}".format(
                idx, round(tr_loss/tr_num, 5)))

            nb_tr_examples += source_ids.size(0)
            nb_tr_steps += 1
            loss.backward()

            if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            if args.save_steps > 0 and global_step % args.save_steps == 0:

                if args.evaluate_during_training:
                    results = evaluate_during_training(args, model, tokenizer)

                logger.info("  "+"*"*20)
                logger.info("  Current PPL:%s", round(results["eval_ppl"], 2))
                logger.info("  Best PPL:%s", round(best_ppl, 2))
                logger.info("  "+"*"*20)

                if results["eval_ppl"] < best_ppl:
                    best_ppl = results["eval_ppl"]

                    checkpoint_prefix = 'checkpoint'
                    output_dir = os.path.join(
                        args.output_dir, "{}".format(checkpoint_prefix))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    output_dir = os.path.join(
                        output_dir, '{}'.format('model.bin'))
                    torch.save(model.module.state_dict(), output_dir)
                    logger.info("Saving model checkpoint to %s", output_dir)
                else:
                    logger.info("Model checkpoint are not saved")


def evaluate_during_training(args, model, tokenizer):

    eval_examples = read_examples(args.eval_data_file)
    eval_features = convert_examples_to_features(
        eval_examples, tokenizer, args)
    all_source_ids = torch.tensor(
        [f.source_ids for f in eval_features], dtype=torch.long)
    all_source_mask = torch.tensor(
        [f.source_mask for f in eval_features], dtype=torch.long)
    all_target_ids = torch.tensor(
        [f.target_ids for f in eval_features], dtype=torch.long)
    all_target_mask = torch.tensor(
        [f.target_mask for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(
        all_source_ids, all_source_mask, all_target_ids, all_target_mask)

    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(
        eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # if args.n_gpu > 1:
    #     model = torch.nn.DataParallel(model)

    logger.info("\n***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    eval_loss, tokens_num = 0, 0
    bar = tqdm(eval_dataloader, total=len(eval_dataloader))

    for batch in bar:
        batch = tuple(t.to(args.device) for t in batch)
        source_ids, source_mask, target_ids, target_mask = batch
        with torch.no_grad():
            _, loss, num = model(source_ids=source_ids, source_mask=source_mask,
                                 target_ids=target_ids, target_mask=target_mask)
            eval_loss += loss.sum().item()
            tokens_num += num.sum().item()
    eval_loss = eval_loss / tokens_num

    result = {
        "eval_ppl": np.exp(eval_loss)
    }

    return result


def evaluate(args, model, tokenizer):

    eval_examples = read_examples(args.eval_data_file)
    eval_features = convert_examples_to_features(
        eval_examples, tokenizer, args)
    all_source_ids = torch.tensor(
        [f.source_ids for f in eval_features], dtype=torch.long)
    all_source_mask = torch.tensor(
        [f.source_mask for f in eval_features], dtype=torch.long)
    all_target_ids = torch.tensor(
        [f.target_ids for f in eval_features], dtype=torch.long)
    all_target_mask = torch.tensor(
        [f.target_mask for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(
        all_source_ids, all_source_mask, all_target_ids, all_target_mask)

    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(
        eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("\n***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    bar = tqdm(eval_dataloader, total=len(eval_dataloader))
    p = []
    for batch in bar:
        batch = tuple(t.to(args.device) for t in batch)
        source_ids, source_mask = batch[0], batch[1]
        with torch.no_grad():
            preds = model(source_ids=source_ids, source_mask=source_mask)
            for pred in preds:
                t = pred[0].cpu().numpy()
                t = list(t)
                if 0 in t:
                    t = t[:t.index(0)]
                text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                p.append(text)

    accs = []
    refs = []
    golds = []

    for ref, gold in zip(p, eval_examples):
        accs.append(ref == gold.target)
        refs.append(ref.strip().split())
        golds.append(gold.target.strip().split())
    
    dev_bleu = compute_bleu(refs, golds, max_order=4, smooth=True)
    result = {
        "eval_bleu": dev_bleu,
        "eval_acc": np.mean(accs)
    }

    logger.info("***** Eval results *****")
    for key in result.keys():
        logger.info("  %s = %s", key, str(round(result[key], 4)))
        
    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data_file", default=None, type=str,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default="./", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--beam_size", default=5, type=int)
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
    args.n_gpu = torch.cuda.device_count()

    args.per_gpu_train_batch_size = args.train_batch_size//args.n_gpu
    args.per_gpu_eval_batch_size = args.eval_batch_size//args.n_gpu

    set_seed(args.seed)

    args.model_name = "microsoft/codebert-base"
    config = RobertaConfig.from_pretrained(args.model_name)

    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)

    encoder = RobertaModel.from_pretrained(args.model_name, config=config)
    decoder_layer = nn.TransformerDecoderLayer(
        d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model = Seq2Seq(encoder=encoder, decoder=decoder, config=config,
                    beam_size=args.beam_size, max_length=args.block_size,
                    sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)

    if args.do_train:
        train(args, model, tokenizer)

    if args.do_eval:
        checkpoint_prefix = "checkpoint/model.bin"
        output_dir = os.path.join(
            args.output_dir, "{}".format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        evaluate(args, model, tokenizer)


if __name__ == "__main__":
    main()
