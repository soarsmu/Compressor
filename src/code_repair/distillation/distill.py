import os
import torch
import random
import logging
import argparse
import numpy as np
from io import open
from itertools import cycle
import torch.nn as nn

from model import Seq2Seq, RNNModel
from tqdm import tqdm
from bleu import _bleu
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, TensorDataset
from transformers import (AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target


def read_examples(filename):
    """Read examples from filename."""
    examples = []
    assert len(filename.split(',')) == 2
    src_filename = filename.split(',')[0]
    trg_filename = filename.split(',')[1]
    idx = 0
    with open(src_filename) as f1, open(trg_filename) as f2:
        for line1, line2 in zip(f1, f2):
            examples.append(
                Example(
                    idx=idx,
                    source=line1.strip(),
                    target=line2.strip(),
                )
            )
            idx += 1
    return examples


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,

                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask


def convert_examples_to_features(examples, tokenizer, args, stage=None):
    features = []
    for example_index, example in enumerate(examples):
        # source
        source_tokens = tokenizer.tokenize(example.source)[
            :args.max_source_length-2]
        source_tokens = [tokenizer.cls_token] + \
            source_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        source_mask = [1] * (len(source_tokens))
        padding_length = args.max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        source_mask += [0]*padding_length

        # target
        if stage == "test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[
                :args.max_target_length-2]
        target_tokens = [tokenizer.cls_token] + \
            target_tokens+[tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] * len(target_ids)
        padding_length = args.max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id]*padding_length
        target_mask += [0]*padding_length

        if example_index < 5:
            if stage == 'train':
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example.idx))

                logger.info("source_tokens: {}".format(
                    [x.replace('\u0120', '_') for x in source_tokens]))
                logger.info("source_ids: {}".format(
                    ' '.join(map(str, source_ids))))
                logger.info("source_mask: {}".format(
                    ' '.join(map(str, source_mask))))

                logger.info("target_tokens: {}".format(
                    [x.replace('\u0120', '_') for x in target_tokens]))
                logger.info("target_ids: {}".format(
                    ' '.join(map(str, target_ids))))
                logger.info("target_mask: {}".format(
                    ' '.join(map(str, target_mask))))

        features.append(
            InputFeatures(
                example_index,
                source_ids,
                target_ids,
                source_mask,
                target_mask,
            )
        )
    return features


def set_seed(args):
    """set random seed."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default="./", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization.")
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

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO)
    args = parser.parse_args()
    logger.info(args)

    args.device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.per_gpu_train_batch_size = args.train_batch_size//args.n_gpu
    args.per_gpu_eval_batch_size = args.eval_batch_size//args.n_gpu

    logger.info("Device: %s, n_gpu: %s", args.device, args.n_gpu)

    set_seed(args)
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    model_name = "microsoft/codebert-base"
    config = RobertaConfig.from_pretrained(model_name)
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    tokenizer.do_lower_case = True

    encoder = RobertaModel.from_pretrained(model_name, config=config)
    decoder_layer = nn.TransformerDecoderLayer(
        d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model = Seq2Seq(encoder=encoder, decoder=decoder, config=config,
                    beam_size=args.beam_size, max_length=args.max_target_length,
                    sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)

    if args.load_model_path is not None:
        logger.info("reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))

    model.to(args.device)

    if args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)

    if args.do_train:
        # Prepare training data loader
        train_examples = read_examples(args.train_filename)
        train_features = convert_examples_to_features(
            train_examples, tokenizer, args, stage='train')
        all_source_ids = torch.tensor(
            [f.source_ids for f in train_features], dtype=torch.long)
        all_source_mask = torch.tensor(
            [f.source_mask for f in train_features], dtype=torch.long)
        all_target_ids = torch.tensor(
            [f.target_ids for f in train_features], dtype=torch.long)
        all_target_mask = torch.tensor(
            [f.target_mask for f in train_features], dtype=torch.long)
        train_data = TensorDataset(
            all_source_ids, all_source_mask, all_target_ids, all_target_mask)

        train_sampler = RandomSampler(train_data)

        train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                      batch_size=args.train_batch_size//args.gradient_accumulation_steps)

        num_train_optimization_steps = args.train_steps

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

        # Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num epoch = %d", num_train_optimization_steps *
                    args.train_batch_size//len(train_examples))

        model.train()
        dev_dataset = {}
        nb_tr_examples, nb_tr_steps, tr_loss, global_step, best_bleu, best_loss = 0, 0, 0, 0, 0, 1e6
        bar = range(num_train_optimization_steps)
        train_dataloader = cycle(train_dataloader)
        eval_flag = True
        for step in bar:
            batch = next(train_dataloader)
            batch = tuple(t.to(args.device) for t in batch)
            source_ids, source_mask, target_ids, target_mask = batch
            loss, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                               target_ids=target_ids, target_mask=target_mask)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            tr_loss += loss.item()
            train_loss = round(
                tr_loss*args.gradient_accumulation_steps/(nb_tr_steps+1), 4)
            if (global_step + 1) % 100 == 0:
                logger.info("  step {} loss {}".format(
                    global_step + 1, train_loss))
            nb_tr_examples += source_ids.size(0)
            nb_tr_steps += 1
            loss.backward()

            if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                # Update parameters
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                eval_flag = True

            if args.do_eval and ((global_step + 1) % args.eval_steps == 0) and eval_flag:
                # Eval model with dev dataset
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0
                eval_flag = False
                if 'dev_loss' in dev_dataset:
                    eval_examples, eval_data = dev_dataset['dev_loss']
                else:
                    eval_examples = read_examples(args.dev_filename)
                    eval_features = convert_examples_to_features(
                        eval_examples, tokenizer, args, stage='dev')
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
                    dev_dataset['dev_loss'] = eval_examples, eval_data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(
                    eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                logger.info("\n***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)

                # Start Evaling model
                model.eval()
                eval_loss, tokens_num = 0, 0
                for batch in eval_dataloader:
                    batch = tuple(t.to(args.device) for t in batch)
                    source_ids, source_mask, target_ids, target_mask = batch

                    with torch.no_grad():
                        _, loss, num = model(source_ids=source_ids, source_mask=source_mask,
                                             target_ids=target_ids, target_mask=target_mask)
                    eval_loss += loss.sum().item()
                    tokens_num += num.sum().item()
                # Pring loss of dev dataset
                model.train()
                eval_loss = eval_loss / tokens_num
                result = {'eval_ppl': round(np.exp(eval_loss), 5),
                          'global_step': global_step+1,
                          'train_loss': round(train_loss, 5)}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  "+"*"*20)

                # save last checkpoint
                last_output_dir = os.path.join(
                    args.output_dir, 'checkpoint-last')
                if not os.path.exists(last_output_dir):
                    os.makedirs(last_output_dir)
                model_to_save = model.module if hasattr(
                    model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(last_output_dir, "model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)
                if eval_loss < best_loss:
                    logger.info("  Best ppl:%s", round(np.exp(eval_loss), 5))
                    logger.info("  "+"*"*20)
                    best_loss = eval_loss
                    # Save best checkpoint for best ppl
                    output_dir = os.path.join(
                        args.output_dir, 'checkpoint-best-ppl')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(
                        model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)

                # Calculate bleu
                if 'dev_bleu' in dev_dataset:
                    eval_examples, eval_data = dev_dataset['dev_bleu']
                else:
                    eval_examples = read_examples(args.dev_filename)
                    eval_examples = random.sample(
                        eval_examples, min(1000, len(eval_examples)))
                    eval_features = convert_examples_to_features(
                        eval_examples, tokenizer, args, stage='test')
                    all_source_ids = torch.tensor(
                        [f.source_ids for f in eval_features], dtype=torch.long)
                    all_source_mask = torch.tensor(
                        [f.source_mask for f in eval_features], dtype=torch.long)
                    eval_data = TensorDataset(all_source_ids, all_source_mask)
                    dev_dataset['dev_bleu'] = eval_examples, eval_data

                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(
                    eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                model.eval()
                p = []
                for batch in eval_dataloader:
                    batch = tuple(t.to(args.device) for t in batch)
                    source_ids, source_mask = batch
                    with torch.no_grad():
                        preds = model(source_ids=source_ids,
                                      source_mask=source_mask)
                        for pred in preds:
                            t = pred[0].cpu().numpy()
                            t = list(t)
                            if 0 in t:
                                t = t[:t.index(0)]
                            text = tokenizer.decode(
                                t, clean_up_tokenization_spaces=False)
                            p.append(text)
                model.train()
                predictions = []
                accs = []
                with open(os.path.join(args.output_dir, "dev.output"), 'w') as f, open(os.path.join(args.output_dir, "dev.gold"), 'w') as f1:
                    for ref, gold in zip(p, eval_examples):
                        predictions.append(str(gold.idx)+'\t'+ref)
                        f.write(ref+'\n')
                        f1.write(gold.target+'\n')
                        accs.append(ref == gold.target)

                dev_bleu = round(_bleu(os.path.join(args.output_dir, "dev.gold"), os.path.join(
                    args.output_dir, "dev.output")), 2)
                logger.info("  %s = %s " % ("bleu-4", str(dev_bleu)))
                logger.info("  %s = %s " %
                            ("xMatch", str(round(np.mean(accs)*100, 4))))
                logger.info("  "+"*"*20)
                if dev_bleu > best_bleu:
                    logger.info("  Best bleu:%s", dev_bleu)
                    logger.info("  "+"*"*20)
                    best_bleu = dev_bleu
                    # Save best checkpoint for best bleu
                    output_dir = os.path.join(
                        args.output_dir, 'checkpoint-best-bleu')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(
                        model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)

    if args.do_test:
        files = []
        if args.dev_filename is not None:
            files.append(args.dev_filename)
        if args.test_filename is not None:
            files.append(args.test_filename)
        for idx, file in enumerate(files):
            logger.info("Test file: {}".format(file))
            eval_examples = read_examples(file)
            eval_features = convert_examples_to_features(
                eval_examples, tokenizer, args, stage='test')
            all_source_ids = torch.tensor(
                [f.source_ids for f in eval_features], dtype=torch.long)
            all_source_mask = torch.tensor(
                [f.source_mask for f in eval_features], dtype=torch.long)
            eval_data = TensorDataset(all_source_ids, all_source_mask)

            # Calculate bleu
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(
                eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

            model.eval()
            p = []
            for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
                batch = tuple(t.to(args.device) for t in batch)
                source_ids, source_mask = batch
                with torch.no_grad():
                    preds = model(source_ids=source_ids,
                                  source_mask=source_mask)
                    for pred in preds:
                        t = pred[0].cpu().numpy()
                        t = list(t)
                        if 0 in t:
                            t = t[:t.index(0)]
                        text = tokenizer.decode(
                            t, clean_up_tokenization_spaces=False)
                        p.append(text)
            model.train()
            predictions = []
            accs = []
            with open(os.path.join(args.output_dir, "test_{}.output".format(str(idx))), 'w') as f, open(os.path.join(args.output_dir, "test_{}.gold".format(str(idx))), 'w') as f1:
                for ref, gold in zip(p, eval_examples):
                    predictions.append(str(gold.idx)+'\t'+ref)
                    f.write(ref+'\n')
                    f1.write(gold.target+'\n')
                    accs.append(ref == gold.target)
            dev_bleu = round(_bleu(os.path.join(args.output_dir, "test_{}.gold".format(str(idx))).format(file), os.path.join(args.output_dir, "test_{}.output".format(str(idx))).format(file)), 2)
            logger.info("  %s = %s " % ("bleu-4", str(dev_bleu)))
            logger.info("  %s = %s " %
                        ("xMatch", str(round(np.mean(accs)*100, 4))))
            logger.info("  "+"*"*20)


if __name__ == "__main__":
    main()
