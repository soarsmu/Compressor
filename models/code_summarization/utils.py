import os
import torch
import random
import numpy as np


class Example(object):

    def __init__(
            self,
            idx,
            source,
            target,
            ):
        self.idx = idx
        self.source = source
        self.target = target


def read_examples(filename):

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


def convert_examples_to_features(examples, tokenizer, args):
    features = []
    for example_index, example in enumerate(examples):

        source_tokens = tokenizer.tokenize(example.source)[:args.block_size-2]
        source_tokens = [tokenizer.cls_token] + \
            source_tokens+[tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        source_mask = [1] * (len(source_tokens))
        padding_length = args.block_size - len(source_ids)
        source_ids += [tokenizer.pad_token_id]*padding_length
        source_mask += [0]*padding_length

        target_tokens = tokenizer.tokenize(example.target)[:args.block_size-2]
        target_tokens = [tokenizer.cls_token] + \
            target_tokens+[tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] * len(target_ids)
        padding_length = args.block_size - len(target_ids)
        target_ids += [tokenizer.pad_token_id]*padding_length
        target_mask += [0]*padding_length

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


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYHTONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
