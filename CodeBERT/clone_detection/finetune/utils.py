import os
import json
import torch
import pickle
import random
import logging
import numpy as np
import multiprocessing

from tqdm import tqdm
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        index_filename = file_path
        logger.info("Creating features from file at %s ", index_filename)
        url_to_code = {}

        with open("/".join(index_filename.split("/")[:-1])+"/data.jsonl") as f:
            for line in f:
                line = line.strip()
                js = json.loads(line)
                url_to_code[js["idx"]] = js["func"]

        data = []
        with open(index_filename) as f:
            for line in f:
                line = line.strip()
                url1, url2, label = line.split("\t")
                if url1 not in url_to_code or url2 not in url_to_code:
                    continue
                if label == "0":
                    label = 0
                else:
                    label = 1
                data.append((url1, url2, label, tokenizer,
                             args, url_to_code))

        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        self.examples = pool.map(get_example, tqdm(data, total=len(data)))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item].input_ids), torch.tensor(self.examples[item].label)


def load_and_cache_examples(args, tokenizer, evaluate=False, test=False):
    dataset = TextDataset(tokenizer, args, file_path=args.test_data_file if test else (
        args.eval_data_file if evaluate else args.train_data_file))
    return dataset


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


def convert_examples_to_features(code1_tokens, code2_tokens, label, tokenizer, args):

    code1_tokens = code1_tokens[:args.block_size-2]
    code1_tokens = [tokenizer.cls_token]+code1_tokens+[tokenizer.sep_token]
    code2_tokens = code2_tokens[:args.block_size-2]
    code2_tokens = [tokenizer.cls_token]+code2_tokens+[tokenizer.sep_token]

    code1_ids = tokenizer.convert_tokens_to_ids(code1_tokens)
    padding_length = args.block_size - len(code1_ids)
    code1_ids += [tokenizer.pad_token_id]*padding_length

    code2_ids = tokenizer.convert_tokens_to_ids(code2_tokens)
    padding_length = args.block_size - len(code2_ids)
    code2_ids += [tokenizer.pad_token_id]*padding_length

    source_tokens = code1_tokens+code2_tokens
    source_ids = code1_ids+code2_ids
    return InputFeatures(source_tokens, source_ids, label)


def get_example(item):
    url1, url2, label, tokenizer, args, url_to_code = item

    try:
        code = " ".join(url_to_code[url1].split())
    except:
        code = ""
    code1 = tokenizer.tokenize(code)

    try:
        code = " ".join(url_to_code[url2].split())
    except:
        code = ""
    code2 = tokenizer.tokenize(code)

    return convert_examples_to_features(code1, code2, label, tokenizer, args)
