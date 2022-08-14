import os
import json
import torch
import random
import logging
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        logger.info("Creating features from file at %s ", file_path)

        data = []
        with open(file_path) as f:
            for line in f:
                data.append(json.loads(line.strip()))

        for d in tqdm(data):
            self.examples.append(convert_examples_to_features(d, tokenizer, args))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label)

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


def convert_examples_to_features(data, tokenizer, args):
    code = " ".join(data["func"].split())
    code_tokens = tokenizer.tokenize(code)[:args.block_size-2]
    source_tokens = [tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id]*padding_length
    return InputFeatures(source_tokens, source_ids, data["target"])
