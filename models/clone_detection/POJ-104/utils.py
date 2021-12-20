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
        postfix = file_path.split('/')[-1].split('.')[0]
        self.examples = []
        data = []
        logger.info("Creating features from file at %s ", file_path)

        folder = '/'.join(file_path.split('/')[:-1])
        cache_file_path = os.path.join(folder, 'cached_{}.bin'.format(postfix))

        try:
            self.examples = torch.load(cache_file_path)
            logger.info("Loading features from cached file %s",
                        cache_file_path)
        except:
            with open(file_path) as f:
                for line in f:
                    line = line.strip()
                    data.append(json.loads(line))

            for d in tqdm(data):
                self.examples.append(
                    convert_examples_to_features(d, tokenizer, args))

            torch.save(self.examples, cache_file_path)

        self.label_examples = {}
        for item in self.examples:
            if item.label not in self.label_examples:
                self.label_examples[item.label] = []
            self.label_examples[item.label].append(item)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        label = self.examples[i].label
        index = self.examples[i].index
        labels = list(self.label_examples)
        labels.remove(label)

        while True:
            shuffle_example = random.sample(self.label_examples[label], 1)[0]
            if shuffle_example.index != index:
                p_example = shuffle_example
                break

        n_example = random.sample(
            self.label_examples[random.sample(labels, 1)[0]], 1)[0]

        return (torch.tensor(self.examples[i].input_ids), torch.tensor(p_example.input_ids),
                torch.tensor(n_example.input_ids), torch.tensor(label))


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
                 index,
                 label
                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.index = index
        self.label = label


def convert_examples_to_features(data, tokenizer, args):

    code = " ".join(data["code"].split())
    code_tokens = tokenizer.tokenize(code)[:args.block_size-2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id]*padding_length
    return InputFeatures(source_tokens, source_ids, data["index"], int(data["label"]))
