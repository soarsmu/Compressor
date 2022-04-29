import os
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
        postfix = file_path.split('/')[-1].split('.')[0]
        self.examples = []
        logger.info("Creating features from file at %s ", file_path)
        url_to_code = {}

        folder = '/'.join(file_path.split('/')[:-1])
        cache_file_path = os.path.join(folder, 'cached_{}.bin'.format(postfix))

        # try:
        #     self.examples = torch.load(cache_file_path)
        #     # self.examples = self.examples[]
        #     logger.info("Loading features from cached file %s", cache_file_path)
        # except:
        with open(file_path, "rb") as fp:
            data = pickle.load(fp)

        mp_data = []
        for d in data:
            mp_data.append((d, tokenizer, args))
        # if "test" not in postfix:
        mp_data = mp_data[:5500]
        # mp_data = random.sample(mp_data, int(len(mp_data)*0.01))

        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        self.examples = pool.map(get_example, tqdm(mp_data, total=len(mp_data)))
            # torch.save(self.examples, cache_file_path)

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
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class InputFeatures(object):

    def __init__(self,
                 input_tokens,
                 input_ids,
                 label = 1
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
    try:
        (code_1, code_2, label), tokenizer, args = item
    except:
        (code_1, code_2), tokenizer, args = item
        label = 1
    try:
        code = " ".join(code_1.split())
    except:
        code = ""
    code1 = tokenizer.tokenize(code)

    try:
        code = " ".join(code_2.split())
    except:
        code = ""
    code2 = tokenizer.tokenize(code)

    return convert_examples_to_features(code1, code2, label, tokenizer, args)
