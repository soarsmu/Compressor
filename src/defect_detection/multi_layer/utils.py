import os
import json
import torch
import random
import logging
import numpy as np

from tqdm import tqdm
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class DistilledDataset(Dataset):
    def __init__(self, args, teacher_tokenizer, vocab_size=10000, file_path=None):
        postfix = file_path.split('/')[-1].split('.')[0]
        self.examples = []
        logger.info("Creating features from file at %s ", file_path)

        folder = '/'.join(file_path.split('/')[:-1])
        cache_file_path = os.path.join(folder, 'cached_{}.bin'.format(postfix+"_dis"))

        try:
            self.examples = torch.load(cache_file_path)
            logger.info("Loading features from cached file %s", cache_file_path)
        except:
            data = []
            with open(file_path) as f:
                for line in f:
                    data.append(json.loads(line.strip()))

            if os.path.exists("./tokenizer_"+str(vocab_size)+".json"):
                tokenizer = Tokenizer.from_file("./tokenizer_"+str(vocab_size)+".json")
            else:
                tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
                trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]"], vocab_size=10000)
                tokenizer.pre_tokenizer = Whitespace()
                texts = [" ".join(d["func"].split()) for d in data]
                tokenizer.train_from_iterator(texts, trainer)
                tokenizer.save("./tokenizer_"+str(vocab_size)+".json")

            for d in tqdm(data):
                code = " ".join(d["func"].split())
                source_ids = tokenizer.encode(code).ids[:args.block_size-2]
                source_ids = [tokenizer.token_to_id("[CLS]")]+source_ids+[tokenizer.token_to_id("[SEP]")]
                padding_length = args.block_size - len(source_ids)
                source_ids += [tokenizer.token_to_id("[PAD]")] * padding_length
                self.examples.append((InputFeatures(code, source_ids, d["target"]), convert_examples_to_features(d, teacher_tokenizer, args)))

            torch.save(self.examples, cache_file_path)
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i][0].input_ids), torch.tensor(self.examples[i][0].label), torch.tensor(self.examples[i][1].input_ids)


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
