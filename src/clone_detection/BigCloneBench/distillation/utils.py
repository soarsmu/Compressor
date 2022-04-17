import os
import json
import torch
import random
import logging
import numpy as np
import multiprocessing

from tqdm import tqdm
from torch.utils.data import Dataset
from tokenizers import ByteLevelBPETokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    def __init__(self, teacher_tokenizer, args, file_path=None):
        postfix = file_path.split('/')[-1].split('.')[0]
        self.examples = []
        index_filename = file_path
        logger.info("Creating features from file at %s ", index_filename)
        url_to_code = {}

        folder = '/'.join(file_path.split('/')[:-1])
        cache_file_path = os.path.join(folder, 'cached_{}.bin'.format(postfix+"_dis_"+str(args.vocab_size)))

        try:
            self.examples = torch.load(cache_file_path)
            logger.info("Loading features from cached file %s", cache_file_path)
        except:
            with open('/'.join(index_filename.split('/')[:-1])+'/data.jsonl') as f:
                for line in f:
                    line = line.strip()
                    js = json.loads(line)
                    url_to_code[js['idx']] = js['func']

            data = []
            with open(index_filename) as f:
                for line in f:
                    line = line.strip()
                    url1, url2, label = line.split('\t')
                    if url1 not in url_to_code or url2 not in url_to_code:
                        continue
                    if label == '0':
                        label = 0
                    else:
                        label = 1
                    data.append((url1, url2, label))

            if "test" not in postfix:
                data = random.sample(data, int(len(data)*0.1))

            if os.path.exists("./tokenizer_"+str(args.vocab_size)):
                logger.info("Loading vocabulary from file %s", "./tokenizer_"+str(args.vocab_size))
                tokenizer = ByteLevelBPETokenizer.from_file("./tokenizer_"+str(args.vocab_size)+"/vocab.json", "./tokenizer_"+str(args.vocab_size)+"/merges.txt")
            else:
                logger.info("Creating vocabulary to file %s", "./tokenizer_"+str(args.vocab_size))
                tokenizer = ByteLevelBPETokenizer(lowercase=True)

                texts = []
                for d in data:
                    texts.append(" ".join(url_to_code[d[0]].split()))
                    texts.append(" ".join(url_to_code[d[1]].split()))

                tokenizer.train_from_iterator(texts, vocab_size=args.vocab_size, show_progress=False, special_tokens=["<s>", "<pad>", "</s>", "<unk>"])
                os.makedirs("./tokenizer_"+str(args.vocab_size), exist_ok=True)
                tokenizer.save_model("./tokenizer_"+str(args.vocab_size))

            mp_data = []
            for d in data:
                mp_data.append((d, tokenizer, teacher_tokenizer, args, url_to_code))

            pool = multiprocessing.Pool(multiprocessing.cpu_count())
            self.examples = pool.map(preprocess, tqdm(mp_data, total=len(mp_data)))
            torch.save(self.examples, cache_file_path)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i][0].input_ids), torch.tensor(self.examples[i][0].label), torch.tensor(self.examples[i][1].input_ids)

def load_and_cache_examples(args, tokenizer, evaluate=False, test=False):
    dataset = TextDataset(tokenizer, args, file_path=args.test_data_file if test else (
        args.eval_data_file if evaluate else args.train_data_file))
    return dataset

def preprocess(item):
    d, tokenizer, teacher_tokenizer, args, url_to_code = item
    code1 = " ".join(url_to_code[d[0]].split())
    code2 = " ".join(url_to_code[d[1]].split())
    code1_ids = tokenizer.encode(code1).ids[:args.block_size-2]
    code2_ids = tokenizer.encode(code2).ids[:args.block_size-2]
    code1_ids = [tokenizer.token_to_id("<s>")]+code1_ids+[tokenizer.token_to_id("</s>")]
    code2_ids = [tokenizer.token_to_id("<s>")]+code2_ids+[tokenizer.token_to_id("</s>")]
    padding_length = args.block_size - len(code1_ids)
    code1_ids += [tokenizer.token_to_id("<pad>")] * padding_length
    padding_length = args.block_size - len(code2_ids)
    code2_ids += [tokenizer.token_to_id("<pad>")] * padding_length

    source_tokens = code1 + code2
    source_ids = code1_ids + code2_ids

    return (InputFeatures(source_tokens, source_ids, d[2]), convert_examples_to_features(teacher_tokenizer.tokenize(code1), teacher_tokenizer.tokenize(code2), d[2], teacher_tokenizer, args.block_size))

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
                 label
                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label = label


def convert_examples_to_features(code1_tokens, code2_tokens, label, tokenizer, block_size):

    code1_tokens = code1_tokens[:block_size-2]
    code1_tokens = [tokenizer.cls_token]+code1_tokens+[tokenizer.sep_token]
    code2_tokens = code2_tokens[:block_size-2]
    code2_tokens = [tokenizer.cls_token]+code2_tokens+[tokenizer.sep_token]

    code1_ids = tokenizer.convert_tokens_to_ids(code1_tokens)
    padding_length = block_size - len(code1_ids)
    code1_ids += [tokenizer.pad_token_id]*padding_length

    code2_ids = tokenizer.convert_tokens_to_ids(code2_tokens)
    padding_length = block_size - len(code2_ids)
    code2_ids += [tokenizer.pad_token_id]*padding_length

    source_tokens = code1_tokens+code2_tokens
    source_ids = code1_ids+code2_ids
    return InputFeatures(source_tokens, source_ids, label)
