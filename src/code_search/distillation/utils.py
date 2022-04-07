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

        folder = '/'.join(file_path.split('/')[:-1])
        cache_file_path = os.path.join(folder, 'cached_{}.bin'.format(postfix+"_dis_"+str(args.vocab_size)))

        try:
            self.examples = torch.load(cache_file_path)
            logger.info("Loading features from cached file %s", cache_file_path)
        except:
            data = []
            with open(file_path) as f:
                for line in f:
                    line = line.strip()
                    js = json.loads(line)
                    data.append(js)
                    if len(data) > 10000:
                        break

            if os.path.exists("./tokenizer_"+str(args.vocab_size)):
                logger.info("Loading vocabulary from file %s", "./tokenizer_"+str(args.vocab_size))
                tokenizer = ByteLevelBPETokenizer.from_file("./tokenizer_"+str(args.vocab_size)+"/vocab.json", "./tokenizer_"+str(args.vocab_size)+"/merges.txt")
            else:
                logger.info("Creating vocabulary to file %s", "./tokenizer_"+str(args.vocab_size))
                tokenizer = ByteLevelBPETokenizer(lowercase=True)

                texts = []
                for d in data:
                    if "code_tokens" in d:
                        texts.append(" ".join(d["code_tokens"]))
                    else:
                        texts.append(" ".join(d["function_tokens"]))
                    texts.append(" ".join(d["docstring_tokens"]))

                tokenizer.train_from_iterator(texts, vocab_size=args.vocab_size, show_progress=False, special_tokens=["<s>", "<pad>", "</s>", "<unk>"])
                os.makedirs("./tokenizer_"+str(args.vocab_size), exist_ok=True)
                tokenizer.save_model("./tokenizer_"+str(args.vocab_size))

            mp_data = []
            for d in data:
                mp_data.append((d, tokenizer, teacher_tokenizer, args))

            pool = multiprocessing.Pool(multiprocessing.cpu_count())
            self.examples = pool.map(preprocess, tqdm(mp_data, total=len(mp_data)))
            torch.save(self.examples, cache_file_path)
             
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):   
        return torch.tensor(self.examples[i][0].code_ids), torch.tensor(self.examples[i][0].nl_ids), torch.tensor(self.examples[i][1].code_ids), torch.tensor(self.examples[i][1].nl_ids)
            

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_tokens,
                 code_ids,
                 nl_tokens,
                 nl_ids

    ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids

def preprocess(item):
    d, tokenizer, teacher_tokenizer, args = item
    if "code_tokens" in d:
        code1 = " ".join(d["code_tokens"])
    else:
        code1 = " ".join(d["function_tokens"])
    code2 = " ".join(d["docstring_tokens"])
    code1_ids = tokenizer.encode(code1).ids[:args.block_size-2]
    code2_ids = tokenizer.encode(code2).ids[:args.block_size-2]
    code1_ids = [tokenizer.token_to_id("<s>")]+code1_ids+[tokenizer.token_to_id("</s>")]
    code2_ids = [tokenizer.token_to_id("<s>")]+code2_ids+[tokenizer.token_to_id("</s>")]
    padding_length = args.block_size - len(code1_ids)
    code1_ids += [tokenizer.token_to_id("<pad>")] * padding_length
    padding_length = args.block_size - len(code2_ids)
    code2_ids += [tokenizer.token_to_id("<pad>")] * padding_length

    return (InputFeatures(code1, code1_ids, code2, code2_ids), convert_examples_to_features(d, teacher_tokenizer, args))

        
def convert_examples_to_features(js,tokenizer,args):
    #code
    if 'code_tokens' in js:
        code=' '.join(js['code_tokens'])
    else:
        code=' '.join(js['function_tokens'])
    code_tokens = tokenizer.tokenize(code)[:args.block_size-2]
    code_tokens = [tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.block_size - len(code_ids)
    code_ids+=[tokenizer.pad_token_id]*padding_length
    
    nl=' '.join(js['docstring_tokens'])
    nl_tokens = tokenizer.tokenize(nl)[:args.block_size-2]
    nl_tokens = [tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.block_size - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id]*padding_length    
    
    return InputFeatures(code_tokens, code_ids, nl_tokens, nl_ids)


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
