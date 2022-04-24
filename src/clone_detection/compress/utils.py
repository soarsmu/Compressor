import os
import json
import torch
import random
import multiprocessing
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors, normalizers

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def BPE(texts, vocab_size, file_path, logger):
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Lowercase()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<s>", "<pad>", "</s>", "<unk>"]
    )

    tokenizer.train_from_iterator(texts, trainer)
    folder = "/".join(file_path.split("/")[:-1])
    tokenizer_path = os.path.join(folder, "BPE" + "_" + str(vocab_size) + ".json")
    tokenizer.save(tokenizer_path, pretty=True)
    logger.info("Creating vocabulary to file %s", tokenizer_path)
    
    return tokenizer

class DistilledDataset(Dataset):
    def __init__(self, args, vocab_size, file_path, logger):
        postfix = file_path.split("/")[-1].split(".")[0]
        self.examples = []
        logger.info("Creating features from file at %s ", file_path)

        url_to_code = {}
        folder = "/".join(file_path.split("/")[:-1])
        cache_file_path = os.path.join(folder, "cached_{}.bin".format(postfix+"_dis_"+str(vocab_size)))

        try:
            self.examples = torch.load(cache_file_path)
            logger.info("Loading features from cached file %s", cache_file_path)
        except:
            with open("/".join(file_path.split("/")[:-1])+"/data.jsonl") as f:
                for line in f:
                    line = line.strip()
                    js = json.loads(line)
                    url_to_code[js["idx"]] = js["func"]

            data = []
            with open(file_path) as f:
                for line in f:
                    line = line.strip()
                    url1, url2, label = line.split("\t")
                    if url1 not in url_to_code or url2 not in url_to_code:
                        continue
                    if label == "0":
                        label = 0
                    else:
                        label = 1
                    data.append((url1, url2, label))

            # preds = np.load(os.path.join(folder, "preds_"+postfix+".npy")).astype(int).tolist()
            preds = np.zeros(len(data)).astype(int).tolist()
                        # print(len(preds))
            # print(len(data))
            assert len(data) == len(preds)
            mp_data = []
            for d, pred in tqdm(zip(data, preds)):
                mp_data.append((d, args, url_to_code, pred))

            if "train_sampled" in postfix:
                mp_data = random.sample(mp_data, int(len(mp_data)*0.1))

            tokenizer_path = os.path.join(folder, "BPE" + "_" + str(vocab_size) + ".json")
            
            if os.path.exists(tokenizer_path):
                tokenizer = Tokenizer.from_file(tokenizer_path)
                logger.info("Loading vocabulary from file %s", tokenizer_path)
            else:
                texts = []
                for d in mp_data:
                    texts.append(" ".join(url_to_code[d[0][0]].split()))
                    texts.append(" ".join(url_to_code[d[0][1]].split()))
                tokenizer = BPE(texts, vocab_size, file_path, logger)

            _mp_data = []
            for d in mp_data:
                lst = list(d)
                lst.append(tokenizer)
                _mp_data.append(tuple(lst))
    
            pool = multiprocessing.Pool(multiprocessing.cpu_count())
            self.examples = pool.map(preprocess, tqdm(_mp_data, total=len(_mp_data)))
            torch.save(self.examples, cache_file_path)
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label), torch.tensor(self.examples[i].pred)


def preprocess(item):
    d, args, url_to_code, pred, tokenizer = item
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

    return InputFeatures(source_tokens, source_ids, d[2], pred)


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
                 label,
                 pred
                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label = label
        self.pred = pred