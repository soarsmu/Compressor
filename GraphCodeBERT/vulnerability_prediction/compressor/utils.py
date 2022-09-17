import os
import json
import torch
import random
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
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
    tokenizer_path = os.path.join(
        folder, "BPE" + "_" + str(vocab_size) + ".json")
    tokenizer.save(tokenizer_path, pretty=True)
    logger.info("Creating vocabulary to file %s", tokenizer_path)

    return tokenizer


class DistilledDataset(Dataset):
    def __init__(self, args, vocab_size, file_path, logger):
        postfix = file_path.split("/")[-1].split(".")[0]
        self.examples = []
        logger.info("Creating features from file at %s ", file_path)

        folder = "/".join(file_path.split("/")[:-1])

        data = []
        with open(file_path) as f:
            for line in f:
                data.append(json.loads(line.strip()))
        tokenizer_path = os.path.join(
            folder, "BPE" + "_" + str(vocab_size) + ".json")

        if os.path.exists(tokenizer_path):
            tokenizer = Tokenizer.from_file(tokenizer_path)
            logger.info("Loading vocabulary from file %s", tokenizer_path)
        else:
            texts = [" ".join(d["func"].split()) for d in data]
            tokenizer = BPE(texts, vocab_size, file_path, logger)

        for d in tqdm(data):
            code = " ".join(d["func"].split())
            source_ids = tokenizer.encode(code).ids[:args.block_size-2]
            source_ids = [tokenizer.token_to_id(
                "<s>")]+source_ids+[tokenizer.token_to_id("</s>")]
            padding_length = args.block_size - len(source_ids)
            source_ids += [tokenizer.token_to_id("<pad>")] * padding_length
            if "train" in postfix:
                self.examples.append(
                    (InputFeatures(code, source_ids, d["pred"], d["pred"], d["soft_label"])))
            else:
                self.examples.append(
                    (InputFeatures(code, source_ids, d["target"])))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label), torch.tensor(self.examples[i].pred), torch.tensor(self.examples[i].soft_label)


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
                 pred=0,
                 soft_label=[0.1, 0.1]
                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label = label
        self.pred = pred
        self.soft_label = soft_label


# deprecated this class
class Token_Encoder(object):
    def __init__(self, vocab_size, encoding, file_path, logger):
        self.vocab_size = vocab_size
        self.logger = logger
        self.file_path = file_path
        self.encoding = encoding
        folder = "/".join(file_path.split("/")[:-1])
        try:
            tokenizer_path = os.path.join(
                folder, "token_encoder", encoding + "_" + str(vocab_size) + ".json")
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
            self.logger.info("Loading vocabulary from file %s", tokenizer_path)
        except:
            data = []
            with open(self.file_path) as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            texts = [" ".join(d["func"].split()) for d in data]

            if encoding == "token":
                self.tokenizer = self.token(texts)
            elif encoding == "subtoken":
                self.tokenizer = self.subtoken(texts)
            elif encoding == "BPE":
                self.tokenizer = self.BPE(texts)

    def token(self, texts):
        tokenizer = Tokenizer(models.WordLevel(unk_token="<unk>"))
        tokenizer.normalizer = normalizers.Lowercase()
        tokenizer.pre_tokenizer = Whitespace()
        trainer = trainers.WordLevelTrainer(vocab_size=self.vocab_size, special_tokens=[
                                            "<s>", "<pad>", "</s>", "<unk>"])
        tokenizer.train_from_iterator(texts, trainer)
        folder = "/".join(self.file_path.split("/")[:-1])
        os.makedirs(os.path.join(folder, "token_encoder"), exist_ok=True)
        tokenizer_path = os.path.join(
            folder, "token_encoder", self.encoding + "_" + str(self.vocab_size) + ".json")
        tokenizer.save(tokenizer_path, pretty=True)
        self.logger.info("Creating vocabulary to file %s", tokenizer_path)
        return tokenizer

    def subtoken(self, texts):
        tokenizer = Tokenizer(models.WordLevel(unk_token="<unk>"))
        tokenizer.normalizer = normalizers.Lowercase()
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(vocab_size=self.vocab_size, special_tokens=[
                                   "<s>", "<pad>", "</s>", "<unk>"])
        tokenizer.train_from_iterator(texts, trainer)
        folder = "/".join(self.file_path.split("/")[:-1])
        tokenizer_path = os.path.join(
            folder, "token_encoder", self.encoding + "_" + str(self.vocab_size) + ".json")
        os.makedirs(os.path.join(folder, "token_encoder"), exist_ok=True)
        tokenizer.save(tokenizer_path, pretty=True)
        self.logger.info("Creating vocabulary to file %s", tokenizer_path)
        return tokenizer

    def BPE(self, texts):
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.normalizer = normalizers.Lowercase()
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=True)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=["<s>", "<pad>", "</s>", "<unk>"]
        )

        tokenizer.train_from_iterator(texts, trainer)
        folder = "/".join(self.file_path.split("/")[:-1])
        tokenizer_path = os.path.join(
            folder, "token_encoder", self.encoding + "_" + str(self.vocab_size) + ".json")
        os.makedirs(os.path.join(folder, "token_encoder"), exist_ok=True)
        tokenizer.save(tokenizer_path, pretty=True)
        self.logger.info("Creating vocabulary to file %s", tokenizer_path)
        return tokenizer
