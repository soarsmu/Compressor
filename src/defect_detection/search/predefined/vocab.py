import os
import re
import json
import time
import string
import pickle
import logging
import argparse

import multiprocessing as mp

manager = mp.Manager()
q_to_store = manager.Queue()

from tqdm import tqdm
from spiral import ronin
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from tokenizers import ByteLevelBPETokenizer, Tokenizer, models, pre_tokenizers, decoders, trainers, processors, normalizers


class Token_Encoder(object):
    def __init__(self, vocab_size, encoding, file_path, logger):
        self.vocab_size = vocab_size
        self.logger = logger
        self.file_path = file_path
        self.encoding = encoding
        folder = "/".join(file_path.split("/")[:-1])
        try:
            tokenizer_path = os.path.join(folder, "token_encoder", encoding + "_" + str(vocab_size) + ".json")
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
        trainer = trainers.WordLevelTrainer(vocab_size=self.vocab_size, special_tokens=["<s>", "<pad>", "</s>", "<unk>"])
        tokenizer.train_from_iterator(texts, trainer)
        folder = "/".join(self.file_path.split("/")[:-1])
        os.makedirs(os.path.join(folder, "token_encoder"), exist_ok=True)
        tokenizer_path = os.path.join(folder, "token_encoder", self.encoding + "_" + str(self.vocab_size) + ".json")
        tokenizer.save(tokenizer_path, pretty=True)
        self.logger.info("Creating vocabulary to file %s", tokenizer_path)
        return tokenizer

    def subtoken(self, texts):
        tokenizer = Tokenizer(models.WordLevel(unk_token="<unk>"))
        tokenizer.normalizer = normalizers.Lowercase()
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(vocab_size=self.vocab_size, special_tokens=["<s>", "<pad>", "</s>", "<unk>"])
        tokenizer.train_from_iterator(texts, trainer)
        folder = "/".join(self.file_path.split("/")[:-1])
        tokenizer_path = os.path.join(folder, "token_encoder", self.encoding + "_" + str(self.vocab_size) + ".json")
        os.makedirs(os.path.join(folder, "token_encoder"), exist_ok=True)
        tokenizer.save(tokenizer_path, pretty=True)
        self.logger.info("Creating vocabulary to file %s", tokenizer_path)
        return tokenizer

    def BPE(self, texts):
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.normalizer = normalizers.Lowercase()
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=["<s>", "<pad>", "</s>", "<unk>"]
        )

        tokenizer.train_from_iterator(texts, trainer)
        folder = "/".join(self.file_path.split("/")[:-1])
        tokenizer_path = os.path.join(folder, "token_encoder", self.encoding + "_" + str(self.vocab_size) + ".json")
        os.makedirs(os.path.join(folder, "token_encoder"), exist_ok=True)
        tokenizer.save(tokenizer_path, pretty=True)
        self.logger.info("Creating vocabulary to file %s", tokenizer_path)
        return tokenizer
