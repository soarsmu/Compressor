from lib2to3.pgen2 import token
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
from tokenizers import ByteLevelBPETokenizer, WordLevel, Tokenizer


class Token_Encoder(object):
    def __init__(self, vocab_size, encoder_type, file_path, logger):
        self.vocab_size = vocab_size
        self.logger = logger
        self.file_path = file_path
        try:
            if encoder_type in {"token", "subtoken"}:
                tokenizer_path = os.path.join("./token_encoder", encoder_type + "_" + str(vocab_size) + ".json")
                self.tokenizer = Tokenizer.from_file(tokenizer_path)
                self.logger.info("Loading vocabulary from file %s", tokenizer_path)
            else:
                tokenizer_path = os.path.join("./token_encoder", encoder_type + "_" + str(vocab_size))
                self.tokenizer = ByteLevelBPETokenizer.from_file(tokenizer_path+"/vocab.json", tokenizer_path+"/merges.txt")
                self.logger.info("Loading vocabulary from file %s", tokenizer_path)
        except:
            data = []
            with open(self.file_path) as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            texts = [" ".join(d["func"].split()) for d in data]

            if encoder_type == "token":
                self.tokenizer = self.token(texts)
            elif encoder_type == "subtoken":
                self.tokenizer = self.subtoken(texts)
            elif encoder_type == "BPE":
                self.tokenizer = self.BPE(texts)

    def token(self, texts):
        tokenizer = Tokenizer(WordLevel(unk_token="<unk>", sep_token="</s>", cls_token="<s>", pad_token="<pad>", lowercase=True))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(vocab_size=self.vocab_size)
        tokenizer.train_from_iterator(texts, trainer)
        tokenizer_path = os.path.join("./token_encoder", "token" + "_" + str(self.vocab_size) + ".json")
        tokenizer.save(tokenizer_path)
        self.logger.info("Creating vocabulary to file %s", tokenizer_path)
        return tokenizer

    def subtoken(self, texts):
        tokenizer = Tokenizer(WordLevel(unk_token="<unk>", sep_token="</s>", cls_token="<s>", pad_token="<pad>", lowercase=True))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(vocab_size=self.vocab_size)
        tokenizer.train_from_iterator(texts, trainer)
        tokenizer_path = os.path.join("./token_encoder", "subtoken" + "_" + str(self.vocab_size) + ".json")
        tokenizer.save(tokenizer_path)
        self.logger.info("Creating vocabulary to file %s", tokenizer_path)
        return tokenizer

    def BPE(self, texts):
        tokenizer = ByteLevelBPETokenizer(unk_token="<unk>", sep_token="</s>", cls_token="<s>", pad_token="<pad>", lowercase=True)
        tokenizer.train_from_iterator(texts, vocab_size=self.vocab_size)
        tokenizer_path = os.path.join("./token_encoder", "BPE" + "_" + str(self.vocab_size))
        os.makedirs(tokenizer_path, exist_ok=True)
        tokenizer.save_model(tokenizer_path)
        return tokenizer
