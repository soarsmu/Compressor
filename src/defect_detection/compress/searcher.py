import os
import sys
import copy
import torch
import random
import logging
import hashlib
import warnings
import argparse
import numpy as np

from tqdm import tqdm
from thop import profile
# from utils import GATextDataset, TextDataset
from torchinfo import summary
from predefined.models import biLSTM, biGRU, CodeBERT
from sklearn.metrics import recall_score, precision_score, f1_score
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup, RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer

warnings.filterwarnings("ignore")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

logger = logging.getLogger(__name__)


class Genome(object):
    def __init__(self, gene_param=None):
        self.fitness = 0.0
        self.gene_param = gene_param

        if not self.gene_param:
            self.hash = 0
        else:
            self.update_hash()
    
    def update_hash(self):
        gene_string = str(self.gene_param["model_arch"])+ \
                        str(self.gene_param["vocab_size"]) + \
                        str(self.gene_param["input_dim"]) + \
                        str(self.gene_param["hidden_dim"]) + \
                        str(self.gene_param["n_layers"]) 
        self.hash = hashlib.md5(gene_string.encode("UTF-8")).hexdigest()

    def mutation(self, search_space):
        mutated_gene = random.choice(list(self.gene_param.keys()))
        current_value = self.gene_param[mutated_gene]
        possible_choices = copy.deepcopy(search_space[mutated_gene])
        possible_choices.remove(current_value)
        self.gene_param[mutated_gene] = random.choice(possible_choices)
        self.update_hash()


class GA_search():
    def __init__(self, args, search_space, retain_chance=0.3, mutate_chance=0.5):
        self.args = args
        self.search_space = search_space
        self.retain_chance = retain_chance
        self.mutate_chance = mutate_chance
        self.population = []

    def is_duplicate(self, new_genome):
        for genome in self.population:
            if new_genome.hash == genome.hash:
                return True

    def initialization(self):
        count = 0

        while count < self.args.population_size:
            gene_param = {}
            for key in self.search_space:
                gene_param[key] = random.choice(self.search_space[key])
            new_genome = Genome(gene_param)
            
            if len(self.population) > 0:
                while self.is_duplicate(new_genome):
                    new_genome.mutation()

            self.population.append(new_genome)
            count += 1
        
        # logger.info(self.population)
    
    def fitness(self, genome):
        # try:
        #     teacher_preds = np.load("teacher_preds.npy")
        # except:
        #     config = RobertaConfig.from_pretrained("microsoft/codebert-base")
        #     config.num_labels = 2
            
        #     tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        #     tokenizer.do_lower_case = True

        #     if self.args.block_size <= 0:
        #         self.args.block_size = tokenizer.max_len_single_sentence
        #     self.args.block_size = min(self.args.block_size, tokenizer.max_len_single_sentence)

        #     teacher_model = CodeBERT(RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", config=config))
        #     total_params = sum(p.numel() for p in teacher_model.parameters())
        #     logger.info(total_params)
            
        #     eval_dataset = TextDataset(tokenizer, self.args, self.args.eval_data_file)
        #     eval_sampler = SequentialSampler(eval_dataset)
        #     eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size, num_workers=8, pin_memory=True)

        #     teacher_model.eval()
        #     teacher_model.load_state_dict(torch.load("../checkpoint/model.bin"))
        #     teacher_model.to("cuda")
        #     teacher_preds = []
        #     bar = tqdm(eval_dataloader, total=len(eval_dataloader))
        #     with torch.no_grad():
        #         for batch in bar:
        #             inputs = batch[0].to("cuda")
        #             preds = teacher_model(inputs)
        #             teacher_preds.append(preds.cpu().numpy())
        #     teacher_preds = np.concatenate(teacher_preds, 0)
        #     teacher_preds = teacher_preds[:, 0] > 0.5
        #     np.save("teacher_preds", teacher_preds)

        model_arch = genome.gene_param["model_arch"]
        vocab_size = genome.gene_param["vocab_size"]
        input_dim = genome.gene_param["input_dim"]
        hidden_dim = genome.gene_param["hidden_dim"]
        n_layers = genome.gene_param["n_layers"]
        n_labels = 2

        if model_arch == "biLSTM":
            model = biLSTM(vocab_size, input_dim, hidden_dim, n_labels, n_layers)
        elif model_arch == "biGRU":
            model = biGRU(vocab_size, input_dim, hidden_dim, n_labels, n_layers)

        # train_dataset = GATextDataset(self.args, vocab_size, encoding, self.args.train_data_file, logger)
        # train_sampler = RandomSampler(train_dataset)
        # train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

        # eval_dataset = GATextDataset(self.args, vocab_size, encoding, self.args.eval_data_file, logger)
        # eval_sampler = SequentialSampler(eval_dataset)
        # eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size, num_workers=8, pin_memory=True)

        # model.to("cuda")
        # agreements = train(model, lr, train_dataloader, eval_dataloader, teacher_preds)
        inputs = torch.randint(vocab_size, (1, 400))
        flops, _ = profile(model, (inputs, ), verbose=False)
        params = sum(p.numel() for p in model.parameters())
        # summary(model, (1, 400), dtypes=[torch.long], verbose=2,
        # col_width=16,
        # col_names=["kernel_size", "output_size", "num_params", "mult_adds"],
        # row_settings=["var_names"],)
        # exit()
        # total_params = sum(p.numel() for p in model.parameters())
        # genome.fitness = agreements * (124647170/total_params)
        # logger.info([agreements)
        # logger.info(flops - params)
        # logger.info(params)
        # genome.fitness = flops - params
        logger.info(flops/1e9 - abs(3 - params*4/1e6))
        logger.info("size %f", params*4.0/1e6)
        # # logger.info(params)
        genome.fitness = flops/1e9 - abs(3 - params*4/1e6)

    def crossover_and_mutation(self, parents):
        children = []
        parent_1, parent_2 = parents
        genome_len = len(self.search_space)
        recomb_loc = random.randint(1, genome_len - 1)

        child_1 = {}
        child_2 = {}

        keys = list(self.search_space)
        keys = sorted(keys)

        for x in range(0, genome_len):
            if x < recomb_loc:
                child_1[keys[x]] = parent_1.gene_param[keys[x]]
                child_2[keys[x]] = parent_2.gene_param[keys[x]]
            else:
                child_1[keys[x]] = parent_2.gene_param[keys[x]]
                child_2[keys[x]] = parent_1.gene_param[keys[x]]

        genome_1 = Genome(child_1)
        genome_2 = Genome(child_2)

        if self.mutate_chance > random.random():
            genome_1.mutation(self.search_space)

        if self.mutate_chance > random.random():
            genome_2.mutation(self.search_space)

        while self.is_duplicate(genome_1):
            genome_1.mutation(self.search_space)

        while self.is_duplicate(genome_2):
            genome_2.mutation(self.search_space)

        children.append(genome_1)
        children.append(genome_2)

        return children

    def generation(self):
        for genome in self.population:
            self.fitness(genome)
        graded_genome = [x for x in sorted(self.population, key=lambda x: x.fitness, reverse=True)]
        logger.info(graded_genome[0].gene_param)
        logger.info(graded_genome[0].fitness)
        retain_length = int(len(graded_genome) * self.retain_chance)
        new_generation = graded_genome[:retain_length]
        desired_length = len(self.population) - len(new_generation)

        children = []
        while len(children) < desired_length:
            parents_id = random.sample(range(len(new_generation)-1), k=2)
            parents = (new_generation[parents_id[0]], new_generation[parents_id[1]])
            babies = self.crossover_and_mutation(parents)

            for baby in babies:
                if len(children) < desired_length:
                    children.append(baby)

        new_generation.extend(children)
        self.population = new_generation

def train(model, lr, train_loader, eval_loader, teacher_preds):
    num_steps = len(train_loader) * 5
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_steps*0.1,
                                                num_training_steps=num_steps)
    dev_best_acc = 0

    for epoch in range(5):
        model.train()
        tr_num = 0
        train_loss = 0

        logger.info('Epoch [{}/{}]'.format(epoch + 1, 5))
        bar = tqdm(train_loader, total=len(train_loader))
        bar.set_description("Train")
        for step, batch in enumerate(bar):
            texts = batch[0].to("cuda")
            labels = batch[1].to("cuda")
            loss, _ = model(texts, labels)

            loss.backward()
            train_loss += loss.item()
            tr_num += 1

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        dev_results = evaluate(model, eval_loader, teacher_preds)
        dev_acc = dev_results["agreements"]
        if dev_acc >= dev_best_acc:
            dev_best_acc = dev_acc

    return dev_best_acc

def evaluate(model, test_loader, teacher_preds):
    model.eval()
    predict_all = []
    labels_all = []

    with torch.no_grad():
        bar = tqdm(test_loader, total=len(test_loader))
        bar.set_description("Evaluation")
        for batch in bar:
            texts = batch[0].to("cuda")        
            label = batch[1].to("cuda")
            prob = model(texts)

            predict_all.append(prob.cpu().numpy())
            labels_all.append(label.cpu().numpy())

    predict_all = np.concatenate(predict_all, 0)
    labels_all = np.concatenate(labels_all, 0)

    preds = predict_all[:, 0] > 0.5
    recall = recall_score(labels_all, preds)
    precision = precision_score(labels_all, preds)
    f1 = f1_score(labels_all, preds)
    results = {
        "eval_acc": np.mean(labels_all==preds),
        "eval_precision": float(precision),
        "eval_recall": float(recall),
        "eval_f1": float(f1),
        "agreements": np.sum(teacher_preds==preds)
    }
    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--population_size", default=10, type=int, required=True)
    parser.add_argument("--generation_size", default=20, type=int, required=True)
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--model_dir", default="./", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=42,
                        help="random seed for initialization")

    args = parser.parse_args()
    search_space = {
        "model_arch": ["biGRU", "biLSTM"],
        "vocab_size": [*range(1000, 26000, 1000)],
        "input_dim": [*range(16, 769, 16)],
        "hidden_dim": [*range(16, 769, 16)],
        "n_layers": [*range(1, 13)]
    }

    logger.info("***Start GA search for %d generations and %d population***" %
          (args.generation_size, args.population_size))

    searcher = GA_search(args, search_space)
    searcher.initialization()

    for gen in tqdm(range(args.generation_size)):
        logger.info("***Start generate %d***" %(gen))
        searcher.generation()
    
    for genome in searcher.population:
            searcher.fitness(genome)
    graded_genome = [x for x in sorted(searcher.population, key=lambda x: x.fitness, reverse=True)]

    logger.info(graded_genome[0].gene_param)
    logger.info(graded_genome[0].fitness)


if __name__ == "__main__":
    main()
