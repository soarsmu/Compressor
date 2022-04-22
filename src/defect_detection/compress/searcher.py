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
from models import biLSTM, biGRU

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
    def __init__(self, args, search_space, desired_length=20, cross_chance=0.8):
        self.args = args
        self.search_space = search_space
        self.cross_chance = cross_chance
        self.desired_length = desired_length
        self.population = []
        self.best_gene = []

    def is_duplicate(self, new_genome):
        for genome in self.population:
            if new_genome.hash == genome.hash:
                return True
        return False

    def initialization(self):
        count = 0

        while count < self.args.population_size:
            gene_param = {}
            for key in self.search_space:
                gene_param[key] = random.choice(self.search_space[key])
            new_genome = Genome(gene_param)
            
            if len(self.population) > 0:
                while self.is_duplicate(new_genome):
                    new_genome.mutation(self.search_space)

            self.population.append(copy.deepcopy(new_genome))
            count += 1
    
    def fitness(self, genome):
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

        inputs = torch.randint(vocab_size, (1, 400))
        flops, _ = profile(model, (inputs, ), verbose=False)
        params = sum(p.numel() for p in model.parameters())
        size_diff = abs(self.args.target_size - params)*4/1e6
        logger.info(flops/1e9 - size_diff)
        logger.info("size %f", params*4.0/1e6)
        logger.info("flops %f", flops/1e9)

        genome.fitness = flops/1e9 - size_diff

    def crossover_and_mutation(self, parents):
        children = []
        parent_1, parent_2 = parents

        if self.cross_chance > random.random():
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
        else:
            genome_1 = copy.deepcopy(parent_1)
            genome_2 = copy.deepcopy(parent_2)

            genome_1.mutation(self.search_space)
            genome_2.mutation(self.search_space)

        # while self.is_duplicate(genome_1):
        #     genome_1.mutation(self.search_space)

        # while self.is_duplicate(genome_2):
        #     genome_2.mutation(self.search_space)

        children.append(genome_1)
        children.append(genome_2)

        return children

    def generation(self):
        children = []
        while len(children) < self.desired_length:
            parents = random.sample(self.population, k=2)
            children.extend(self.crossover_and_mutation(parents))

        # deduplication
        for genome in children:
            while self.is_duplicate(genome):
                genome.mutation(self.search_space)
            self.population.append(copy.deepcopy(genome))
        
        for genome in self.population:
            self.fitness(genome)

        graded_genome = [x for x in sorted(self.population, key=lambda x: x.fitness, reverse=True)]
        logger.info(graded_genome[0].gene_param)
        logger.info(graded_genome[0].fitness)
        self.best_gene.append((graded_genome[0].gene_param, graded_genome[0].fitness))
        self.population = graded_genome[:self.args.population_size]

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--population_size", default=20, type=int)
    parser.add_argument("--generation_size", default=20, type=int)
    parser.add_argument("--target_size", default=0.01, type=float)
    parser.add_argument("--target_flops", default=33989813760, type=float)

    args = parser.parse_args()
    search_space = {
        "model_arch": ["biGRU", "biLSTM"],
        "vocab_size": [*range(1000, 53000, 1000)],
        "input_dim": [*range(16, 769, 16)],
        "hidden_dim": [*range(16, 769, 16)],
        "n_layers": [*range(1, 13)]
    }
    params = 124647170
    args.target_size = params * 0.01
    logger.info("***Start GA search for %d generations, %d population, target model size %d MB***" %
          (args.generation_size, args.population_size, args.target_size*4/1e6))

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
    logger.info(searcher.best_gene)


if __name__ == "__main__":
    main()
