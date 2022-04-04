import random
import logging
import hashlib
import copy

from train import train_and_score

class Genome():
    """
    Represents one genome and all relevant utility functions (add, mutate, etc.).
    """

    def __init__(self, all_possible_genes=None, geneparam={}, u_ID=0, mom_ID=0, dad_ID=0, gen=0):
        
        self.accuracy = 0.0
        self.all_possible_genes = all_possible_genes
        self.geneparam = geneparam
        self.u_ID = u_ID
        self.parents = [mom_ID, dad_ID]
        self.generation = gen

        if not geneparam:
            self.hash = 0
        else:
            self.update_hash()

    def update_hash(self):
        genh = str(self.all_possible_genes["vocab_size"]) + str(self.all_possible_genes["input_dim"]) + str(self.all_possible_genes["hidden_dim"]) + str(self.all_possible_genes["n_layers"]) + str(self.all_possible_genes["alpha"]) + str(self.all_possible_genes["lr"]) + str(self.all_possible_genes["temperature"])
        
        self.hash = hashlib.md5(genh.encode("UTF-8")).hexdigest()

        self.accuracy = 0.0

    def set_genes_random(self):
        """Create a random genome."""

        self.parents = [0, 0]

        for key in self.all_possible_genes:
            self.geneparam[key] = random.choice(self.all_possible_genes[key])

        self.update_hash()

    def mutate_one_gene(self):
        """
        Randomly mutate one gene in the genome.

        Args:
            network (dict): The genome parameters to mutate

        Returns:
            (Genome): A randomly mutated genome object

        """
        # Which gene shall we mutate? Choose one of N possible keys/genes.
        gene_to_mutate = random.choice(list(self.all_possible_genes.keys()))

        # And then let"s mutate one of the genes.
        # Make sure that this actually creates mutation
        current_value = self.geneparam[gene_to_mutate]

        possible_choices = copy.deepcopy(
            self.all_possible_genes[gene_to_mutate])

        possible_choices.remove(current_value)

        self.geneparam[gene_to_mutate] = random.choice(possible_choices)

        self.update_hash()

    def set_generation(self, generation):
        self.generation = generation

    def train(self, trainingset):
        if self.accuracy == 0.0:
            self.accuracy = train_and_score(self, trainingset)

    def print_genome(self):

        logging.info("Acc: %.2f%%" % (self.accuracy * 100))
        logging.info("UniID: %d" % self.u_ID)
        logging.info("Parents %d %d" % (self.parents[0], self.parents[1]))
        logging.info("Gen: %d" % self.generation)
        logging.info("Hash: %s" % self.hash)

class AllGenomes():
    def __init__(self, firstgenome):
        self.population = []
        self.population.append(firstgenome)

    def add_genome(self, genome):
        for i in range(0, len(self.population)):
            if (genome.hash == self.population[i].hash):
                logging.info(
                    "add_genome() ERROR: hash clash - duplicate genome")
                return False

        self.population.append(genome)

        return True

    def set_accuracy(self, genome):
        for i in range(0, len(self.population)):
            if (genome.hash == self.population[i].hash):
                self.population[i].accuracy = genome.accuracy
                return

        logging.info("set_accuracy() ERROR: Genome not found")

    def is_duplicate(self, genome):
        if len(self.population) > 1:
            for i in range(0, len(self.population)):
                if (genome.hash == self.population[i].hash):
                    return True
        return False
