import torch
import logging
from tqdm import tqdm
from evolution import Evolver

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

logger = logging.getLogger(__name__)

def train_genomes(genomes, dataset):
    """Train each genome.
    Args:
        networks (list): Current population of genomes
        dataset (str): Dataset to use for training/evaluating
    """
    logger.info("***train_networks(networks, dataset)***")

    pbar = tqdm(total=len(genomes))

    for genome in genomes:
        genome.train(dataset)
        pbar.update()

    pbar.close()


def get_average_accuracy(genomes):
    """Get the average accuracy for a group of networks/genomes.
    Args:
        networks (list): List of networks/genomes
    Returns:
        float: The average accuracy of a population of networks/genomes.
    """
    total_accuracy = 0

    for genome in genomes:
        total_accuracy += genome.accuracy

    return total_accuracy / len(genomes)


def generate(generations, population, all_possible_genes, dataset):
    """Generate a network with the genetic algorithm.
    Args:
        generations (int): Number of times to evolve the population
        population (int): Number of networks in each generation
        all_possible_genes (dict): Parameter choices for networks
        dataset (str): Dataset to use for training/evaluating
    """
    logger.info(
        "***generate(generations, population, all_possible_genes, dataset)***")

    evolution = Evolver(all_possible_genes)

    genomes = evolution.create_population(population)

    # Evolve the generation.
    for i in range(generations):

        logger.info("***Now in generation %d of %d***" % (i + 1, generations))

        print_genomes(genomes)

        # Train and get accuracy for networks/genomes.
        train_genomes(genomes, dataset)

        # Get the average accuracy for this generation.
        average_accuracy = get_average_accuracy(genomes)

        # Print out the average accuracy each generation.
        logger.info("Generation average: %.2f%%" % (average_accuracy * 100))
        logger.info("-"*80)  # -----------

        # Evolve, except on the last iteration.
        if i != generations - 1:
            # Evolve!
            genomes = evolution.evolve(genomes)

    # Sort our final population according to performance.
    genomes = sorted(genomes, key=lambda x: x.accuracy)

    # Print out the top 5 networks/genomes.
    print_genomes(genomes[:5])

    #save_path = saver.save(sess, "/output/model.ckpt")
    #logger.info("Model saved in file: %s" % save_path)


def print_genomes(genomes):
    """Print a list of genomes.
    Args:
        genomes (list): The population of networks/genomes
    """
    logger.info("-"*80)

    for genome in genomes:
        genome.print_genome()


def main():
    population = 10
    generations = 20
    dataset = "../data/defect_detection/train.jsonl"
    search_space = {
        "vocab_size": [*range(1000, 51000, 1000)],
        "input_dim": [*range(1, 769)],
        "hidden_dim": [*range(1, 769)],
        "n_layers": [*range(1, 13)],
        "alpha": torch.arange(0, 1, 0.02).tolist(),
        "lr": [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4], 
        "temperature": [*range(1, 11)]
    }

    logger.info("***Evolving for %d generations with population size = %d***" %
          (generations, population))

    generate(generations, population, search_space, dataset)


if __name__ == "__main__":
    main()
