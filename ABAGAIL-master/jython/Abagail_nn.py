"""
Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
find optimal weights to a neural network that is classifying abalone as having either fewer
or more than 15 rings.

Based on AbaloneTest.java by Hannah Lau
"""

from __future__ import with_statement

import sys
sys.path.append("C:/Users/Myles/Documents/OMSCS/CS7641 ML/Assignment 2/ABAGAIL-master/ABAGAIL.jar")

import os
import csv
import time

from func.nn.backprop import BackPropagationNetworkFactory
from shared import SumOfSquaresError, DataSet, Instance
from opt.example import NeuralNetworkOptimizationProblem

import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm



INPUT_FILE = os.path.join("..", "src", "opt", "test", "incometrainingdata.csv")
TEST_FILE = os.path.join("..", "src", "opt", "test", "incometestdata.csv")

INPUT_LAYER = 47
HIDDEN_LAYER = 3
OUTPUT_LAYER = 1
TRAINING_ITERATIONS = 1000


def initialize_instances():
    """Read the incomedata.csv CSV data into a list of instances."""
    instances = []

    # Read in the incomedata.txt CSV file
    with open(INPUT_FILE, "r") as incomedata:
        reader = csv.reader(incomedata)

        for row in reader:
            instance = Instance([float(value) for value in row[:-1]])
            instance.setLabel(Instance(0 if float(row[-1]) == 0 else 1))
            instances.append(instance)

    return instances

def initialize_testinstances():
    """Read the incomedata.csv CSV data into a list of instances."""
    instances = []

    # Read in the incomedata.txt CSV file
    with open(TEST_FILE, "r") as incomedata:
        reader = csv.reader(incomedata)

        for row in reader:
            instance = Instance([float(value) for value in row[:-1]])
            instance.setLabel(Instance(0 if float(row[-1]) == 0 else 1))
            instances.append(instance)

    return instances


def train(oa, network, oaName, instances, measure):
    """Train a given network on a set of instances.

    :param OptimizationAlgorithm oa:
    :param BackPropagationNetwork network:
    :param str oaName:
    :param list[Instance] instances:
    :param AbstractErrorMeasure measure:
    """
    print("\nError results for %s\n---------------------------" % (oaName,))

    for iteration in xrange(TRAINING_ITERATIONS):
        oa.train()

        error = 0.00
        for instance in instances:
            network.setInputValues(instance.getData())
            network.run()

            output = instance.getLabel()
            output_values = network.getOutputValues()
            example = Instance(output_values, Instance(output_values.get(0)))
            error += measure.value(output, example)

        print("%0.03f" % error)


def main():
    """Run algorithms on the abalone dataset."""
    instances = initialize_instances()
    test_instances = initialize_testinstances()
    factory = BackPropagationNetworkFactory()
    measure = SumOfSquaresError()
    data_set = DataSet(instances)

    networks = []  # BackPropagationNetwork
    nnop = []  # NeuralNetworkOptimizationProblem
    oa = []  # OptimizationAlgorithm
    #oa_names = ["RHC", "SA", "GA"]
    #oa_names = ["RHC", "SA"]
    oa_names = ["GA"]
    results = ""

    for name in oa_names:
        classification_network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER])
        networks.append(classification_network)
        nnop.append(NeuralNetworkOptimizationProblem(data_set, classification_network, measure))

    #oa.append(RandomizedHillClimbing(nnop[0]))
    #oa.append(SimulatedAnnealing(1E5, .99, nnop[0]))
    oa.append(StandardGeneticAlgorithm(200, 200, 200, nnop[0]))

    for i, name in enumerate(oa_names):
        start = time.time()
        traincorrect = 0
        trainincorrect = 0
        testcorrect = 0
        testincorrect = 0

        train(oa[i], networks[i], oa_names[i], instances, measure)
        end = time.time()
        training_time = end - start

        optimal_instance = oa[i].getOptimal()
        networks[i].setWeights(optimal_instance.getData())

        start = time.time()
        for instance in instances:
            networks[i].setInputValues(instance.getData())
            networks[i].run()

            predicted = instance.getLabel().getContinuous()
            actual = networks[i].getOutputValues().get(0)

            if abs(predicted - actual) < 0.5:
                traincorrect += 1
            else:
                trainincorrect += 1

        end = time.time()
        train_results_time = end - start

        start = time.time()
        for instance in test_instances:
            networks[i].setInputValues(instance.getData())
            networks[i].run()

            predicted = instance.getLabel().getContinuous()
            actual = networks[i].getOutputValues().get(0)

            if abs(predicted - actual) < 0.5:
                testcorrect += 1
            else:
                testincorrect += 1

        end = time.time()
        testing_time = end - start

        results += "\nResults for %s: \nCorrectly classified %d instances." % (name, traincorrect)
        results += "\nIncorrectly classified %d instances.\nPercent correctly classified: %0.03f%%" % (trainincorrect, float(traincorrect)/(traincorrect+trainincorrect)*100.0)
        results += "\nCorrectly classified %d test instances." % (testcorrect)
        results += "\nPercent of test correctly classified: %0.03f%%" % (float(testcorrect) / (testcorrect + testincorrect) * 100.0)
        results += "\nTraining time: %0.03f seconds" % (training_time,)
        results += "\nTraining results time: %0.03f seconds" % (train_results_time,)
        results += "\nTesting time: %0.03f seconds\n" % (testing_time,)

    print(results)


if __name__ == "__main__":
    main()

