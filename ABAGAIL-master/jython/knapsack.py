import sys
import os
import time

import sys
sys.path.append("C:/Users/Myles/Documents/OMSCS/CS7641 ML/Assignment 2/ABAGAIL-master/ABAGAIL.jar")

import java.io.FileReader as FileReader
import java.io.File as File
import java.lang.String as String
import java.lang.StringBuffer as StringBuffer
import java.lang.Boolean as Boolean
import java.util.Random as Random

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import dist.Distribution as Distribution
import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import opt.EvaluationFunction as EvaluationFunction
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.HillClimbingProblem as HillClimbingProblem
import opt.NeighborFunction as NeighborFunction
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.example.FourPeaksEvaluationFunction as FourPeaksEvaluationFunction
import opt.ga.CrossoverFunction as CrossoverFunction
import opt.ga.SingleCrossOver as SingleCrossOver
import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.GeneticAlgorithmProblem as GeneticAlgorithmProblem
import opt.ga.MutationFunction as MutationFunction
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.ga.UniformCrossOver as UniformCrossOver
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import opt.prob.ProbabilisticOptimizationProblem as ProbabilisticOptimizationProblem
import shared.FixedIterationTrainer as FixedIterationTrainer
import opt.example.KnapsackEvaluationFunction as KnapsackEvaluationFunction
from array import array




"""
Commandline parameter(s):
    none
"""

# Random number generator */
random = Random()
# The number of items
NUM_ITEMS = 20
# The number of copies each
COPIES_EACH = 1
# The maximum weight for a single element
MAX_WEIGHT = 1
# The maximum volume for a single element
MAX_VOLUME = 1
# The volume of the knapsack 
KNAPSACK_VOLUME = MAX_VOLUME * NUM_ITEMS * COPIES_EACH * .4

print("NUM_ITEMS: " + str(NUM_ITEMS))
print("COPIES_EACH: " + str(COPIES_EACH))
print("MAX_WEIGHT: " + str(MAX_WEIGHT))
print("MAX_VOLUME: " + str(MAX_VOLUME))
print("KNAPSACK_VOLUME: " + str(KNAPSACK_VOLUME))

# create copies
fill = [COPIES_EACH] * NUM_ITEMS
copies = array('i', fill)

# create weights and volumes
fill = [0] * NUM_ITEMS
weights = array('d', fill)
volumes = array('d', fill)
for i in range(0, NUM_ITEMS):
    weights[i] = random.nextDouble() * MAX_WEIGHT
    volumes[i] = random.nextDouble() * MAX_VOLUME


# create range
fill = [COPIES_EACH + 1] * NUM_ITEMS
ranges = array('i', fill)

ef = KnapsackEvaluationFunction(weights, volumes, KNAPSACK_VOLUME, copies)
odd = DiscreteUniformDistribution(ranges)
nf = DiscreteChangeOneNeighbor(ranges)
mf = DiscreteChangeOneMutation(ranges)
cf = UniformCrossOver()
df = DiscreteDependencyTree(.1, ranges)
hcp = GenericHillClimbingProblem(ef, odd, nf)
gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
pop = GenericProbabilisticOptimizationProblem(ef, odd, df)

start = time.time()
rhc = RandomizedHillClimbing(hcp)
fit = FixedIterationTrainer(rhc, 200000)
fit.train()
print "\nRHC: " + str(ef.value(rhc.getOptimal()))
end = time.time()
traintime = end - start
print("RHC results time: %0.03f seconds" % (traintime,))

start = time.time()
sa = SimulatedAnnealing(100, .95, hcp)
fit = FixedIterationTrainer(sa, 200000)
fit.train()
print "\nSA: " + str(ef.value(sa.getOptimal()))
end = time.time()
traintime = end - start
print("SA results time: %0.03f seconds" % (traintime,))

start = time.time()
ga = StandardGeneticAlgorithm(200, 150, 25, gap)
fit = FixedIterationTrainer(ga, 1000)
fit.train()
print "\nGA: " + str(ef.value(ga.getOptimal()))
end = time.time()
traintime = end - start
print("GA results time: %0.03f seconds" % (traintime,))

start = time.time()
mimic = MIMIC(200, 100, pop)
fit = FixedIterationTrainer(mimic, 1000)
fit.train()
print "\nMIMIC: " + str(ef.value(mimic.getOptimal()))
end = time.time()
traintime = end - start
print("MIMIC results time: %0.03f seconds" % (traintime,))