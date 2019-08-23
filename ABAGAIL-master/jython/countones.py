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
import time

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
import opt.example.CountOnesEvaluationFunction as CountOnesEvaluationFunction
from array import array




"""
Commandline parameter(s):
   none
"""

N=8
fill = [2] * N
ranges = array('i', fill)

print("N: " + str(N))

ef = CountOnesEvaluationFunction()
odd = DiscreteUniformDistribution(ranges)
nf = DiscreteChangeOneNeighbor(ranges)
mf = DiscreteChangeOneMutation(ranges)
cf = SingleCrossOver()
df = DiscreteDependencyTree(.1, ranges)
hcp = GenericHillClimbingProblem(ef, odd, nf)
gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
pop = GenericProbabilisticOptimizationProblem(ef, odd, df)

start = time.time()
rhc = RandomizedHillClimbing(hcp)
rhcIterations = 100
fit = FixedIterationTrainer(rhc, rhcIterations)
fit.train()
print("\nRHC: " + str(ef.value(rhc.getOptimal())))
print("RHC Iterations: " + str(rhcIterations))
end = time.time()
traintime = end - start
print("RHC results time: %0.03f seconds" % (traintime,))

start = time.time()
sa = SimulatedAnnealing(100, .95, hcp)
saIterations = 200
fit = FixedIterationTrainer(sa, saIterations)
fit.train()
print("\nSA: " + str(ef.value(sa.getOptimal())))
print("SA Iterations: " + str(saIterations))
end = time.time()
traintime = end - start
print("SA results time: %0.03f seconds" % (traintime,))

start = time.time()
ga = StandardGeneticAlgorithm(20, 20, 0, gap)
gaIterations = 200
fit = FixedIterationTrainer(ga, gaIterations)
fit.train()
print("\nGA: " + str(ef.value(ga.getOptimal())))
print("GA Iterations: " + str(gaIterations))
end = time.time()
traintime = end - start
print("GA results time: %0.03f seconds" % (traintime,))

start = time.time()
mimic = MIMIC(50, 10, pop)
mimicIterations = 100
fit = FixedIterationTrainer(mimic, mimicIterations)
fit.train()
print("\nMIMIC: " + str(ef.value(mimic.getOptimal())))
print("MIMIC Iterations: " + str(mimicIterations))
end = time.time()
traintime = end - start
print("MIMIC results time: %0.03f seconds" % (traintime,))