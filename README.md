# Randomized Optimization

### Introduction

In this project I implemented 4 randomized optimization algorithms on the adult income data set and 3 problem domains.

Randomized Optimization Algorithms:
- Randomized Hill Climbing (RHC)
- Simulated Annealing (SA)
- Genetic Algorithms (GA)
- MIMIC

Problem Domains:
- Count Ones
- The Traveling Salesman Problem
- The Knapsack Problem

The full report is available here: [Report](/Analysis.pdf)

#### Adult Income Data Set

The **adult income** problem classifies adults into one of two income categories: ‘>50K’, or ‘<=50K’. The ‘>50K’ category identifies individuals that earned more than $50,000 in the given year, 1994. The ‘<=50K’ category identifies individuals that earned less than or equal to $50,000. $50,000 in 1994 is approximately $81,000 in today’s terms. The data has 13 attributes, 5 of which are real valued (age, hours worked per week, etc), and 8 of which are categorical (education, marital status, race, etc).

![Income Randomized Optimization Chart](./Images/Adult%20Income%20Randomized%20Optimization%20Results.png)

Different algorithms have different performance rates when training neural networks. In this example, backpropagation was most closely approximated by using randomized hill climbing. The algorithm performs only marginally worse on the test and training set and took less than 1/3rd the training time. Simulated annealing and the genetic algorithm also performed better than chance on the test set (76%) and all have shorter training times than backpropagation.

#### Count Ones

Count Ones is a well understood problem where the parameter space is a bit string (a string of 1s and 0s) and the optimal result is the string of all 1s.

We can use randomized optimization to solve this problem quickly and reliably. The evaluation function is simply the sum of the bits in the bit string (000011101 sums to 4). By starting with a random string, the optimization algorithms can search through the neighborhood of strings to find the single global maximum.

With enough iterations and a sufficiently small parameter space, any optimization algorithm should eventually be able to solve the count ones problem. Since the domain is unstructured, we would expect RHC and SA to perform better than GA and MIMIC. 

With N (bit string size) = 800 the problem is very complex. We could expect GA and MIMIC to fail to converge in reasonable time, while RHC and SA should still succeed. Indeed, we see convergence in 20,000 iterations for both RHC and SA. Total clock time for either algorithm is under .25 seconds. MIMIC approaches 5 minutes of runtime at 500 iterations and is far from convergence at that point. 

![Count Ones N=800 Accuracy](./Images/Count%20Ones%20N=800%20Accuracy.png)
![Count Ones N=800 Times](./Images/Count%20Ones%20N=800%20Times.png)

#### The Traveling Salesman Problem

The traveling salesman problem defines the optimization problem where an algorithm must traverse every node in a connected network while traveling the minimum distance in that network. 

With enough iterations and a sufficiently small parameter space, any optimization algorithm should eventually be able to solve the traveling salesman problem. Since the domain is structured, we would expect GA and MIMIC to perform better than SA and RHC. 

As we can see in the output below, with only 5 nodes, any of our algorithms converge on the solution for the traveling salesman problem and do so relatively quickly (<1 second). For problems with many nodes, genetic algorithms stand out with the ability to find much better solutions without too much cost.

![TSP Distance](./Images/TSP%20Distance.png)
![TSP Times](./Images/TSP%20Times.png)

#### The Knapsack Problem

The knapsack problem defines the optimization problem where an algorithm must add items to a knapsack such that the total value of included items is maximized and the weight constraint is not exceeded. An algorithm must identify which items to include and which to exclude.

With enough iterations and a sufficiently small parameter space, any optimization algorithm should eventually be able to solve the knapsack problem. Since the domain is structured, we would GA and MIMIC to perform better than SA and RHC. 

To compare the algorithms, I first selected a simple problem space: 2 available items, 1 copy of each, with a knapsack that holds 1 unit of weight and has 1 unit of volume. As you can see in the analysis below, SA, GA, and MIMIC all performed similarly and had the same total value with similar runtimes. The available items were increased to increase the complexity of the problem. It’s easy to see MIMIC succeeding with Genetic Algorithms not far behind. 

![Knapsack Value](./Images/Knapsack%20Value.png)
![Knapsack Times](./Images/Knapsack%20Times.png)

#### Summary

These three problem domains, Count Ones, Traveling Salesman, and Knapsack, can all be optimized using the randomized optimization techniques discussed in this course, but each algorithm performs differently on each problem domain. The simulated annealing algorithm performs best on unstructured problems like count ones. Genetic algorithms perform best on a structured problem like the traveling salesman problem. MIMIC is able to perform best on a complex structured problem like the knapsack problem.