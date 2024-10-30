from jmetal.problem import ZDT1
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.algorithm.multiobjective.nsgaiii import NSGAIII, UniformReferenceDirectionFactory
from jmetal.operator import PolynomialMutation, SBXCrossover
from jmetal.algorithm.multiobjective.spea2 import SPEA2

problem = ZDT1()



# Configurar el algoritmo
problem = problem
algorithm = SPEA2(
    problem=problem,
    population_size=40,
    offspring_population_size=40,
    mutation=PolynomialMutation(probability=0.2, distribution_index=20),
    crossover=SBXCrossover(probability=1.0, distribution_index=20),
    termination_criterion=StoppingByEvaluations(max_evaluations=1000)
)

algorithm.run()
solutions = algorithm.result()

print(solutions)
