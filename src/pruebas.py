from jmetal.algorithm.multiobjective.ibea import IBEA
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem import ZDT1
from jmetal.util.termination_criterion import StoppingByEvaluations

problem = ZDT1()

max_evaluations = 2500

algorithm = IBEA(
    problem=problem,
    kappa=1.,
    population_size=100,
    offspring_population_size=100,
    mutation=PolynomialMutation(probability=1.0 / float(problem.number_of_variables()), distribution_index=20),
    crossover=SBXCrossover(probability=1.0, distribution_index=20),
    termination_criterion=StoppingByEvaluations(max_evaluations)
)

algorithm.run()
front = algorithm.result()

from jmetal.lab.visualization.plotting import Plot

plot_front = Plot(title='Pareto front approximation', axis_labels=['x', 'y'])
plot_front.plot(front, label='IBEA-ZDT1', filename='IBEA-ZDT1', format='png')