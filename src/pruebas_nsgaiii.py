from jmetal.problem import ZDT1
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.algorithm.multiobjective.nsgaiii import NSGAIII, UniformReferenceDirectionFactory
from jmetal.operator import PolynomialMutation, SBXCrossover

problem = ZDT1()

# Parámetros
n_dim = 2  # Número de objetivos
n_points = 100  # Número de puntos de referencia

# Crear la fábrica de direcciones de referencia
reference_directions = UniformReferenceDirectionFactory(n_dim=n_dim, n_points=n_points)


# Configurar el algoritmo
problem = problem
algorithm = NSGAIII(
    reference_directions=reference_directions,
    problem=problem,
    mutation=PolynomialMutation(probability=0.2, distribution_index=20),
    crossover=SBXCrossover(probability=1.0, distribution_index=20),
    termination_criterion=StoppingByEvaluations(max_evaluations=2500)
)

algorithm.run()
solutions = algorithm.result()

print(solutions)


## Funciona cambiando el np.int por int