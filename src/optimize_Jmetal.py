import metrics
import parameters

import numpy as np
import random
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from jmetal.core.problem import FloatProblem, FloatSolution
from jmetal.algorithm.multiobjective.nsgaii import NSGAII


class PM_miner_problem(FloatProblem):

    def __init__(self, miner, log, metrics_func):
        
        super(PM_miner_problem, self).__init__()

        self.miner = miner                 # pm4py miner e.g. heuristic_miner
        self.log = log                     # XES log (already loaded by pm4py)
        self.metrics_func = metrics_func   # How to calculate fitness

        self.number_of_objectives = metrics.N_BASE_METRICS
        self.number_of_variables = self.__get_n_genes()
        self.number_of_constraints = 0

        self.lower_bound, self.upper_bound = self.__get_bounds()

        #self.obj_directions = [self.MINIMIZE]
        self.obj_labels = metrics.LABELS_BASE_METRICS

    # Determines the value range for the parameters of the specified miner
    def __get_bounds(self):
        if self.miner == heuristics_miner:
            lower_bound = [i[0] for i in parameters.heu_param_range.values()]
            upper_bound = [i[1] for i in parameters.heu_param_range.values()]
        else: pass

        return lower_bound, upper_bound
    
    def __get_n_genes(self):
        if self.miner == heuristics_miner:
            return len(parameters.heu_param_range)
        
    def evaluate(self, solution: FloatSolution) -> FloatSolution :
        print(solution)
        params = {key: solution.objectives[idx] for idx, key in enumerate(parameters.heu_param_range.keys())}
        petri, _, _ = self.miner.apply(self.log, parameters= params)
        solution.objectives = self.metrics_func(petri)
        
        return solution

    def create_solution(self) -> FloatSolution:
        new_solution = FloatSolution(number_of_constraints=self.number_of_constraints,
                                     number_of_objectives=self.number_of_objectives,
                                     lower_bound = self.lower_bound,
                                     upper_bound = self.upper_bound)
        
        # Random Solution
        random_sol = list()
        for i in parameters.heu_param_type.values():
            if i == int: 
                random_sol.append(random.randint(self.lower_bound[0], self.upper_bound[1]))
            else:
                random_sol.append(random.uniform(self.lower_bound[0], self.upper_bound[1]))

        new_solution.variables = random_sol
        return new_solution

    def name(self) -> str:
        return 'Custom Process Mining Problem'
    
    def number_of_constraints(self) -> int:
        return super().number_of_constraints()
    
    def number_of_variables(self) -> int:
        return super().number_of_variables()
    
    def number_of_objectives(self) -> int:
        return super().number_of_objectives()

class Opt_NSGAII():
     
    def __init__(self, miner, log, metrics_func):
        self.problem = PM_miner_problem(miner, log, metrics_func)

    def discover(self, pop_size, off_pop_size, mutation, crossover, termination_criterion):
        
        self.algorithm = NSGAII(problem= self.problem,
                                population_size=pop_size,
                                offspring_population_size = off_pop_size,
                                mutation = mutation,
                                crossover = crossover,
                                termination_criterion = termination_criterion)
        
        self.algorithm.run()
        self.result = self.algorithm.result()
        return self.result
    
    def get_best_solution(self):
        return self.result[0]  # TO-DO

    def get_petri_net(self):
        best_solution = self.get_best_solution()
        print(best_solution)
        params = {key: best_solution.variables[idx] for idx, key in enumerate(parameters.heu_param_range.keys())}

        # Minar el registro con los parámetros asignados
        petri_net, initial_marking, final_marking = self.problem.miner.apply(self.problem.log, parameters=params)
        return petri_net, initial_marking, final_marking
    

if __name__ == "__main__":
    from pm4py.objects.log.importer.xes import importer as xes_importer
    from pm4py.visualization.petri_net import visualizer as pn_visualizer
    from jmetal.operator.crossover import SBXCrossover
    from jmetal.operator.mutation import PolynomialMutation
    from jmetal.util.termination_criterion import StoppingByEvaluations

    max_evaluations = 100

    log = xes_importer.apply('test/Closed/BPI_Challenge_2013_closed_problems.xes')

    opt = Opt_NSGAII(heuristics_miner, log, metrics.get_base_metrics)
    opt.discover(pop_size=100,
                 off_pop_size=100,
                 mutation = PolynomialMutation(probability=1.0 / opt.problem.number_of_variables, distribution_index=20),
                 crossover = SBXCrossover(probability=1.0, distribution_index=20),
                 termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations))
    
    optimal_petri_net, initial_marking, final_marking = opt.get_petri_net()

    # visualize petri net
    gviz = pn_visualizer.apply(optimal_petri_net, initial_marking, final_marking)
    pn_visualizer.view(gviz)