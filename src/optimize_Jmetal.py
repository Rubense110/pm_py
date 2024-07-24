import metrics
import parameters

import numpy as np
import random
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from jmetal.core.problem import FloatProblem, FloatSolution
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.util.solution import get_non_dominated_solutions



class PM_miner_problem(FloatProblem):

    def __init__(self, miner, log, metrics_obj: metrics.Metrics, parameters_info):
        
        super(PM_miner_problem, self).__init__()

        self.miner = miner                       # pm4py miner e.g. heuristic_miner
        self.log = log                           # XES log (already loaded by pm4py)
        self.metrics_obj = metrics_obj           # How to calculate fitness
        self.parameters_info = parameters_info   # Miner pararameters (selected automatically)

        self.number_of_objectives = metrics_obj.get_n_of_metrics()
        self.number_of_variables = self.__get_n_genes()
        self.number_of_constraints = 0

        self.lower_bound, self.upper_bound = self.__get_bounds()

        #self.obj_directions = [self.MINIMIZE]
        self.obj_labels = metrics_obj.get_labels()

    # Determines the value range for the parameters of the specified miner
    def __get_bounds(self):
        if self.miner == heuristics_miner:
            lower_bound = [i[0] for i in self.parameters_info.param_range.values()]
            upper_bound = [i[1] for i in self.parameters_info.param_range.values()]
        else: pass

        return lower_bound, upper_bound
    
    def __get_n_genes(self):
        if self.miner == heuristics_miner:
            return len(self.parameters_info.param_range)
        
    def evaluate(self, solution: FloatSolution) -> FloatSolution :
        
        params = {key: solution.variables[idx] for idx, key in enumerate(self.parameters_info.param_range.keys())}
        petri, _, _ = self.miner.apply(self.log, parameters= params)
        solution.objectives = self.metrics_obj.get_metrics_array(petri)
        solution.number_of_objectives = self.number_of_objectives
        #print(solution)
        return solution

    def create_solution(self) -> FloatSolution:
        new_solution = FloatSolution(number_of_constraints=self.number_of_constraints,
                                     number_of_objectives=self.number_of_objectives,
                                     lower_bound = self.lower_bound,
                                     upper_bound = self.upper_bound)
        
        # Random Solution
        random_sol = list()
        for i in self.parameters_info.param_range.values():
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
     
    def __init__(self, miner, log, metrics_obj):
        self.parameters_info = self.__get_parameters(miner)
        self.problem = PM_miner_problem(miner, log, metrics_obj, self.parameters_info)
        

    def __get_parameters(self, miner):
        if miner == heuristics_miner:
            return parameters.Heuristic_Parameters
        else:
            raise ValueError(f"Miner '{miner}' not supported. Available miners are: Heuristic, Inductive")

    def discover(self, pop_size, off_pop_size, mutation, crossover, termination_criterion):
        
        self.algorithm = NSGAII(problem= self.problem,
                                population_size=pop_size,
                                offspring_population_size = off_pop_size,
                                mutation = mutation,
                                crossover = crossover,
                                termination_criterion = termination_criterion)
        
        self.algorithm.run()
        self.result = self.algorithm.result()
        self.non_dom_sols =  get_non_dominated_solutions(self.algorithm.result()) ## Añadí .all() a archive.py (def add)
        return self.result
    
    def get_best_solution(self):
        return self.result[0]  # TO-DO

    def get_petri_net(self):
        best_solution = self.get_best_solution()
        print(best_solution)
        params = {key: best_solution.variables[idx] for idx, key in enumerate(self.parameters_info.param_range.keys())}

        petri_net, initial_marking, final_marking = self.problem.miner.apply(self.problem.log, parameters=params)
        return petri_net, initial_marking, final_marking
    
    def get_non_dominated_sols(self):
        return self.non_dom_sols
    
## Testing
if __name__ == "__main__":

    from pm4py.objects.log.importer.xes import importer as xes_importer
    from pm4py.visualization.petri_net import visualizer as pn_visualizer
    from jmetal.operator.crossover import SBXCrossover
    from jmetal.operator.mutation import PolynomialMutation
    from jmetal.util.termination_criterion import StoppingByEvaluations
    from jmetal.lab.visualization import Plot


    max_evaluations = 100

    log = xes_importer.apply('test/Closed/BPI_Challenge_2013_closed_problems.xes')
    metrics_obj = metrics.Basic_Metrics()

    opt = Opt_NSGAII(heuristics_miner, log, metrics_obj)
    opt.discover(pop_size=100,
                 off_pop_size=100,
                 mutation = PolynomialMutation(probability=1.0 / opt.problem.number_of_variables, distribution_index=20),
                 crossover = SBXCrossover(probability=1.0, distribution_index=20),
                 termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations))
    
    optimal_petri_net, initial_marking, final_marking = opt.get_petri_net()

    # visualize petri net
    gviz = pn_visualizer.apply(optimal_petri_net, initial_marking, final_marking)
    #pn_visualizer.view(gviz)

    front = opt.get_non_dominated_sols()
    

    plot_front = Plot(title='Pareto front approximation', axis_labels=['x', 'y', 'z', 'a', 'b', 'c', 'd'])
    plot_front.plot(front, label='NSGAII-ZDT1', filename='NSGAII-ZDT1', format='png')
