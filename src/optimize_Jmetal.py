import metrics
import parameters

import numpy as np
import random
import os
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from jmetal.core.problem import FloatProblem, FloatSolution
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.util.solution import get_non_dominated_solutions
from jmetal.lab.visualization import Plot



class PM_miner_problem(FloatProblem):

    current_iteration = 0

    def __init__(self, miner, log, metrics_obj: metrics.Metrics, parameters_info, verbose):
        
        super(PM_miner_problem, self).__init__()

        self.miner = miner                       # pm4py miner e.g. heuristic_miner
        self.log = log                           # XES log (already loaded by pm4py)
        self.metrics_obj = metrics_obj           # How to calculate fitness
        self.parameters_info = parameters_info   # Miner pararameters (selected automatically)
        self.verbose = verbose                   # more vebosity == more runtime

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
        if self.verbose == 1:
            print(f'Iteracion: {self.current_iteration}', end='\r')
            self.current_iteration+=1
        params = {key: solution.variables[idx] for idx, key in enumerate(self.parameters_info.param_range.keys())}
        petri, _, _ = self.miner.apply(self.log, parameters= params)
        solution.objectives = self.metrics_obj.get_metrics_array(petri)
        solution.number_of_objectives = self.number_of_objectives
        
        return solution
    
    def create_solution(self) -> FloatSolution:
        new_solution = FloatSolution(number_of_constraints=self.number_of_constraints,
                                     number_of_objectives=self.number_of_objectives,
                                     lower_bound = self.lower_bound,
                                     upper_bound = self.upper_bound)   
        # Random Solution
        random_sol = list()
        for index, param_and_type in enumerate(self.parameters_info.param_type.items()):
            data_type = param_and_type[1]
            if data_type == int: 
                random_sol.append(random.randint(self.lower_bound[index], self.upper_bound[index]))
            else:
                random_sol.append(random.uniform(self.lower_bound[index], self.upper_bound[index]))


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

class Opt_NSGAII(PM_miner_problem):
     
    def __init__(self, miner, log, metrics_obj, verbose):
        parameters_info = self.__get_parameters(miner, log)
        super().__init__(miner, log, metrics_obj, parameters_info, verbose)
        

    def __get_parameters(self, miner, log):
        if miner == heuristics_miner:
            params = parameters.Heuristic_Parameters()
            params.adjust_heu_params(log)
            return params
        else:
            raise ValueError(f"Miner '{miner}' not supported. Available miners are: Heuristic, Inductive")

    def __show_result(self):
        print("\n### RESULT ###\n")
        for (i,j)in enumerate(self.result):
            print("Solution ",i," :")
            print("     variables: ",j.variables)
            print("     objectives:",j.objectives.tolist(),"\n")
        print("##############")

    def discover(self, **params):
        self.algorithm = NSGAII(problem= self,**params)
        
        self.algorithm.run()
        self.result = self.algorithm.result()
        self.__show_result()
        self.non_dom_sols =  get_non_dominated_solutions(self.algorithm.result()) ## Añadí .all() a archive.py (def add)

    def get_result(self):
        return self.result
        
    def get_best_solution(self):
        return self.result[0]  # TO-DO

    def get_petri_net(self, sol=None):
        if sol == None:
            sol = self.get_best_solution()
    
        print("\n### Solution ###\n")
        print(sol)
        print("\n#####################\n")

        params = {key: sol.variables[idx] for idx, key in enumerate(self.parameters_info.param_range.keys())}

        petri_net, initial_marking, final_marking = self.miner.apply(self.log, parameters=params)
        return petri_net, initial_marking, final_marking
    
    def get_non_dominated_sols(self):
        return self.non_dom_sols
    
    def plot_pareto_front(self, title, label, filename, format):
        front = self.get_non_dominated_sols()
        plot_front = Plot(title=title, axis_labels=self.metrics_obj.get_labels())
        plot_front.plot(front, label=label, filename=filename, format=format)

    def get_pareto_front_petri_nets(self):
        front = self.get_non_dominated_sols()
        petri_nets_from_pareto_sols = list()
        for sol in front:
            petri_nets_from_pareto_sols.append(self.get_petri_net(sol))
        return petri_nets_from_pareto_sols

    
## Testing

if __name__ == "__main__":

    from pm4py.objects.log.importer.xes import importer as xes_importer
    from pm4py.visualization.petri_net import visualizer as pn_visualizer
    from jmetal.operator.crossover import SBXCrossover
    from jmetal.operator.mutation import PolynomialMutation
    from jmetal.util.termination_criterion import StoppingByEvaluations



    max_evaluations = 1000

    log = xes_importer.apply('test/Closed/BPI_Challenge_2013_closed_problems.xes')
    metrics_obj = metrics.Basic_Metrics()

    opt = Opt_NSGAII(heuristics_miner, log, metrics_obj, verbose=0)
    opt.discover(population_size=100,
                 offspring_population_size=100,
                 mutation = PolynomialMutation(probability=1.0 / opt.number_of_variables, distribution_index=20),
                 crossover = SBXCrossover(probability=1.0, distribution_index=20),
                 termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations))
    
    optimal_petri_net, initial_marking, final_marking = opt.get_petri_net()

    # visualize petri net
    gviz = pn_visualizer.apply(optimal_petri_net, initial_marking, final_marking)
    pn_visualizer.view(gviz)

    # plot Pareto front
    opt.plot_pareto_front(title='Pareto front approximation', label='NSGAII-Pareto-Closed', filename='NSGAII-Pareto-Closed', format='png')
