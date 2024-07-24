import metrics
import parameters

import numpy as np
import pygad
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner

### DEPRECATED - Does not work ###

class Opt_problem():
    
    fitness = np.array([])

    def __init__(self, miner, log, metrics_func):
        self.miner = miner                #pm4py miner e.g. heuristic_miner
        self.log = log                    # XES log (already loaded by pm4py)
        self.metrics_func = metrics_func   # How to calculate fitness
        
    # Determines the value range for the parameters of the specified miner
    def __get_param_range(self):
        param_range_list = list()
        if self.miner == heuristics_miner:
             param_range_list = [i for i in parameters.heu_param_range.values()]
        else:
             pass
        return param_range_list
    
    def __get_n_genes(self):
        if self.miner == heuristics_miner:
            return len(parameters.heu_param_range)

    def discover_genetics(self, n_generations, n_pops, n_sols):

        # Discover petri net with the chromosome (params) and evaluate with the specified metrics
        def fitness(ga_instance, solution, solution_idx): 
            params = {key: solution[idx] for idx, key in enumerate(parameters.heu_param_range.keys())}
            petri, _, _ = self.miner.apply(self.log, parameters= params)
            metrics = self.metrics_func(petri)

            return metrics

        # Creating genetic algorithm instance
        ga_instance = pygad.GA(num_generations=n_generations,
                               num_parents_mating=n_pops,
                               sol_per_pop=n_sols,
                               num_genes=self.__get_n_genes(),
                               fitness_func= fitness,
                               gene_space= self.__get_param_range(),
                               parent_selection_type='nsga2')
        
        ga_instance.run()

        solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
        print(f"\nOptimal Solution parameters: {solution}")
        print(f"Optimal Solution fitness: {solution_fitness}")

        optimal_params = {key: solution[idx] for idx, key in enumerate(parameters.heu_param_range.keys())}
        print(f"Optimal Parameters for Heuristic Miner: {optimal_params}")
        optimal_petri_net, initial_marking, final_marking = self.miner.apply(self.log, parameters=optimal_params)
        
        return optimal_petri_net, initial_marking, final_marking



## TESTING
if __name__ == "__main__":
    from pm4py.objects.log.importer.xes import importer as xes_importer
    from pm4py.visualization.petri_net import visualizer as pn_visualizer

    log = xes_importer.apply('test/Closed/BPI_Challenge_2013_closed_problems.xes')

    opt = Opt_problem(heuristics_miner, log, metrics.get_base_metrics)
    optimal_petri_net, initial_marking, final_marking = opt.discover_genetics(n_generations=100, n_pops=10, n_sols=10)


    # visualize petri net
    gviz = pn_visualizer.apply(optimal_petri_net, initial_marking, final_marking)
    pn_visualizer.view(gviz)