# activate venv -> .\.venv\Scripts\activate

import pm4py
from pm4py.algo.discovery.heuristics.variants.classic import Parameters as HeuristicsParameters
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner

import parameters
import optimize_Jmetal
import optimize_pyGAD
import metrics 

class Process_miner:

    miner_mapping = {
        #'inductive': pm4py.algo.discovery.inductive,
        'heuristic': heuristics_miner,
    }

    metrics_mapping = {
        'basic': metrics.Basic_Metrics()
    }

    def __init__(self, miner_type, opt_type, metrics,  log):
        self.miner = self.__get_miner_alg(miner_type)
        self.log = log
        self.params = self.__init_params(miner_type)
        self.metrics_obj = self.__get_metrics_type(metrics)

        self.opt = self.__get_opt_type(opt_type)

    def __get_miner_alg(self, miner):
        if miner not in self.miner_mapping:
            raise ValueError(f"Minero '{miner}' no está soportado. Los mineros disponibles son: {list(self.miner_mapping.keys())}")
        return self.miner_mapping[miner]
    
    def __init_params(self, miner_type):
        if miner_type== 'heuristic':
            params = parameters.Heuristic_Parameters.base_params
        return params

    def __get_metrics_type(self, metrics):
        if metrics not in self.metrics_mapping:
            raise ValueError(f"Las métricas '{metrics}' no están soportadas. Las métricas disponibles son: {list(self.metrics_mapping.keys())}")
        return self.metrics_mapping[metrics]

    def __get_opt_type(self, opt_type):
        if opt_type == 'NSGA-II': 
            opt = optimize_Jmetal.Opt_NSGAII(self.miner, self.log, self.metrics_obj)
        else: raise ValueError(f'Optimizador {opt_type} no soportado o es incorrecto')
        return opt
    
    def discover(self, **params):
        return self.opt.discover(**params)
        

## TESTING
if __name__ == "__main__":
        
    from pm4py.objects.log.importer.xes import importer as xes_importer
    from pm4py.visualization.petri_net import visualizer as pn_visualizer
    from jmetal.operator.crossover import SBXCrossover
    from jmetal.operator.mutation import PolynomialMutation
    from jmetal.util.termination_criterion import StoppingByEvaluations
    from jmetal.lab.visualization import Plot


    max_evaluations = 1000

    log = xes_importer.apply('test/Closed/BPI_Challenge_2013_closed_problems.xes')
    p_miner = Process_miner(miner_type='heuristic',
                            opt_type='NSGA-II',
                            metrics='basic',
                            log = log)
    
    p_miner.discover(pop_size=100,
                     off_pop_size=100,
                     mutation = PolynomialMutation(probability=1.0 / p_miner.opt.problem.number_of_variables, distribution_index=20),
                     crossover = SBXCrossover(probability=1.0, distribution_index=20),
                     termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations))
    
    # obtain optimal petri net
    optimal_petri_net, initial_marking, final_marking = p_miner.opt.get_petri_net()

    # visualize petri net
    gviz = pn_visualizer.apply(optimal_petri_net, initial_marking, final_marking)
    pn_visualizer.view(gviz)

    # plot Pareto front
    p_miner.opt.plot_pareto_front(title='Pareto front approximation', label='NSGAII-Pareto-Closed', filename='NSGAII-Pareto-Closed', format='png')