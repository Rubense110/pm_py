# activate venv -> .\.venv\Scripts\activate

from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.log.importer.xes import importer as xes_importer

import time
import os

import optimize
import metrics 
import utils

class Process_miner:

    log_file = 'doc/log.csv'

    miner_mapping = {
        'inductive': inductive_miner, 
        'heuristic': heuristics_miner,
    }

    metrics_mapping = {
        'basic': metrics.Basic_Metrics(),
        'basic_useful_simple': metrics.Basic_Metrics_Usefull_simple()
    }

    def __init__(self, miner_type, metrics,  log, verbose):

        self.miner_type = miner_type
        self.metrics_type = metrics
        self.log_name = os.path.basename(log)
        self.verbose = verbose

        self.miner = self.__get_miner_alg(miner_type)
        self.log = xes_importer.apply(log)
        self.metrics_obj = self.__get_metrics_type(metrics)

        ## TO-DO -> Manage exception for non supported algos
        self.opt = optimize.Optimizer(self.miner, self.log, self.metrics_obj, verbose)

        self.local_time = time.strftime("[%Y_%m_%d - %H:%M:%S]", time.localtime())
        self.star_time = time.time()

    def __get_miner_alg(self, miner):
        if miner not in self.miner_mapping:
            raise ValueError(f"Minero '{miner}' no está soportado. Los mineros disponibles son: {list(self.miner_mapping.keys())}")
        return self.miner_mapping[miner]

    def __get_metrics_type(self, metrics):
        if metrics not in self.metrics_mapping:
            raise ValueError(f"Las métricas '{metrics}' no están soportadas. Las métricas disponibles son: {list(self.metrics_mapping.keys())}")
        return self.metrics_mapping[metrics]
    
    def __log(self):
        if os.path.isfile(self.log_file):
            with open(self.log_file, 'a') as log:
                runtime = str(self.end_time - self.star_time)
                log.write(f'\n{self.local_time};{self.verbose};{runtime};{self.log_name};{self.miner_type};{self.opt_type};{self.__extract_params()};{self.metrics_type};{self.opt.get_best_solution().variables}')
        else:
            with open(self.log_file, 'w') as log:
                log.write('Timestamp, Verbosity, Runtime, Log Name, Miner Type, Opt type, Opt Parameters, Metrics type, Optimal solution')
            self.__log()

    def __extract_params(self):
        dicc = dict()
        for (attr_name, attr_value) in self.params.items():
            if hasattr(attr_value, '__dict__'):
                dicc[attr_name] = [attr_value.__class__, attr_value.__dict__]
            else:
                dicc[attr_name] = attr_value
        return dicc
    
    def __save(self):
        outpath = f'out/{self.local_time}-{self.log_name}-{self.opt_type}'
        os.makedirs(outpath, exist_ok=True)

        for index,petri in enumerate(self.opt.get_pareto_front_petri_nets()):
            gviz = pn_visualizer.apply(petri[0], petri[1], petri[2])
            pn_visualizer.save(gviz, f'{outpath}/petri_pareto_{index}.png')

        self.opt.plot_pareto_front(title='Pareto front approximation', filename=f'{outpath}/Pareto Front')

        with open(f"{outpath}/results_variables.csv", 'w') as log:
            parameter_names = ",".join(self.opt.parameters_info.base_params.keys())
            log.write(parameter_names+"\n")
            for sol in self.opt.get_result():
                log.write(f'{",".join(map(str, sol.variables))}\n')

        with open(f"{outpath}/results_objectives.csv", 'w') as log:
            metrics_labels = ",".join(self.metrics_obj.get_labels())
            log.write(metrics_labels+"\n")
            for sol in self.opt.get_result():
                log.write(f'{",".join(map(str, sol.objectives))}\n')

    def discover(self, algorithm_class, **params):
        self.params = params
        disc = self.opt.discover(algorithm_class=algorithm_class, **params)
        self.opt_type = algorithm_class.__name__
        self.end_time = time.time()
        self.__log()
        self.__save()
        return disc
    
    def set_log_file(self, logpath):
        self.log_file = logpath

## TESTING
if __name__ == "__main__":
        
    from pm4py.visualization.petri_net import visualizer as pn_visualizer
    from jmetal.operator.crossover import SBXCrossover
    from jmetal.operator.mutation import PolynomialMutation
    from jmetal.util.termination_criterion import StoppingByEvaluations



    max_evaluations = 10000

    log = 'test/Closed/BPI_Challenge_2013_closed_problems.xes'
    #log = 'test/Financial/BPI_Challenge_2012.xes'
    
    p_miner = Process_miner(miner_type='heuristic',
                            metrics='basic_useful_simple',
                            log = log, 
                            verbose = 0)
    
    nsgaii_params = {'population_size': 100,
                     'offspring_population_size': 100,
                     'mutation': PolynomialMutation(probability=1.0 / p_miner.opt.number_of_variables, distribution_index=20),
                     'crossover': SBXCrossover(probability=1.0, distribution_index=20),
                     'termination_criterion': StoppingByEvaluations(max_evaluations=max_evaluations)}
    
    p_miner.discover(algorithm_class=NSGAII, **nsgaii_params)
