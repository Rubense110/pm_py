from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.algorithm.multiobjective.nsgaii import NSGAII

from src.process_miner import Process_miner
from src.optimize import Optimizer
import src.metrics as metrics

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.discovery.inductive import algorithm as inductive_miner


## Common Parameters ##
max_evaluations = 1000

log = 'event_logs/Closed/BPI_Challenge_2013_closed_problems.xes'
#log = 'event_logs/Financial/BPI_Challenge_2012.xes'

def process_miner_test(test_ID):

    p_miner = Process_miner(miner_type='heuristic',
                        metrics='basic',
                        log = log,)

    nsgaii_params = {'population_size': 100,
                        'offspring_population_size': 100,
                        'mutation': PolynomialMutation(probability=1.0 / p_miner.opt.number_of_variables, distribution_index=20),
                        'crossover': SBXCrossover(probability=1.0, distribution_index=20),
                        'termination_criterion': StoppingByEvaluations(max_evaluations=max_evaluations)}

    p_miner.discover(algorithm_class=NSGAII, **nsgaii_params)

def optimize_test(test_ID):

    log = xes_importer.apply(log)
    metrics_obj = metrics.Basic_Metrics()

    opt = Optimizer(heuristics_miner, log, metrics_obj)

    nsgaii_params = {'population_size': 100,
                     'offspring_population_size': 100,
                     'mutation': PolynomialMutation(probability=1.0 / opt.number_of_variables, distribution_index=20),
                     'crossover': SBXCrossover(probability=1.0, distribution_index=20),
                     'termination_criterion': StoppingByEvaluations(max_evaluations=max_evaluations)}
    
    opt.discover(algorithm_class=NSGAII, **nsgaii_params)
    
    optimal_petri_net, initial_marking, final_marking = opt.get_petri_net()

    # visualize petri net
    gviz = pn_visualizer.apply(optimal_petri_net, initial_marking, final_marking)
    pn_visualizer.view(gviz)

    # plot Pareto front
    opt.plot_pareto_front(title='Pareto front approximation', filename='NSGAII-Pareto-Closed')