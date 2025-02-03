import sys
import os
import psutil
from dask.distributed import Client
from distributed import LocalCluster
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.evaluator import MultiprocessEvaluator
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pm')))

import common
from pm.process_miner import ProcessMiner
from pm import parameters
from pm import config

def run_process_mining_dask():

    client = Client(LocalCluster(n_workers=8))
    ncores = sum(client.ncores().values())
    print(f"{ncores} cores available")

    log = common.log_closed
    max_evaluations = 1000

    p_miner = ProcessMiner(miner_name='heuristic',
                           metrics='basic',
                           log=log)

    d_nsgaii_params = {
        'population_size': 100,
        'mutation': PolynomialMutation(probability=0.17, distribution_index=20),
        'crossover': SBXCrossover(probability=1.0, distribution_index=20),
        'termination_criterion': StoppingByEvaluations(max_evaluations=max_evaluations),
        'number_of_cores': ncores,
        'client': client
    }

    p_miner.discover(algorithm_name='NSGAII-D', **d_nsgaii_params)
    p_miner.show_pareto_iterations()
    print(p_miner.end_time - p_miner.star_time)

def parallel_nsgaii(log):

    p_miner = ProcessMiner(miner_name='heuristic',
                           metrics='quality',
                           log=log)
    cpu_cores= psutil.cpu_count(logical=False)

    parallel_params = {
        "population_evaluator": MultiprocessEvaluator(cpu_cores),
        "problem": p_miner.opt.problem,
        "population_size": 100,
        "offspring_population_size": 100,
        "mutation": PolynomialMutation(probability=0.17, distribution_index=20),
        "crossover": SBXCrossover(probability=1.0, distribution_index=20),
        "termination_criterion": StoppingByEvaluations(max_evaluations=1000)
    }
    
    p_miner.parallel_discover(**parallel_params)
    p_miner.show_pareto_iterations()
    p_miner.save_petris_graphs()

    print(p_miner.end_time - p_miner.star_time)

if __name__ == "__main__":
    log = common.log_closed

    #run_process_mining_dask()
    parallel_nsgaii(log)
    
    # 4 cores 8 hilos

    ## Closed
    # Parallel 8 workers -> 184.78
    # Parallel 4 workers -> 255.39
    # No Parallel        -> 534.92 - 558.07

    ## Open
    # Parallel 8 workers -> 45.38
    # No Parallel        -> 157.29