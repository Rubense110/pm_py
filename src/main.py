from process_miner import Process_miner

from jmetal.operator.mutation import *
from jmetal.util.termination_criterion import *
from jmetal.operator.crossover import *
from jmetal.core.quality_indicator import *
from jmetal.algorithm.multiobjective.nsgaii import NSGAII

from enum import Enum


class Log(Enum):
    CLOSED    ='test/Closed/BPI_Challenge_2013_closed_problems.xes'
    FINANCIAL = 'test/Financial/BPI_Challenge_2012.xes'
    INCIDENTS = 'test/Incidents/BPI_Challenge_2013_incidents.xes'
    OPEN      = 'test/Open/BPI_Challenge_2013_open_problems.xes'

class Crossover:
    def __init__(self, probability, distribution_index, perturbation):
        self.probability = probability,
        self.distribution_index = distribution_index, #base value 20
        Crossover_map = {
            'NullCrossover' :       self.__null_crossover(),
            'PMXCrossover' :        self.__pmx_crossover(),
            'CXCrossover' :         self.__cxc_crossover(),
            'SBXCrossover' :        self.__sbx_crossover(),
            'IntegerSBXCrossover' : self.__integer_sbx_crossover(),
            'SPXCrossover' :        self.__spx_crossover(),
            #'DifferentialEvolutionCrossover' :,
            #'CompositeCrossover' :,

        }
    def __null_crossover(self):
        return NullCrossover()
    
    def __pmx_crossover(self):
        return PMXCrossover(probability=self.probability)
    
    def __cxc_crossover(self):
        return CXCrossover(probability=self.probability)
    
    def __sbx_crossover(self):
        return SBXCrossover(probability=self.probability,
                            distribution_index=self.distribution_index)

    def __integer_sbx_crossover(self):
        return IntegerSBXCrossover(probability=self.probability,
                                   distribution_index= self.distribution_index)
    def __spx_crossover(self):
        return SPXCrossover(probability=self.probability)
    
class Mutation:
    def __init__(self, mutation_type, probability, distribution_index, perturbation,  max_iterations):
        mutation_map = {
            'NullMutation' :                self.__null_mutation(),
            'BitFlitMutation' :             self.__bit_flip_mutation(),
            'PolynomialMutation' :          self.__polynomial_mutation(),
            'IntegerPolynomialMutation' :   self.__integer_polynomial_mutation(),
            'SimpleRandomMutation' :        self.__simple_random_mutation(),
            'UniformMutation' :             self.__uniform_mutation(),
            'NonUniformMutation' :          self.__non_uniform_mutation()
            #'PermutationSwapMutation': PermutationSwapMutation(),
            #'CompositeMutation' : CompositeMutation(mutation_operator_list=),
            #'ScrambleMutation' : ScrambleMutation()
        }

        self.probability = probability,
        self.distribution_index = distribution_index,
        self.perturbation = perturbation,
        self.max_iterations = max_iterations

        if mutation_type in mutation_map:
            self.mutation = mutation_map[mutation_type]

    def __null_mutation(self):
        return NullMutation()
    
    def __bit_flip_mutation(self):
        return BitFlipMutation(probability=self.probability)
    
    def __polynomial_mutation(self):
        return PolynomialMutation(probability=self.probability,
                                  distribution_index=self.distribution_index)
    
    def __integer_polynomial_mutation(self):
        return IntegerPolynomialMutation(probability=self.probability,
                                         distribution_index=self.distribution_index)
    
    def __simple_random_mutation(self):
        return SimpleRandomMutation(probability=self.probability)
    
    def __uniform_mutation(self):
        return UniformMutation(probability=self.probability,
                               perturbation=self.perturbation)
    
    def __non_uniform_mutation(self):
        return NonUniformMutation(probability=self.probability,
                                  perturbation=self.perturbation,
                                  max_iterations=self.max_iterations)


class TerminationCriterion:
    def __init__(self, criterion, amount, quality_indicator = FitnessValue(), expected_value = 0.5, degree = 1.0):
        criterion_map = {
            'evaluations':  self.__stop_by_evaluations(),
            'time':         self.__stop_by_time(),
            'quality' :     self.__stop_by_quality_indicator(),
        }
        self.quality_indicator = quality_indicator
        self.expected_value = expected_value
        self.degree = degree
        self.amount = amount
        
        
        if criterion in criterion_map:
            self.criterion = criterion_map[criterion]
        else:
            raise ValueError(f"Criterio de terminación no válido: {criterion}")
        
    def __stop_by_evaluations(self):
        return StoppingByEvaluations(max_evaluations=self.amount)
    
    def __stop_by_time(self):
        return StoppingByTime(max_seconds= self.amount)
    
    def __stop_by_quality_indicator(self):
        return StoppingByQualityIndicator(quality_indicator=self.quality_indicator,
                                          expected_value=self.expected_value, 
                                          degree=self.degree) 


class EvolutionaryAlgorithm:
    def __init__(self, folder, log, population_size, offspring_population_size, mutation, crossover, termination_criterion):

        self.folder = folder # where to store everything 
        self.log = log  
        self.population_size = population_size
        self.offspring_population_size = offspring_population_size
        self.mutation = mutation
        self.crossover = crossover
        self.termination_criterion = termination_criterion
        self.__init_kwargs()
    
    def __init_kwargs(self):
        self.kwargs = { 'population_size' :           self.population_size,
                        'offspring_population_size' : self.offspring_population_size,
                        'mutation':                   self.mutation ,
                        'crossover' :                 self.crossover,
                        'termination_criterion':      self.termination_criterion}

    def 



max_evaluations = 10000
log = Log.CLOSED

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

