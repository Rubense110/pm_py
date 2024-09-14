import numpy as np
from jmetal.core.problem import FloatSolution
import matplotlib.pyplot as plt

# hacer que trabaje con las solution, no solo los objectives
def calculate_pareto_front(solution_objectives):
    solution_objectives = np.array(solution_objectives)
    front = []

    for i, sol1 in enumerate(solution_objectives):
        dominated = False
        for j, sol2 in enumerate(solution_objectives):
            if i!=j:
                if np.all(sol2 <= sol1) and np.any(sol2 < sol1):
                    dominated = True
                    break
        if not dominated and not any(np.array_equal(sol1, f) for f in front):
            front.append(sol1)

    return np.array(front)

def determine_color(index):
    colors = ['red', 'green', 'blue', 'darkslategray', 'purple', 'orange']
    if index > len(colors):
        index = index - len(colors)
        determine_color(index)
    else:
        color = colors[index]
    return color

def plot_pareto_front(objectives, axis_labels, title, filename):
    objectives = np.array(objectives)
    pareto_front = calculate_pareto_front(objectives)
    dominated_solutions = np.setdiff1d(objectives, pareto_front)

    plt.figure(figsize=(10, 6))
    x = np.arange(len(objectives[0]))
                  
    for solution in dominated_solutions:
        plt.plot(x, solution, color= 'paleturqoise')

    for index,optimal_solution in enumerate(pareto_front):
        plt.plot(x, optimal_solution, color= determine_color(index))

    plt.xlabel('Objetivos', fontweight= 550, fontsize = 14)
    plt.ylabel('Valor', fontweight= 550, fontsize = 14)
    plt.title(title, fontweight= 550, fontsize = 20)
    plt.xticks(x, axis_labels)
    plt.grid()
    plt.savefig(filename)

if __name__ == "__main__":
    ## TESTING

    import pandas as pd
    opt_sols_data = pd.read_csv('out/[2024_09_14 - 13:31:38]-BPI_Challenge_2013_closed_problems.xes-NSGA-II/results_objectives.csv')
    pareto_front = calculate_pareto_front(opt_sols_data.values.tolist())
    print(pareto_front)
    #for i in pareto_front:
    #    print(i)
    labels = ['n_places', 'n_arcs', 'n_transitions', 'cycl_complx', 'ratio', 'joins', 'splits']
    plot_pareto_front(opt_sols_data.values.tolist(), labels, title= "pareto", filename='pareto')

