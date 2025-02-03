import numpy as np
from jmetal.core.problem import FloatSolution
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from typing import List
import sys
import os
import sqlite3
import json
from contextlib import contextmanager
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from scipy.spatial.distance import euclidean
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import common

def calculate_pareto_front(solutions : List[FloatSolution]):
    front = []
    solution_objectives = np.array([sol.objectives for sol in solutions])
    for i, sol1 in enumerate(solution_objectives):
        dominated = False
        for j, sol2 in enumerate(solution_objectives):
            if i!=j:
                if np.all(sol2 <= sol1) and np.any(sol2 < sol1):
                    dominated = True
                    break
        is_already_in_front = any(np.array_equal(sol1, f.objectives) for f in front)
        if not dominated and not is_already_in_front:
            front.append(solutions[i])

    return front

def determine_color(index):
    colors = ['red', 'green', 'black', 'darkslategray', 'purple', 'orange']
    if index >= len(colors):
        index = index - len(colors)
        determine_color(index)
    else:
        color = colors[index]
    return color

def plot_pareto_front(solutions : List[FloatSolution], axis_labels, title, filename):
    pareto_front_objectives = [sol.objectives for sol in calculate_pareto_front(solutions)]
    dominated_solutions_objectives = [sol.objectives for sol in solutions if (any((sol.objectives==pareto_sol).all()) for pareto_sol in pareto_front_objectives)]

    pareto_front_objectives = [[abs(value) for value in sol] for sol in pareto_front_objectives]
    dominated_solutions_objectives = [[abs(value) for value in sol] for sol in dominated_solutions_objectives]
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(pareto_front_objectives[0]))
                  
    for solution in dominated_solutions_objectives:
       plt.plot(x, solution, color= 'paleturquoise')

    for optimal_solution in pareto_front_objectives:
        plt.plot(x, optimal_solution)

    plt.xlabel('Objetivos', fontweight= 550, fontsize = 14)
    plt.ylabel('Valor', fontweight= 550, fontsize = 14)
    plt.title(title, fontweight= 550, fontsize = 20)
    plt.xticks(x, axis_labels)
    plt.grid()
    plt.savefig(filename)

def petri_net_to_graph(petri_net):
    """
    Convierte una Red de Petri de PM4Py en un grafo dirigido de NetworkX.
    """
    G = nx.DiGraph()
    
    for place in petri_net.places:
        G.add_node(place.name, type='place')
    
    for transition in petri_net.transitions:
        if transition.label is not None:  # Solo agregamos transiciones etiquetadas
            G.add_node(transition.name, type='transition')

    for arc in petri_net.arcs:
        G.add_edge(arc.source.name, arc.target.name)
    
    return G

def load_petris_as_graphs(execution_id):
    """
    Loads stored petri nets from the sqlite DB as networkx directed graphs.
    Returns a list with all petri graphs
    """

    conn = sqlite3.connect(common.DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT places, transitions, arcs FROM petri_nets WHERE execution_id = ?", (execution_id,))
    petri_nets_data = cursor.fetchall()
    conn.close() 

    graphs = []
    for places, transitions, arcs in petri_nets_data:
        G = nx.DiGraph()

        places = json.loads(places)
        for place in places:
            G.add_node(place, type="place")

        transitions = json.loads(transitions)
        for transition in transitions:
            G.add_node(transition, type="transition")

        arcs = json.loads(arcs)
        for arc in arcs: # as they are strings source an target must be divided
            source, target = arc.split("->") 
            G.add_edge(source, target)

        graphs.append(G)
    
    return graphs

def plot_petri_graph(graph, filename):
    """
    Plots a petri graph using matplotlib
    """
    G = graph
    plt.figure(figsize=(16, 12))
    #pos = nx.spring_layout(G)  # Posicionamiento automático
    pos = graphviz_layout(graph, prog="dot", args="-Grankdir=LR -Gnodesep=0.5 -Granksep=1")   # Posicionamiento Izda a dcha

    places = [node for node, data in graph.nodes(data=True) if data.get("type") == "place"]
    transitions = [node for node, data in graph.nodes(data=True) if data.get("type") == "transition"]
    
    labels = {}
    for node, data in graph.nodes(data=True):
        # Si el nodo es de tipo "transition" y contiene una tupla, extraer el segundo elemento de la tupla
        if data.get("type") == "transition":
            node_str = str(node).strip("'()")  # Eliminar paréntesis y comillas
            parts = node_str.split(', ')
            if len(parts) > 1:
                labels[node] = parts[1].strip("'")  # Extraer el segundo elemento (sin las comillas)
            else:
                labels[node] = node  # Si no es una tupla, usar el nombre del nodo tal cual
        else:
            labels[node] = node  # Para los demás nodos, mantener su nombre como etiqueta

    labels = {node: " " if str(node).startswith(("pre_", "intplace_")) else labels.get(node, str(node)) for node in graph.nodes()}
    
    transitions_none = [
        node for node, data in graph.nodes(data=True)
        if data.get("type") == "transition" and labels[node] == "None"
    ]

    transitions_normal = [
        node for node in transitions
        if node != "None" and labels[node] != "None"
    ]

    nx.draw_networkx_nodes(graph, pos, nodelist=places, node_shape="o", node_color="lightblue",
                           edgecolors='black', node_size=5000)
    nx.draw_networkx_nodes(graph, pos, nodelist=transitions_normal, node_shape="s", node_color="lightyellow",
                           edgecolors='black', node_size=5000)
    nx.draw_networkx_nodes(graph, pos, nodelist=transitions_none, node_shape="s", node_color="black",
                           edgecolors='black', node_size=5000)
    
    nx.draw_networkx_edges(graph, pos, edge_color="gray", width=3, arrowstyle='->',
                           arrows=True, arrowsize=50, node_size=5000)

    nx.draw_networkx_labels(graph, pos, labels, font_size=10, font_color="black")
    petri_index = 'p'
    plt.title(f"Red de Petri {petri_index}")
    plt.savefig(filename, format="png")

if __name__ == "__main__":
    import pandas as pd
    ## TESTING

    def test_pareto():
        opt_sols_data = pd.read_csv('out/[2024_09_14 - 13:31:38]-BPI_Challenge_2013_closed_problems.xes-NSGA-II/results_objectives.csv')
        #pareto_front = calculate_pareto_front(opt_sols_data.values.tolist())
        #print(pareto_front)
        #for i in pareto_front:
        #    print(i)
        labels = ['n_places', 'n_arcs', 'n_transitions', 'cycl_complx', 'ratio', 'joins', 'splits']
        plot_pareto_front(opt_sols_data.values.tolist(), labels, title= "pareto", filename='pareto')

    #def compare_test():
        