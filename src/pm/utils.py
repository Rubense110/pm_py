import numpy as np
from jmetal.core.problem import FloatSolution
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyArrowPatch
from typing import List
import sys
import itertools
import os
import sqlite3
import re
from itertools import combinations

import json
from contextlib import contextmanager
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from scipy.spatial.distance import euclidean
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import seaborn as sns
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
            if transition.startswith("(") and transition.endswith(")"):
                transition = tuple(e.strip("'") for e in transition.strip("()").split(", "))[0]
            G.add_node(transition, type="transition")

        arcs = json.loads(arcs)
        for arc in arcs: # as they are strings source an target must be divided
            source, target = arc.split("->") 

            if source.startswith("(") and source.endswith(")"):
                source = tuple(e.strip("'") for e in source.strip("()").split(", "))[0]

            if target.startswith("(") and target.endswith(")"):
                target = tuple(e.strip("'") for e in target.strip("()").split(", "))[0]

            G.add_edge(source, target)

        #print(G.nodes(data=True), "\n")
        #print(G.edges())
        #print("---------")
        graphs.append(G)
    
    return graphs

def plot_petri_graph(graph, filename):
    """
    Plots a petri graph using matplotlib
    """
    G = graph
    plt.figure(figsize=(24, 16))
    #pos = nx.spring_layout(G)  # Posicionamiento automático
    pos = graphviz_layout(graph, prog="dot", args="-Grankdir=LR -Gnodesep=0.5 -Granksep=1")   # Posicionamiento Izda a dcha

    places = [node for node, data in graph.nodes(data=True) if data.get("type") == "place"]
    transitions = [node for node, data in graph.nodes(data=True) if data.get("type") == "transition"]

    labels = {}
    for node, data in graph.nodes(data=True):
        # Si el nodo es de tipo "transition" y contiene una tupla, extraer el segundo elemento de la tupla
        if data.get("type") == "transition":
            node_data = node  # Asumimos que 'node' ya es una tupla
            if isinstance(node_data, tuple):  # Verificamos si es una tupla
                if len(node_data) > 1:
                    labels[node] = node_data[1]  # Extraemos el segundo elemento
                else:
                    labels[node] = str(node_data[0])  # Si es una tupla de un solo elemento, usamos ese
            else:
                labels[node] = str(node_data)  # Si no es tupla, lo tratamos como cadena
        else:
            labels[node] = str(node)  # Para los demás nodos, mantener su nombre como etiqueta

    # Asignamos etiquetas vacías a nodos especiales
    labels = {node: " " if str(node).startswith(("pre_", "intplace_")) else labels.get(node, str(node)) for node in graph.nodes()}
    print("labels: ", labels)
    # Filtrar transiciones con etiquetas "None"
    transitions_none = [
        node for node, data in graph.nodes(data=True)
        if data.get("type") == "transition" and re.match(r"^hid_\d+$", labels[node])
    ]

    transitions_normal = [
        node for node in transitions
        if node != "None" and labels[node] != "None"
    ]

    # Dibujamos los nodos y las transiciones
    nx.draw_networkx_nodes(graph, pos, nodelist=places, node_shape="o", node_color="lightblue",
                           edgecolors='black', node_size=5000)
    nx.draw_networkx_nodes(graph, pos, nodelist=transitions_normal, node_shape="s", node_color="lightyellow",
                           edgecolors='black', node_size=5000)
    nx.draw_networkx_nodes(graph, pos, nodelist=transitions_none, node_shape="s", node_color="black",
                           edgecolors='black', node_size=5000)
    
    # Dibujamos los bordes y las etiquetas
    nx.draw_networkx_edges(graph, pos, edge_color="gray", width=3, arrowstyle='->',
                           arrows=True, arrowsize=50, node_size=5000)

    nx.draw_networkx_labels(graph, pos, labels, font_size=10, font_color="black")
    
    petri_index = 'p'
    plt.title(f"Red de Petri {petri_index}")
    plt.savefig(filename, format="png")

def plot_petri_graph_(graph, filename):
    """
    Plots a petri graph using matplotlib
    """
    G = graph
    plt.figure(figsize=(24, 16))
    #pos = nx.spring_layout(G)  # Posicionamiento automático
    pos = graphviz_layout(graph, prog="dot", args="-Grankdir=LR -Gnodesep=0.5 -Granksep=1")   # Posicionamiento Izda a dcha

    places = [node for node, data in graph.nodes(data=True) if data.get("type") == "place"]
    transitions = [node for node, data in graph.nodes(data=True) if data.get("type") == "transition"]

    nx.draw_networkx_nodes(graph, pos, nodelist=places, node_shape="o", node_color="lightblue",
                           edgecolors='black', node_size=5000)
    nx.draw_networkx_nodes(graph, pos, nodelist=transitions, node_shape="s", node_color="lightyellow",
                           edgecolors='black', node_size=5000)
    
    nx.draw_networkx_edges(graph, pos, edge_color="gray", width=3, arrowstyle='->',
                           arrows=True, arrowsize=50, node_size=5000)

    petri_index = 'p'
    plt.title(f"Red de Petri {petri_index}")
    plt.savefig(filename, format="png")

def adjacency_spectral_distance(G1, G2):
    # Obtener matrices de adyacencia
    A1 = nx.adjacency_matrix(G1).todense()
    A2 = nx.adjacency_matrix(G2).todense()

    # Calcular espectros (autovalores)
    eigs1 = np.linalg.eigvalsh(A1)
    eigs2 = np.linalg.eigvalsh(A2)

    # Ordenar los autovalores de mayor a menor
    eigs1 = np.sort(eigs1)[::-1]
    eigs2 = np.sort(eigs2)[::-1]

    # Igualar la longitud si los grafos tienen diferente número de nodos
    n = max(len(eigs1), len(eigs2))
    eigs1 = np.pad(eigs1, (0, n - len(eigs1)), mode='constant')
    eigs2 = np.pad(eigs2, (0, n - len(eigs2)), mode='constant')

    # Calcular la distancia euclidiana entre los espectros
    distance = np.linalg.norm(eigs1 - eigs2)
    
    return distance

def plot_petri_distances(filename:str, petri_graphs:list, distance= adjacency_spectral_distance):
    
    # Crear matriz de distancias
    n = len(petri_graphs)
    distance_matrix = np.zeros((n, n))

    for i, j in itertools.combinations(range(n), 2):
        dist = distance(petri_graphs[i], petri_graphs[j])
        distance_matrix[i, j] = distance_matrix[j, i] = dist

    #print("Matriz de distancias espectrales:")
    #print(distance_matrix)

    plt.figure(figsize=(12,12))
    sns.heatmap(distance_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Matriz de Distancia Espectral entre Redes de Petri", fontsize=14)
    plt.xlabel("Red de Petri nº", fontsize=12)
    plt.ylabel("Red de Petri nº", fontsize=12)
    plt.savefig(filename, dpi=300, bbox_inches='tight')

def compare_petri_nets(G_A, G_B):
    """Genera matrices de comparación de nodos y aristas entre dos redes de Petri."""
    nodes_A = set(G_A.nodes())
    nodes_B = set(G_B.nodes())
    edges_A = set(G_A.edges())
    edges_B = set(G_B.edges())
    
    # Lista de nodos combinada
    all_nodes = sorted(nodes_A | nodes_B)
    node_index = {node: i for i, node in enumerate(all_nodes)}
    
    # Matriz de Nodos (M_N)
    M_N = np.zeros((len(all_nodes), 1), dtype=int)
    for node in nodes_A:
        M_N[node_index[node], 0] += 1  # Nodo en A
    for node in nodes_B:
        M_N[node_index[node], 0] += 2  # Nodo en B
    
    # Matriz de Aristas (M_E)
    all_edges = sorted(edges_A | edges_B)
    edge_index = {edge: i for i, edge in enumerate(all_edges)}
    M_E = np.zeros((len(all_edges), 1), dtype=int)
    for edge in edges_A:
        M_E[edge_index[edge], 0] += 1  # Arista en A
    for edge in edges_B:
        M_E[edge_index[edge], 0] += 2  # Arista en B
    
    return M_N, M_E, nodes_A, nodes_B, edges_A, edges_B

def get_all_paths(G, start, end):
    """ Encuentra todos los caminos desde start hasta end en un grafo dirigido """
    return list(nx.all_simple_paths(G, start, end))

def compare_structural_differences(G_A, G_B):
    """Compara dos redes de Petri e identifica nodos y caminos únicos."""
    M_N, M_E, nodes_A, nodes_B, edges_A, edges_B = compare_petri_nets(G_A, G_B)

    # Identificar diferencias estructurales
    unique_nodes_A = nodes_A - nodes_B
    unique_nodes_B = nodes_B - nodes_A
    common_nodes = nodes_A & nodes_B

    unique_edges_A = edges_A - edges_B
    unique_edges_B = edges_B - edges_A
    common_edges = edges_A & edges_B

    # Encontrar nodos de inicio y fin
    start_nodes = [n for n in common_nodes if G_A.in_degree(n) == 0 or G_B.in_degree(n) == 0]
    end_nodes = [n for n in common_nodes if G_A.out_degree(n) == 0 or G_B.out_degree(n) == 0]

    # Comparación de caminos
    paths_A = set()
    paths_B = set()
    for start in start_nodes:
        for end in end_nodes:
            paths_A.update(tuple(p) for p in get_all_paths(G_A, start, end))
            paths_B.update(tuple(p) for p in get_all_paths(G_B, start, end))
    
    common_paths = paths_A & paths_B
    exclusive_A = paths_A - paths_B
    exclusive_B = paths_B - paths_A

    # Generar resumen
    summary = f"""
    🔹 Nodos exclusivos de A: {unique_nodes_A}
    🔹 Nodos exclusivos de B: {unique_nodes_B}
    🔹 Nodos comunes: {common_nodes}

    🔹 Aristas exclusivas de A: {unique_edges_A}
    🔹 Aristas exclusivas de B: {unique_edges_B}
    🔹 Aristas comunes: {common_edges}

    🔹 Caminos comunes: {common_paths}
    🔹 Caminos exclusivos de A: {exclusive_A}
    🔹 Caminos exclusivos de B: {exclusive_B}
    """

    return summary

def compare_multiple_petri_nets(output_dir, petri_graphs):
    """Compara una lista de redes de Petri y guarda los resultados en archivos de texto."""
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  

    preprocessed_graphs = [preprocess_petri_net(G) for G in petri_graphs]

    num_graphs = len(preprocessed_graphs)
    for i in range(num_graphs):
        for j in range(i + 1, num_graphs):
            G_A = preprocessed_graphs[i]
            G_B = preprocessed_graphs[j]

            comparison_result = compare_structural_differences(G_A, G_B)

            file_name = f"compare_{i}_{j}.txt"
            file_path = os.path.join(output_dir, file_name)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(comparison_result)

            print(f"Comparaciones (Matrix) guardadas en: {file_path}")


def preprocess_petri_net(G):
    """
    Preprocesa una red de Petri eliminando las transiciones silenciosas 
    y conectando directamente los nodos entrantes con los salientes. 
    Esto facilita las cosas al comparar las redes.
    """
    G = G.copy()  # Trabajamos sobre una copia para no modificar el original
    print(G.nodes())
    
    silent_transitions = [node for node in G.nodes if 'hid' in node]

    for st in silent_transitions:
        predecessors = list(G.predecessors(st))
        successors = list(G.successors(st))

        # Conectar los predecesores con los sucesores
        for pred in predecessors:
            for succ in successors:
                G.add_edge(pred, succ)

        # Eliminar la transición silenciosa
        G.remove_node(st)

    print('silent_transitions', silent_transitions)
    return G

def find_max_common_subgraph(G1, G2):
    """
    Encuentra el mayor subgrafo común entre dos grafos dirigidos.
    """
    matcher = nx.algorithms.isomorphism.DiGraphMatcher(G1, G2)
    max_common_subgraph = None
    max_size = 0

    for subgraph in matcher.subgraph_isomorphisms_iter():
        subgraph_size = len(subgraph)
        if subgraph_size > max_size:
            max_size = subgraph_size
            max_common_subgraph = subgraph

    return max_common_subgraph

def analyze_similar_petrinets_MSC(output_dir, petri_graphs, top_n=4):
    """
    Analiza una lista de redes de Petri y devuelve los top_n pares con menor adjacency spectral distance,
    junto con su mayor subgrafo común.
    
    :param petri_nets: Lista de redes de Petri (grafos dirigidos).
    :param top_n: Número de pares a seleccionar (por defecto 4).
    :return: Lista de tuplas con los pares seleccionados, su distancia y su mayor subgrafo común.
    """
    # Paso 1: Calcular adjacency spectral distance para todos los pares
    distances = []
    for G1, G2 in combinations(petri_graphs, 2):
        distance = adjacency_spectral_distance(G1, G2)
        distances.append((G1, G2, distance))

    # Paso 2: Ordenar los pares por distancia (de menor a mayor)
    distances.sort(key=lambda x: x[2])

    # Paso 3: Seleccionar los top_n pares con menor distancia
    selected_pairs = distances[:top_n]

    # Paso 4: Calcular el mayor subgrafo común para los pares seleccionados
    results = []
    for idx, (G1, G2, distance)  in enumerate(selected_pairs):
        mcs = find_max_common_subgraph(G1, G2)
        if mcs:
            common_graph = nx.DiGraph()
            for node in mcs.keys():
                common_graph.add_node(node)
            for u, v in G1.edges():
                if u in mcs and v in mcs:
                    common_graph.add_edge(u, v)
            filename = os.path.join(output_dir, f"msc_{idx}_{len(results)}.png")
            draw_graph(common_graph, filename, title=f"MSC (Distancia = {distance:.2f})")
            results.append((G1, G2, distance, mcs))
        else: 
            print(f"No se encontró un subgrafo común para el par con distancia {distance}.")

    for G1, G2, distance, mcs in results:
        print(f"Par de redes con distancia {distance}:")
        print(f"Mayor subgrafo común: {mcs}")
        print("---")

    return results


# Función para dibujar un grafo
def draw_graph(graph, filename, title="Grafo"):
    """
    Dibuja un grafo dirigido utilizando matplotlib.
    """
    pos = graphviz_layout(graph, prog="dot", args="-Grankdir=LR -Gnodesep=0.5 -Granksep=1") 
    plt.figure(figsize=(10, 6))
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
    plt.title(title)
    plt.savefig(filename, dpi=300, bbox_inches='tight')


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
        