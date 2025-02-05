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

def shortest_path_matrix(graph):
    """
    Calcula la matriz de distancias de camino más corto entre todos los pares de nodos en un grafo.
    
    :param graph: El grafo de NetworkX
    :return: Matriz de distancias de camino más corto (numpy array)
    """
    # Obtener todos los nodos del grafo
    nodes = list(graph.nodes)
    n = len(nodes)
    
    # Crear una matriz de distancias inicializada con infinitos
    dist_matrix = np.full((n, n), np.inf)
    
    # La distancia de un nodo consigo mismo es 0
    np.fill_diagonal(dist_matrix, 0)
    
    # Calcular la distancia de camino más corto entre todos los pares de nodos
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if u != v:
                try:
                    dist_matrix[i, j] = nx.shortest_path_length(graph, source=u, target=v)
                except nx.NetworkXNoPath:
                    dist_matrix[i, j] = np.inf  # Si no hay camino entre u y v

    return dist_matrix, nodes

def shortest_path_distance(graph1, graph2):
    """
    Calcula la distancia entre dos grafos usando la distancia de camino más corto.
    
    :param graph1: Primer grafo
    :param graph2: Segundo grafo
    :return: Distancia entre los dos grafos
    """
    # Calcular las matrices de distancias de camino más corto para ambos grafos
    dist_matrix1, nodes1 = shortest_path_matrix(graph1)
    dist_matrix2, nodes2 = shortest_path_matrix(graph2)
    
    # Asegurarse de que los nodos de ambos grafos estén alineados
    assert nodes1 == nodes2, "Los nodos de los dos grafos deben ser los mismos."
    
    # Calcular la diferencia entre las matrices
    dist_diff = np.abs(dist_matrix1 - dist_matrix2)
    
    # Utilizar la norma L1 para comparar las matrices
    graph_distance = np.sum(dist_diff)
    
    return graph_distance

def resistance_matrix_eg(graph):
    """
    Calcula la matriz de resistencia gráfica efectiva entre todos los pares de nodos en un grafo.
    
    :param graph: El grafo de NetworkX
    :return: Matriz de resistencias gráficas efectivas (numpy array)
    """
    # Obtener la matriz de adyacencia
    A = nx.adjacency_matrix(graph).todense()
    
    # Calcular la matriz de grados D
    D = np.diag([deg for node, deg in graph.degree()])
    
    # Calcular la matriz Laplaciana L = D - A
    L = D - A
    
    # Calcular la inversa de la matriz Laplaciana
    L_inv = np.linalg.inv(L)
    
    # Crear una matriz de resistencias
    n = len(graph.nodes)
    resistance_matrix = np.zeros((n, n))
    
    # Calcular la distancia de resistencia efectiva entre cada par de nodos
    for i in range(n):
        for j in range(n):
            if i != j:
                resistance_matrix[i, j] = L_inv[i, i] + L_inv[j, j] - 2 * L_inv[i, j]
    
    return resistance_matrix

def effective_graph_resistance(graph1, graph2):
    """
    Calcula la distancia entre dos grafos utilizando la resistencia gráfica efectiva.
    
    :param graph1: Primer grafo
    :param graph2: Segundo grafo
    :return: Distancia entre los dos grafos
    """
    # Calcular las matrices de resistencia gráfica para ambos grafos
    resistance_matrix1 = resistance_matrix_eg(graph1)
    resistance_matrix2 = resistance_matrix_eg(graph2)
    
    # Calcular la diferencia entre las matrices
    resistance_diff = np.abs(resistance_matrix1 - resistance_matrix2)
    
    # Utilizar la norma L1 para comparar las matrices
    graph_distance = np.sum(resistance_diff)
    
    return graph_distance

def edit_distance(graph1, graph2):
    """
    Calcula la Edit Distance entre dos grafos utilizando sus matrices de adyacencia.
    
    :param graph1: Primer grafo
    :param graph2: Segundo grafo
    :return: Distancia de edición entre los dos grafos
    """
    # Obtener las matrices de adyacencia de ambos grafos
    A1 = nx.adjacency_matrix(graph1).todense()
    A2 = nx.adjacency_matrix(graph2).todense()
    
    # Calcular la diferencia entre las matrices de adyacencia
    diff = np.abs(A1 - A2)
    
    # Calcular la suma de las diferencias absolutas (norma L1)
    distance = np.sum(diff)
    
    return distance

def resistance_matrix(graph):
    """
    Calcula la matriz de resistencias de un grafo usando la matriz de Laplaciana.
    """
    # Calcular la matriz de Laplaciana
    L = nx.laplacian_matrix(graph).todense()

    # Invertir la matriz Laplaciana (excluyendo las filas y columnas correspondientes a un nodo)
    n = len(graph.nodes)
    R = np.linalg.pinv(L)  # Matriz de resistencias

    # Retornar la matriz de resistencias
    return R

def resistance_perturbation_distance(graph1, graph2):
    """
    Calcula la Resistance-Perturbation Distance entre dos grafos.
    
    :param graph1: Primer grafo
    :param graph2: Segundo grafo
    :return: Distancia de perturbación entre los dos grafos
    """
    # Calcular la matriz de resistencias de ambos grafos
    R1 = resistance_matrix(graph1)
    R2 = resistance_matrix(graph2)
    
    # Calcular la diferencia entre las matrices de resistencias
    diff = np.abs(R1 - R2)
    
    # Calcular la norma L1 de la diferencia (suma de los valores absolutos)
    distance = np.sum(diff)
    
    return distance

def delta_con_distance(graph1, graph2, alpha=1.0):
    """
    Calcula la DELTACON Distance entre dos grafos usando la propagación rápida de creencias.
    
    :param graph1: Primer grafo (NetworkX Graph)
    :param graph2: Segundo grafo (NetworkX Graph)
    :param alpha: Parámetro de difusión (default: 1.0)
    :return: Distancia DELTACON entre los dos grafos
    """
    # Obtener la matriz de adyacencia de cada grafo
    A1 = nx.adjacency_matrix(graph1).todense()
    A2 = nx.adjacency_matrix(graph2).todense()
    
    # Calcular la matriz de propagación para cada grafo
    I = np.eye(len(graph1.nodes))
    S1 = (I + alpha * A1)**2
    S2 = (I + alpha * A2)**2
    
    # Calcular la distancia de Matusita entre las matrices S1 y S2
    delta_con_distance = np.sqrt(np.sum((np.sqrt(S1) - np.sqrt(S2))**2))
    
    return delta_con_distance

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

def plot_all_petri_distances(filename: str, petri_graphs: list, distances: list):
    """
    Genera una figura con múltiples subgráficas, cada una mostrando una matriz de distancias 
    entre redes de Petri utilizando diferentes métodos de distancia.
    
    :param filename: Nombre base del archivo para guardar la figura generada.
    :param petri_graphs: Lista de redes de Petri a comparar.
    :param distances: Lista de funciones de distancia que se aplicarán entre los grafos.
    """
    # Crear la figura con un número adecuado de subgráficas
    n_distances = len(distances)
    fig, axes = plt.subplots(1, n_distances, figsize=(18, 12))  # 1 fila, n columnas

    if n_distances == 1:  # Caso cuando solo hay una distancia, no se necesita una lista de ejes
        axes = [axes]
    
    # Crear matrices de distancias para cada tipo de distancia
    n = len(petri_graphs)

    for idx, dist_func in enumerate(distances):
        # Inicializar la matriz de distancias
        distance_matrix = np.zeros((n, n))

        # Calcular las distancias entre todos los pares de redes de Petri
        for i, j in itertools.combinations(range(n), 2):
            dist = dist_func(petri_graphs[i], petri_graphs[j])
            distance_matrix[i, j] = distance_matrix[j, i] = dist

        # Crear el mapa de calor para cada subgráfico
        sns.heatmap(distance_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=axes[idx])
        
        # Título y etiquetas
        axes[idx].set_title(f"Matriz de Distancia: {dist_func.__name__}", fontsize=14)
        axes[idx].set_xlabel("Red de Petri nº", fontsize=12)
        axes[idx].set_ylabel("Red de Petri nº", fontsize=12)

    # Ajustar el layout para que las subgráficas no se solapen
    plt.tight_layout()

    # Guardar la figura con todos los mapas de calor
    plt.savefig(f"{filename}_all_distances.png", dpi=300, bbox_inches='tight')
    plt.close()  # Cerrar la figura para liberar memoria

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
        