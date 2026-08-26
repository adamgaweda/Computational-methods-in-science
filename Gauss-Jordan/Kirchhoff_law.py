import numpy as np
import matplotlib.pyplot as plt
import random
import networkx as nx

def zapis(G, filename):
    with open(filename, 'w') as f:
        for u, v, data in G.edges(data=True):
            if 'resistance' in data:
                line = f"{u} {v} {data['resistance']}\n"
                f.write(line)


def odczyt(filename):
    G = nx.Graph()
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            try:
                u, v, resistance = line.split()
                u = int(u)
                v = int(v)
                resistance = float(resistance)

                G.add_edge(u, v, resistance=resistance)

            except ValueError as e:
                continue
    return G

###GENERACJA GRAFÓW
def assign_resistance(G):
    for u,v in G.edges():
        if 'resistance' not in G.edges[u,v]:
            G.edges[u,v]['resistance'] = random.uniform(0.1,20)
    return G
def generate_graf_ER(n=20,p=0.5):
    G = nx.erdos_renyi_graph(n=n, p=p)
    while not nx.is_connected(G):
        G = nx.erdos_renyi_graph(n=n, p=p)
    return assign_resistance(G)
def generate_graf_3regular(n):
    if n%2 != 0: n+=1
    G = nx.random_regular_graph(3,n)
    while not nx.is_connected(G):
        G = nx.random_regular_graph(3, n)
    return assign_resistance(G)
def generate_graf_bridge_connect(n1,n2,p=0.4):
    G1 = nx.erdos_renyi_graph(n1, p)
    while not nx.is_connected(G1):
        G1 = nx.erdos_renyi_graph(n1, p)
    G2 = nx.erdos_renyi_graph(n2, p)
    while not nx.is_connected(G2):
        G2 = nx.erdos_renyi_graph(n2, p)
    G = nx.disjoint_union(G1, G2)
    G.add_edge(0, n1)
    return assign_resistance(G)
def generate_graf_grid(rows,cols):
    G = nx.grid_2d_graph(rows,cols)
    G = nx.convert_node_labels_to_integers(G)
    return assign_resistance(G)
def generate_graf_world(n=20,k=4,p=0.2):
    G = nx.watts_strogatz_graph(n,k,p)
    while not nx.is_connected(G):
        G = nx.watts_strogatz_graph(n, k, p)
    return assign_resistance(G)


def circut_solver(G,s,t,E):
    nodes = list(G.nodes)
    other_nodes = [n for n in nodes if n!=s and n!=t]
    node_to_idx = {n: i for i, n in enumerate(other_nodes)}
    n = len(nodes) -2
    if n <=2:
        DiG = nx.DiGraph()
        DiG.add_nodes_from([s,t])
        if G.has_edge(s,t):
            R = G[s][t]
            I = E/R
            DiG.add_edge(s,t,current=I)
            I = -I
        else:
            I = 0.0
        if I>=0:
            DiG.add_edge(s,t,current=I, voltage_source=True)
        else:
            DiG.add_edge(t,s,current=-I,voltage_source=True)
        return DiG

    A = np.zeros((n,n))
    B = np.zeros(n)
    for i, node in enumerate(other_nodes):
        if node == s or node == t:continue
        sum_cond = 0.0
        for neighbour in G.neighbors(node):
            R = G[node][neighbour]['resistance']
            conductance = 1.0 / R
            if neighbour == s:
                B[i] += conductance * E
                sum_cond += conductance
            elif neighbour == t:
                sum_cond += conductance
            else:
                j = node_to_idx[neighbour]
                A[i, j] -= conductance
                sum_cond += conductance
        A[i, i] = sum_cond

    try:
        voltage_matrix = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        raise ValueError("Macierz osobliwa, aleee jaaaak?")

    voltages = {n: 0.0 for n in nodes}
    voltages[t] = 0.0
    voltages[s] = E
    for i,node in enumerate(other_nodes):
        voltages[node] = voltage_matrix[i]

    currents = {}
    for u, v in G.edges():
        R = G[u][v]['resistance']
        voltage_u = voltages[u]
        voltage_v = voltages[v]
        current = (voltage_u - voltage_v) / R
        currents[(u, v)] = current


    ####
    DiG = nx.DiGraph()
    for u, v in G.edges():
        current = currents[(u, v)]
        if current >= 0:
            DiG.add_edge(u, v, current=abs(current))
        else:
            DiG.add_edge(v, u, current=abs(current))

    I = 0.0
    for neighbor in G.neighbors(s):
        R = G[s][neighbor]['resistance']
        current = (E - voltages[neighbor]) / R
        I += current
    I = -I
    if I >= 0:
        DiG.add_edge(s, t, current=abs(I), voltage_source=True)
    else:
        DiG.add_edge(t, s, current=abs(I), voltage_source=True)

    return DiG



def visualize_graph(DiG):
    plt.figure(figsize=(14, 10))

    # Kamada-Kawai zazwyczaj o wiele lepiej układa grafy obwodów niż spring_layout
    pos = nx.kamada_kawai_layout(DiG)

    resistor_edges = []
    voltage_source_edges = []

    for u, v, d in DiG.edges(data=True):
        if d.get('voltage_source', False):
            voltage_source_edges.append((u, v))
        else:
            resistor_edges.append((u, v))

    resistor_currents = [DiG[u][v]['current'] for u, v in resistor_edges]
    vs_currents = [DiG[u][v]['current'] for u, v in voltage_source_edges]
    all_currents = resistor_currents + vs_currents

    if not all_currents:
        print("Brak prądów do narysowania.")
        return

    # Zmiana colormapy na 'plasma' (lepszy kontrast dla obwodów)
    cmap = plt.cm.plasma
    norm = plt.Normalize(min(all_currents), max(all_currents))

    is_small_graph = len(DiG) < 100
    is_micro_graph = len(DiG) <= 20

    if is_small_graph:
        nx.draw_networkx_nodes(
            DiG, pos,
            node_color='#a0c4ff',
            node_size=500,
            edgecolors='black',
            linewidths=1.5
        )
        nx.draw_networkx_labels(DiG, pos, font_size=12, font_weight='bold')

    # Rysowanie rezystorów z lekkim wygięciem (connectionstyle) zapobiega nakładaniu się krawędzi
    if resistor_edges:
        nx.draw_networkx_edges(
            DiG, pos, edgelist=resistor_edges, edge_color=resistor_currents,
            edge_cmap=cmap, edge_vmin=min(all_currents), edge_vmax=max(all_currents),
            width=2.5, arrows=True, arrowstyle='-|>', arrowsize=18,
            connectionstyle="arc3,rad=0.1"
        )

    # Rysowanie źródła napięcia (wyraźnie odznaczające się, inny kąt wygięcia)
    if voltage_source_edges:
        nx.draw_networkx_edges(
            DiG, pos, edgelist=voltage_source_edges, edge_color='#ff0055',
            width=4, arrows=True, arrowstyle='-[', arrowsize=20,
            connectionstyle="arc3,rad=-0.2"
        )

    # Etykiety prądów
    if is_micro_graph:
        edge_labels = {(u, v): f"{d['current']:.2f}A" for u, v, d in DiG.edges(data=True)}
        nx.draw_networkx_edge_labels(
            DiG, pos, edge_labels=edge_labels,
            font_size=10, font_color='black',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='gray', alpha=0.9),
            rotate=False, label_pos=0.4
        )

    # Colorbar bezpośrednio przypięty do osi, aby uniknąć ostrzeżeń Matplotlib
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    ax = plt.gca()
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Natężenie prądu [A]', fontsize=12, fontweight='bold')

    plt.title(f"Rozpływ prądów w obwodzie (Węzły: {len(DiG)})", fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
