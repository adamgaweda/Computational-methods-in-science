import numpy as np
import matplotlib as plt
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