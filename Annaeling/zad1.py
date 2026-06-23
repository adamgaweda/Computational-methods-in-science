import random

import numpy as np
import networkx as nx
import matplotlib.pyplot as mp

def generate_vertex(n):
    G=[(random.randint(0,n+1),random.randint(0,n+1)) for _ in range (n) for _ in range (n)]
    return G

def distance(a,b):
    return 

def energy(G):
    sum=0
    for i in range(len(G)-1):
        sum+=distance(G[i])
