import math

import numpy as np
import matplotlib.pyplot as mp

s = np.linspace(0, 6.2833 ,100)
t = np.linspace(0, 3.1416 ,100)

S,T = np.meshgrid(s,t)

x = np.cos(S) * np.sin(T)
y = np.sin(S) * np.sin(T)
z = np.cos(T)

V = np.array([x,y,z])

A= [np.random.rand(3,3) for _ in range (3)]

V = A[0] @ A[1] @ A[2] @ V

fig = mp.figure()
ax = fig.add_subplot(projection = '3d')
ax.plot_surface(V[0],V[1],V[2])
ax.set_aspect('equal')

mp.show()

