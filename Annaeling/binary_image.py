import math
import random

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def generate_map(n):
    return np.random.choice([0, 1], size=(n, n), p=[0.5, 0.5])

#TEMPERATURE FUNCTIONS
#TEMPERATURE FUNCTIONS
def temp_log(temperature, parameter, iteration):
    # np.e gwarantuje płynny start od temperatury początkowej w iteracji 0
    return temperature / np.log(np.e + parameter * iteration)

def temp_exp(temperature, parameter, iteration):
    return temperature * (parameter ** iteration)

def temp_lin(temperature, parameter, iteration):
    # Zabezpieczenie przed temperaturą ujemną
    return max(0.0001, temperature - (parameter * iteration))

#SWAP
def swap_random(World):
    n = len(World)
    x1,y1 = np.random.randint(0,n),np.random.randint(0,n)
    x2,y2 = np.random.randint(0,n),np.random.randint(0,n)
    while x1 == x2 and y1==y2:
        x2,y2 = np.random.randint(0,n),np.random.randint(0,n)
    World[x1][y1],World[x2][y2] = World[x2][y2],World[x1][y1]
    return (x1,y1),(x2,y2)
def swap_neighbour(World):
    n = len(World)
    x1,y1 = np.random.randint(0,n),np.random.randint(0,n)
    x2,y2 = (x1+random.randint(0,1))%n , (y1+random.randint(0,1))%n
    World[x1][y1],World[x2][y2]= World[x2][y2],World[x1][y1]
    return (x1, y1), (x2, y2)

def energy_8th(World,a,b):
    n = len(World)
    energy = 0
    for i in range (a-1,a+2):
        i = i%n
        for j in range(b-1,b+2):
            j=j%n
            if i == a and j == b:
                continue
            if World[i][j] != World[a][b]:
                energy+=1
    return energy
def energy_26th(World,a,b):
    n = len(World)
    energy = 0
    for i in range (a-2,a+3):
        i = i%n
        for j in range(b-2,b+3):
            j=j%n
            if i == a and j == b:
                continue
            if World[i][j] != World[a][b]:
                energy+=1
    return energy
def energy_4th(World,a,b):
    n = len(World)
    energy = 0
    points = [(a+1,b),(a-1,b),(a,b-1),(a,b+1)]
    for i,j in points:
        i=i%n
        j=j%n
        if World[a][b] != World[i][j]:
            energy+=1
    return energy

def energy(World,energy_func):
    World = World.copy()
    n = len(World)
    energy = 0
    for i in range(n):
        for j in range(n):
            energy+=energy_func(World,i,j)
    return energy


def solve(world, temp_func, ener_func, swap_func, max_iter, cooling_param):
    n=len(world)

    temperatures = []
    energies = []

    old_energy = energy(world,ener_func)
    best_energy = old_energy
    best_world = world.copy()

    for iteration in range(max_iter):
        temp = temp_func(1.0, cooling_param,iteration)

        temperatures.append(temp)
        energies.append(old_energy)

        energy_diff = 0
        new_world = world.copy()
        for _ in range (int(n**(1/2))):
            a,b = swap_func(new_world)
            energy_diff_old = ener_func(world,a[0],a[1])+ener_func(world,b[0],b[1])
            energy_diff_new = ener_func(new_world,a[0],a[1])+ener_func(new_world,b[0],b[1])
            energy_diff += energy_diff_old-energy_diff_new

        if energy_diff >= 0 or random.random() < np.exp((energy_diff)/max(temp,1e-10)):
            world = new_world
            old_energy -= energy_diff
            if old_energy<best_energy:
                best_world = new_world.copy()
                best_energy = old_energy

    world = best_world
    temperatures.append(temperatures[-1])
    energies.append(best_energy)

    return world, temperatures, energies


def plot_full_report(W_init, W_final, temps, energies, title):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(title, fontsize=16)

    # 1. Stan początkowy (macierz jako obraz)
    axes[0, 0].set_title("Stan początkowy")
    axes[0, 0].imshow(W_init, cmap='binary')
    axes[0, 0].axis('off')  # Wyłącza osie z numerami pikseli

    # 2. Stan końcowy (macierz jako obraz)
    axes[0, 1].set_title("Stan końcowy")
    axes[0, 1].imshow(W_final, cmap='binary')
    axes[0, 1].axis('off')

    # 3. Wykres energii
    axes[1, 0].set_title("Spadek energii")
    axes[1, 0].plot(energies, color='green')
    axes[1, 0].set_xlabel("Iteracja")

    # 4. Wykres temperatury
    axes[1, 1].set_title("Spadek temperatury")
    axes[1, 1].plot(temps, color='orange')
    axes[1, 1].set_xlabel("Iteracja")

    plt.tight_layout()
    plt.show()


# === PRZYKŁAD URUCHOMIENIA ===
if __name__ == "__main__":
    rozmiar_mapy = 50
    World_init = generate_map(rozmiar_mapy)

    W_final, temps, energies = solve(
        world=World_init,
        temp_func=temp_exp,
        ener_func=energy_8th,
        swap_func=swap_random,
        max_iter=5000,
        cooling_param=0.999
    )

    plot_full_report(World_init, W_final, temps, energies, "Symulowane Wyżarzanie - Grupowanie 2D")

import itertools

# 1. Konfiguracja kombinacji
map_sizes = [50, 100, 200]

swap_funcs = {
    "Losowy (Swap Random)": swap_random,
    "Sąsiedzki (Swap Neighbour)": swap_neighbour
}

# Parujemy funkcję z dopasowanym do niej parametrem chłodzenia
temp_configs = {
    "Logarytmiczna": (temp_log, 1.0),
    "Wykładnicza": (temp_exp, 0.9995),
    "Liniowa": (temp_lin, 0.0001)
}

# Wybieramy funkcję energii (np. 8 sąsiadów) oraz liczbę iteracji
chosen_energy_func = energy_8th
max_iter = 50000

# 2. Główna pętla uruchamiająca wszystkie kombinacje
for size, (swap_name, swap_f), (temp_name, (temp_f, param)) in itertools.product(
    map_sizes,
    swap_funcs.items(),
    temp_configs.items()
):
    print(f"Uruchamianie: Mapa {size}x{size} | Swap: {swap_name} | Temp: {temp_name}")

    # Generowanie nowej mapy dla danej wielkości
    world_init = generate_map(size)

    # Uruchomienie algorytmu
    world_final, temps, energies = solve(
        world=world_init.copy(),
        temp_func=temp_f,
        ener_func=chosen_energy_func,
        swap_func=swap_f,
        max_iter=max_iter,
        cooling_param=param
    )

    # Rysowanie raportu
    title = f"Mapa: {size}x{size} | Swap: {swap_name} | Temp: {temp_name}"
    plot_full_report(world_init, world_final, temps, energies, title)

