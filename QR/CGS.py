import numpy as np
import matplotlib.pyplot as plt

def make_Q_better(A):
    m,n = np.shape(A)
    Q = np.zeros((m,n))
    R = np.zeros((n,n))
    R[0,0] = np.linalg.norm(A[:,0])
    Q[:,0] = A[:,0]/ R[0,0]

    for i in range (1,n):
        v = A[:,i].copy()
        for j in range (0,i):
            R[j,i] = np.dot(Q[:,j],v)
            v -= R[j,i] * Q[:,j]
        R[i,i] = np.linalg.norm(v)
        v /= R[i,i]
        Q[:,i] = v

    return Q,R
def make_Q(A):
    m,n = np.shape(A)
    Q = np.zeros((m,n))
    R = np.zeros((n,n))
    R[0,0] = np.linalg.norm(A[:,0])
    Q[:,0] = A[:,0]/ R[0,0]

    for i in range (1,n):
        v = A[:,i].copy()
        for j in range (0,i):
            R[j,i] = np.dot(Q[:,j],A[:,i])
            v -= R[j,i] * Q[:,j]
        R[i,i] = np.linalg.norm(v)
        v /= R[i,i]
        Q[:,i] = v

    return Q,R

def test_loss(Q):
    n = Q.shape[1]
    I = np.eye(n)
    return np.linalg.norm(I - np.dot(Q.T,Q))

def wykresy():
    p_values = np.linspace(0, 16, 33)
    actual_conds = []
    losses_cgs = []
    losses_mgs = []
    losses_numpy = []
    for p in p_values:
        # Generowanie losowych macierzy ortogonalnych U i V (8x8) via QR
        U, _ = np.linalg.qr(np.random.randn(8, 8))
        V, _ = np.linalg.qr(np.random.randn(8, 8))

        # Generowanie wartości osobliwych malejących geometrycznie
        S = np.logspace(0, -p, 8)

        # Konstrukcja macierzy A = U * Sigma * V^T
        A = np.dot(U * S, V.T)  # sprytne mnożenie macierzy U przez diagonalną opartą na S

        # Weryfikacja i zapisanie faktycznego uwarunkowania macierzy
        actual_conds.append(np.linalg.cond(A))

        # --- Obliczenia dla CGS ---
        Q_cgs, _ = make_Q(A)
        losses_cgs.append(test_loss(Q_cgs))

        # --- Obliczenia dla MGS ---
        Q_mgs, _ = make_Q_better(A)  # upewnij się, że tak nazwałeś funkcję dla MGS
        losses_mgs.append(test_loss(Q_mgs))

        # --- Obliczenia dla funkcji bibliotecznej (NumPy wykorzystuje algorytm Householdera) ---
        Q_np, _ = np.linalg.qr(A)
        losses_numpy.append(test_loss(Q_np))

    # 3. Tworzenie wykresu w skali log-log
    plt.figure(figsize=(10, 6))

    plt.loglog(actual_conds, losses_cgs, 'o-', label='Classical Gram-Schmidt (CGS)', color='red')
    plt.loglog(actual_conds, losses_mgs, 's-', label='Modified Gram-Schmidt (MGS)', color='blue')
    plt.loglog(actual_conds, losses_numpy, '^--', label='NumPy QR (Householder)', color='green')

    # Estetyka wykresu
    plt.title('Utrata ortogonalności w zależności od uwarunkowania macierzy A')
    plt.xlabel('Wskaźnik uwarunkowania cond(A) [Skala logarytmiczna]')
    plt.ylabel('Utrata ortogonalności ||I - Q^T Q|| [Skala logarytmiczna]')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()

    # Wyświetlenie wykresu
    plt.show()

wykresy()

# def test_make_Q():
#     # Test 1: Macierz kwadratowa (3x3)
#     A1 = np.array([
#         [12.0, -51.0, 4.0],
#         [6.0, 167.0, -68.0],
#         [-4.0, 24.0, -41.0]
#     ])
#
#     # Test 2: Macierz prostokątna (4x3)
#     A2 = np.array([
#         [1.0, 1.0, 0.0],
#         [1.0, 0.0, 1.0],
#         [0.0, 1.0, 1.0],
#         [0.0, 0.0, 1.0]
#     ])
#
#     matrices_to_test = [A1, A2]
#
#     for idx, A in enumerate(matrices_to_test):
#         Q, R = make_Q(A)
#         _, n = A.shape
#
#         # 1. Sprawdzenie, czy A = Q * R
#         assert np.allclose(A, np.dot(Q, R)), f"Test {idx + 1} nieudany: A nie jest równe Q * R"
#
#         # 2. Sprawdzenie, czy Q jest ortonormalna (Q^T * Q = Macierz Jednostkowa)
#         I = np.eye(n)
#         assert np.allclose(np.dot(Q.T, Q), I), f"Test {idx + 1} nieudany: Q nie jest ortonormalna"
#
#         # 3. Sprawdzenie, czy R jest trójkątna górna
#         assert np.allclose(R, np.triu(R)), f"Test {idx + 1} nieudany: R nie jest trójkątna górna"
#
#     print("Wszystkie testy zakończone sukcesem! Algorytm CGS działa poprawnie.")
#
#
# # Uruchomienie testów
# test_make_Q()