import numpy as np
import time

def gauss_jordan_2arg(A, b):
    return gauss_jordan_1arg(np.hstack((A, b.reshape(-1,1))))


def gauss_jordan_1arg(B):
    n = len(B)
    B= B.copy().astype(float)

    for i in range(n):
        max_row = i
        for j in range(i, n):
            if abs(B[j][i]) >= abs(B[max_row][i]):
                max_row = j
        B[[i, max_row]] = B[[max_row, i]]

        for j in range(n):
            if (i == j):
                continue
            if B[i][i]==0:
                print("wywaliło się, słabe dane")
                return B
            a = B[j][i] / B[i][i]
            for k in range(i, n + 1):
                B[j][k] -= (B[i][k] * a)
            B[j][i] = 0

        a = B[i][i]
        for j in range(i, n + 1):
            B[i][j] /= a

    return B[:,-1]

def main():
    n = 600
    A = np.random.rand(n, n)
    b = np.random.rand(n)

    # Gauss-Jordan
    start = time.time()
    x1 = gauss_jordan_2arg(A.copy(), b.copy())
    t1 = time.time() - start

    # numpy
    start = time.time()
    x2 = np.linalg.solve(A, b)
    t2 = time.time() - start

    print("Gauss-Jordan:", t1)
    print("numpy.linalg.solve:", t2)
    print("błąd:", np.linalg.norm(x1 - x2))

if __name__ == "__main__":
    main()