import numpy as np
import time

def lu_factorization(A):
    n=len(A)

    for j in range(n):
        for i in range(j + 1, n):
            mult = A[i][j] / A[j][j]
            for k in range(j, n):
                A[i][k] -= A[j][k] * mult
            A[i][j] = mult
    return A

def check_LU(n):
    A = np.random.rand(n,n)
    B= A.copy()

    start_time=time.time()
    lu_factorization(B)
    end_time=time.time()
    U = [[B[i][j] if i <= j else 0 for j in range(n)] for i in range(n)]
    L = [[B[i][j] if i > j else  0 for j in range(n)] for i in range(n)]
    for i in range(n):
        L[i][i]=1
    CheckMatrix = np.dot(L,U)

    error = np.linalg.norm(A - CheckMatrix)
    print(error)
    return error , end_time - start_time

def main():
    size = 100
    error, time = check_LU(size)

    print(error,time)
