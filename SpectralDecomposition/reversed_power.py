import numpy as np
import scipy as sc
import scipy.linalg


def reversed_power(Matrix, sigma, max_iter=100000, epsilon=1e-10):
    n = Matrix.shape[0]
    Shadow = Matrix - sigma * np.eye(n)
    lu, piv = scipy.linalg.lu_factor(Shadow)

    vector = np.random.rand(n)
    vector = vector / np.linalg.norm(vector,np.inf)
    diff =epsilon + 1

    new_vector = vector
    iter = 0
    while iter < max_iter and diff > epsilon:
        vector=new_vector
        new_vector = scipy.linalg.lu_solve((lu,piv),vector)
        new_vector = new_vector / np.linalg.norm(new_vector,np.inf)

        diff = np.linalg.norm(new_vector-vector,np.inf)
        iter+=1

    index = np.argmax(np.abs(new_vector))
    mi = new_vector[index] / vector[index]

    return sigma + 1 / mi , new_vector

def test_reverse():
    A = np.random.rand(100,100)
    A = (A + A.T)/2

    eigvals, eigvecs = np.linalg.eig(A)
    sigma = eigvals[np.random.randint(len(eigvals))]
    value = eigvals[np.argmin(np.abs(eigvals - sigma))]

    lambd, vector = reversed_power(A,sigma)


    proportion = np.abs(lambd - value) / np.abs(value)

    print("Difference of using reverse power method to eigenvalue method is of scale: ", proportion)

test_reverse()