import numpy as np
import matplotlib as mplot

def power_iteration(vector,matrix,max_iter,epsilon=1e-10):
    diff = epsilon
    iter=1
    while diff >= epsilon and iter < max_iter:
        next_vector = np.dot(matrix,vector)
        next_vector = next_vector / np.linalg.norm(next_vector, np.inf)
        diff = np.linalg.norm(next_vector - vector,np.inf)
        vector = next_vector
        iter+=1
    return vector

def test_power():
    # Matrixes = [np.random.rand(100,100) for _ in range(5)]
    Matrixes = []
    for _ in range (5):
        A=np.random.rand(10,10)
        A=(A+A.T)/2
        Matrixes.append(A)

    for matrix in Matrixes:
        vector = np.random.rand(A.shape[0])
        vector = vector / np.linalg.norm(vector,np.inf)
        power_val = power_iteration(vector,matrix,1000000)
        eigen_val = np.linalg.eigvals(matrix)
        index = np.argmax(np.abs(eigen_val))
        diff = np.abs(power_val[index] - eigen_val[index])
        prc = power_val[index] / eigen_val[index] * 100
        print(eigen_val)
        print(power_val)
        print("For matrix difference between dominant value via power method and eigenvalue method is ", diff, " and scale of error is " , prc,"%")

def power_iteration_impr(vector,matrix,max_iter,epsilon=1e-8):
    diff = epsilon+1
    iter=1
    next_vector = vector
    while diff > epsilon and iter < max_iter:
        vector = next_vector
        next_vector = np.dot(matrix,vector)
        next_vector = next_vector / np.linalg.norm(next_vector, np.inf)
        diff = np.linalg.norm(next_vector - vector,np.inf)
        iter+=1
    index = np.argmax(np.abs(next_vector))
    return next_vector , next_vector[index]/vector[index]

def test_power_second():
    # Matrixes = [np.random.rand(100,100) for _ in range(5)]
    Matrixes = []
    for _ in range(5):
        A = np.random.rand(100, 100)
        A = (A + A.T) / 2
        Matrixes.append(A)
    for matrix in Matrixes:
        vector = np.random.rand(100)
        vector = vector / np.linalg.norm(vector,np.inf)
        power_vector,power_val = power_iteration_impr(vector,matrix,1000000)
        eig_val , eig_vector = np.linalg.eig(matrix)
        val = eig_val[np.argmax(np.abs(eig_val))]
        diff = np.abs(power_val - val)
        prc = power_val/val * 100
        # prc = val/power_val * 100
        # eigen_val = np.linalg.eigvals(matrix)
        # index = np.argmax(eigen_val)
        # diff = np.abs(power_val[index] - eigen_val[index])
        # prc = power_val[index] / eigen_val[index] * 100
        print("For matrix difference between dominant value via power method and eigenvalue method is ", diff, " and scale of error is " , prc)


test_power()
