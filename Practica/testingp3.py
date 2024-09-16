import numpy as np
# Testeo con la matriz del pdf
A= np.array([[1,-1, 0,1],
             [0, 1,4, 0],
             [2,-1,0,-2],
             [-3,3,0,-1]])

n = A.shape[0]
# Sé que la primer fila de L solo tiene 1 en la diagonal, i.e. L[0,0] = 1

L = np.eye(n)
for i in range(n-1):
    L_i = np.eye(n)
    for j in range(i+1,n):
        L[j,i] = -(A[j,i]/A[i,i])
        L_i[j,i] = -(A[j,i]/A[i,i])
    print("L = \n",L)
    A = np.dot(L_i,A)
    print("A = \n",A)