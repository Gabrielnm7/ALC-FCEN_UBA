"""
Práctica 1 - Ejercicio 21
"""

import numpy as np

def traza(A):
    resp = 0
    for i in range(len(A)):
        resp += A[i][i]
    ## 
    return resp

def sumamodulo(A):
    resp = 0
    for i in range(len(A)):
        for j in range(len(A[i])):
            resp += abs(A[i][j])
    ## 
    return resp


def positivosmayoresanegativos(A):
    resp = True
    positivos = 0
    negativos = 0
    for i in range(len(A)):
        for j in range(len(A[i])):
            if A[i][j] > 0:
                positivos += 1
            elif A[i][j] < 0:
                negativos += 1
    if positivos <= negativos:
        resp = False    
    ## 
    return resp


def main():

    #Definición de matriz
    A = np.array([[ 1, -2, -1, -2],
                  [-4, -4,  5, -9],
                  [-9,  0,  8, -1],
                  [ 8,  3,  3, -3]])

    print('Matriz A\n', A)
    
    print('Traza de A:', traza(A))
    assert traza(A)==2
    
    print('Suma en módulo de elementos de A:', sumamodulo(A))
    assert sumamodulo(A)==63
    
    print('Positivos mayores a negativos?:', positivosmayoresanegativos(A))
    assert positivosmayoresanegativos(A)==False


if __name__ == "__main__":
    main()
