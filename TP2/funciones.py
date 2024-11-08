import numpy as np


def descomposicion_LU_sin_pivoteo(A):
    """
    Realiza la descomposición LU de una matriz cuadrada A asumiendo que no es necesario realizar pivoteos
    Args:
        A (array): matriz cuadrada
    Return:
        L (array): matriz triangular inferior
        U (array): matriz triangular superior
    """
    A = np.array(A)
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy()

    for i in range(n-1):
        for j in range(i+1, n):
            # Calculamos los factores de la matriz de eliminación de Gauss
            factor = U[j, i] / U[i, i]
            L[j, i] = factor  # Guarda el factor en la columna i de L
            # Actualiza la fila j de U
            U[j, i:] = U[j, i:] - factor * U[i, i:]

    return L, U


def permutar_filas(A, i, j):
    """
    Intercambia las filas i y j de la matriz A.
    Args:
        A: matriz cuadrada de tamaño nxn
        i, j: índices de las filas a intercambiar
    Returns:
        A_permutada: matriz A con las filas i y j permutadas
        P: matriz de permutación de filas
    """
    # Convertirmos A en un array de numpy para garantizar que es manipulable
    A = np.array(A)
    n = A.shape[0]

    # Creamos la matriz de permutación
    P = np.eye(n)
    P[[i, j]] = P[[j, i]]

    # Permutamos las filas de A
    A_permutada = np.dot(P, A)
    return A_permutada, P


def triangular_columna(A, L, j):
    """
    Esta funcion es una funcion auxiliar para usar en calcularLU. Triangula la matriz A por la columna j
    Args:
        A (array): matriz cuadrada
        j (int): indice de la columna a triangular
    Return:
        A (array): matriz triangularizada
        L (array): matriz de factores de la matriz de eliminación de Gauss
    """
    n = A.shape[0]
    for i in range(j+1, n):
        # Calculamos los factores de la matriz de eliminación de Gauss
        factor = A[i, j] / A[j, j]
        L[i, j] = factor
        # Actualizamos A
        A[i, j:] = A[i, j:] - factor * A[j, j:]

    return A, L


def calcularLU(A):
    """
    Realiza la descomposición LU de una matriz cuadrada A.
    Args:
        A (array): matriz cuadrada
    Return:
        P (array): matriz de permutación. Es la identidad si no se realizaron permutaciones
        L (array): matriz triangular inferior
        U (array): matriz triangular superior
    """
    n = A.shape[0]
    L = np.eye(n)  # Inicializamos L como una matriz identidad
    U = A.copy()  # U comienza como una copia de A
    P = np.eye(n)  # Inicializamos P como una matriz identidad

    for i in range(n-1):
        # Me fijo si A[i,i] es distinto de 0
        if U[i, i] != 0:
            # Triangulamos
            U, L = triangular_columna(U, L, i)

        else:  # Hay que permutar filas
            # Buscamos el indice de la fila con el mayor valor en la columna j (en modulo)
            max_index = i + np.argmax(np.abs(U[i:, i]))
            # Intercambiamos filas
            U, P_aux = permutar_filas(U, i, max_index)
            # Actualizamos P
            P = np.dot(P_aux, P)
            # Triangulamos
            U, L = triangular_columna(U, L, i)

    # Si P es la identidad
    if np.allclose(P, np.eye(n)):
        return P, L, U
    else:
        L, U = descomposicion_LU_sin_pivoteo(np.dot(P, A))
        return P, L, U


def Ly(L, b):
    """
    Resuelve el sistema Ly = b donde L es una matriz triangular inferior.

    Args:
        L (array): Matriz triangular inferior.
        b (array): Vector de términos independientes.

    Returns:
        y (array): Solución del sistema Ly = b.
    """
    L = np.array(L)
    b = np.array(b)
    n = len(b)
    y = np.zeros(n)

    y[0] = b[0]
    for i in range(1, n):
        # Similar a como lo hice goatcheverry
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    return y


def Ux(U, y):
    """
    Calcula el producto matricial de una matriz triangular superior por un vector
    Args:
        U (array): matriz triangular superior
        y (array): vector
    Return:
        x (array)
    """
    U = np.array(U)
    n = U.shape[0]
    x = np.zeros(n)
    y = np.array(y)

    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    return x


def inversaLU(L, U, P=None):
    """
    Calcula la inversa de una matriz A a partir de su descomposición LU.
    En caso de que se haya realizado permutación de filas, se debe ingresar la matriz de permutación P.

    Args:
        L (array): matriz triangular inferior
        U (array): matriz triangular superior
        P (array): matriz de permutación de filas (default=None)

    Returns:
        A_inv (array): inversa de la matriz A

    """
    n = L.shape[0]
    # Creo la identidad de tamaño nxn y la matriz Y llena de 0s
    I = np.eye(n)
    Y = np.zeros((n, n))

    # Primero resolvemos LY = I para cada columna de I
    for i in range(n):
        Y[i] = Ly(L, I[i])

    Y = Y.T
    # Luego resolvemos UX = Y para cada columna de Y y obtenemos la inversa
    A_inv = np.zeros((n, n))
    for i in range(n):
        A_inv[i] = Ux(U, Y[:, i])

    A_inv = A_inv.T
    if P is not None:
        A_inv = np.dot(A_inv, P)

    return A_inv

## Hasta aca son las mismas funciones que en el TP1

def estado(A, v, k):
    """ 
    Calcula el estado de un vector v luego de k iteraciones de la matriz A (ver Pag. 106 del Apunte General)
    """
    for _ in range(k):
        Av = A @ v
        v = Av / np.linalg.norm(Av,2)
    return v
    
def metodoPotencia(A,k):
    """ 
    Esta funcion encuentra el mayor autovalor de una matriz A (ver Pag. 104 del Apunte General)
    """
    # Calculamos cuantas filas tiene A:
    n = np.shape(A)[0]

    # Creamos un vector aleatorio x_0 teniendo en cuenta el tamaño de A:
    x_0 =  np.random.rand(n)

    # Aplicamos la función estado para obtener v:
    v = estado(A, x_0, k)

    # Por definición del Método de la Potencia es el cociente de Rayleigh:
    maxAutovalor = (np.transpose(v) @ A @ v) / (np.transpose(v) @ v)

    return maxAutovalor

def normalizar_matriz(A):
    """
    Esta funcion normaliza la matriz de coeficientes $A_{rr}$
    """
    n = A.shape[0]
    e = np.ones((n, 1))
    E = np.eye(n) - (1/n) * e @ e.T
    A_normalizada = E @ A
    return A_normalizada

def matriz_de_covarianzas(A_normalizada):
    """
    Esta funcion calcula una matriz de covarianzas a partir de una matriz normalizada
    """
    n = A_normalizada.shape[0]
    C = (A_normalizada.T @ A_normalizada) / (n - 1)
    return C

def deflacion_de_Hotelling(A, k, epsilon):
    """
    Implementa el proceso de Deflación de Hotelling para encontrar los dos primeros autovalores y autovectores de A
    """
    # Implementar
    # Inicializo un vector aliatorio X_0 de norma 1
    n = A.shape[0]
    x_0 = np.random.rand(n)
    x_0 = x_0 / np.linalg.norm(x_0, 2)
    
    return None