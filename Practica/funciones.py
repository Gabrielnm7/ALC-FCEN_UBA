import numpy as np

def row_echelon(A):
    """Return Row Echelon Form of matrix A."""
    A = A.copy()  # Trabajar sobre una copia de A
    r, c = A.shape
    
    if r == 0 or c == 0:
        return A

    for i in range(r):
        if A[i, 0] != 0:
            break
    else:
        B = row_echelon(A[:, 1:])
        return np.hstack([A[:, :1], B])

    if i > 0:
        A[[0, i]] = A[[i, 0]]  # Intercambiar filas

    A[0] = A[0] / A[0, 0]  # Normalizar la primera fila
    A[1:] -= A[0] * A[1:, 0:1]  # Eliminar la primera columna en las filas restantes

    B = row_echelon(A[1:, 1:])
    return np.vstack([A[:1], np.hstack([A[1:, :1], B])])

def row_echelon_augmented(A):
    """Return Row Echelon Form of an augmented matrix A and solve the system."""
    A = A.astype(float)  # Asegurarse de que trabajamos con floats
    r, c = A.shape
    
    for i in range(min(r, c-1)):
        # Seleccionar el pivote
        max_row = np.argmax(np.abs(A[i:, i])) + i
        A[[i, max_row]] = A[[max_row, i]]  # Intercambiar filas si es necesario

        # Si el pivote es 0, continuar con la siguiente columna
        if A[i, i] == 0:
            continue

        # Normalizar la fila con el pivote
        A[i] = A[i] / A[i, i]

        # Eliminar los elementos debajo del pivote
        for j in range(i+1, r):
            A[j] = A[j] - A[j, i] * A[i]
    
    # Volver hacia arriba para eliminar elementos sobre el pivote
    for i in range(min(r, c-1)-1, -1, -1):
        for j in range(i-1, -1, -1):
            A[j] = A[j] - A[j, i] * A[i]
    
    return A

def bashkara(a,b,c):
    """
    Genera las soluciones de la ecuación cuadrática ax^2+bx+c=0
    Args:
    a: coeficiente cuadrático
    b: coeficiente lineal
    c: término independiente
    Returns:
        x1: solución 1
        x2: solución 2
    """
    if a==0:
        return Exception("a no puede ser 0")
    
    return (-b+np.sqrt(b**2-4*a*c))/(2*a),(-b-np.sqrt(b**2-4*a*c))/(2*a)

def solve_linear_system(A_augmented):
    """Resuelve un sistema de ecuaciones lineales dado una matriz.
    Args:
        A_augmented: matriz aumentada del sistema de ecuaciones
    Returns:
        solution: solución del sistema de ecuaciones o None si no se puede resolver
    
    solution tiene la forma de un array de numpy con las soluciones de las incógnitas
    
    """
    # Separar la matriz de coeficientes (A) y el vector de términos independientes (b)
    A = A_augmented[:, :-1]  # Todas las columnas menos la última
    b = A_augmented[:, -1]   # Última columna
    
    # Resolver el sistema de ecuaciones
    try:
        solution = np.linalg.solve(A, b)
    except np.linalg.LinAlgError as e:
        print(f"Error: {e}")
        return None
    
    return solution