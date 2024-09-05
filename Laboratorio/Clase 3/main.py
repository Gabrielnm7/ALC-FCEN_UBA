import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def proyectarPts(T, wz):
    # - chequeo de matriz 2x2 y multiplicacion matricial valida
    if T.shape == (2, 2) and (T.shape[1] == wz.shape[0]):
        xy = None
        # Calculamos el prod matricial T * wz
        # xy = np.dot(T, wz) (otra forma de hacerlo)
        xy = T @ wz
    if T.shape == (3, 3):
        wz_homogeneous = np.vstack([wz, np.ones((1, wz.shape[1]))])
        xy_homogeneous = T @ wz_homogeneous
        xy = xy_homogeneous[:2, :] / xy_homogeneous[2, :]
    else:
        raise ValueError("La matriz de transformación debe ser de 2x2 o 3x3.")
    return xy


def pointsGrid(corners):
    # crear 10 lineas horizontales
    [w1, z1] = np.meshgrid(np.linspace(corners[0, 0], corners[1, 0], 46),
                           np.linspace(corners[0, 1], corners[1, 1], 10))

    [w2, z2] = np.meshgrid(np.linspace(corners[0, 0], corners[1, 0], 10),
                           np.linspace(corners[0, 1], corners[1, 1], 46))

    w = np.concatenate((w1.reshape(1, -1), w2.reshape(1, -1)), 1)
    z = np.concatenate((z1.reshape(1, -1), z2.reshape(1, -1)), 1)
    wz = np.concatenate((w, z))

    return wz


def grid_plot(ax, ab, limits, a_label, b_label):
    ax.plot(ab[0, :], ab[1, :], '.')
    ax.set(aspect='equal',
           xlim=limits[0], ylim=limits[1],
           xlabel=a_label, ylabel=b_label)


def vistform(T, wz, titulo=''):
    # transformar los puntos de entrada usando T
    xy = proyectarPts(T, wz)
    if xy is None:
        print('No fue implementada correctamente la proyeccion de coordenadas')
        return None

    # calcular los limites para ambos plots
    minlim = np.min(np.concatenate((wz, xy), 1), axis=1)
    maxlim = np.max(np.concatenate((wz, xy), 1), axis=1)

    bump = [np.max(((maxlim[0] - minlim[0]) * 0.05, 0.1)),
            np.max(((maxlim[1] - minlim[1]) * 0.05, 0.1))]
    limits = [[minlim[0]-bump[0], maxlim[0]+bump[0]],
              [minlim[1]-bump[1], maxlim[1]+bump[1]]]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(titulo)
    grid_plot(ax1, wz, limits, 'w', 'z')
    grid_plot(ax2, xy, limits, 'x', 'y')


def main():
    print('Ejecutar el programa')
    # - generar el tipo de transformacion dando valores a la matriz T
    # T = pd.read_csv('T.csv', header=None).values

    corners = np.array([[0, 0], [100, 100]])
    wz = pointsGrid(corners)
    
    theta = np.deg2rad(45)  # Cambiar a 45 grados o cualquier otro valor
    # Ejercicio 2 - labo03
    # - Definir el ángulo de rotación en radianes
    # Crear la matriz de rotación R_theta
    T = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

    # - Ejercicio 3 - labo03
    # Escalado
    T_scale = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])
    vistform(T_scale, wz, 'Escalado')

    # Traslación
    T_translate = np.array([[1, 0, 20], [0, 1, 30], [0, 0, 1]])
    vistform(T_translate, wz, 'Traslación')

    # Rotación
    theta = np.deg2rad(45)
    T_rotate = np.array([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta), np.cos(theta), 0],
                         [0, 0, 1]])
    vistform(T_rotate, wz, 'Rotación')

    # Deformación (Shear)
    T_shear = np.array([[1, 0.5, 0], [0.5, 1, 0], [0, 0, 1]])
    vistform(T_shear, wz, 'Deformación')


if __name__ == "__main__":
    main()
