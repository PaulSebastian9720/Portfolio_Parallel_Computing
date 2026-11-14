"""Pruebas de corrección (equivale a pruebas.cpp de la práctica 01)."""

import numpy as np

from comun import comparar, crear_mascara
from filtros_cpu import convolucion_python

# Matriz de la Figura 1 de la guía (5x5), fila tras fila.
MATRIZ_FIGURA = np.array([
    35, 40, 41, 45, 50,
    40, 40, 42, 46, 52,
    42, 46, 50, 55, 55,
    48, 52, 56, 58, 60,
    56, 60, 65, 70, 75,
], dtype=np.float32)

# Máscara de la Figura 1 de la guía.
MASCARA_FIGURA = np.array([-2, -1, 0, -1, 1, 1, 0, 1, 2], dtype=np.float32)


# Devuelve los casos calculados a mano sobre la matriz de la Figura 1. En los bordes,
# los vecinos fuera de la imagen valen 0.
# Ejemplo, nitidez en (0,0): 5*35 - 40 (derecha) - 40 (abajo) = 95.
# Retorna: lista de tuplas (nombre, mascara, fila, col, esperado).
def casos_manuales():
    nitidez = crear_mascara("nitidez", 3)
    laplaciano = crear_mascara("laplaciano", 3)
    media = crear_mascara("media", 3)
    return [
        ("figura", MASCARA_FIGURA, 1, 1, 78.0),
        ("figura", MASCARA_FIGURA, 1, 2, 87.0),
        ("nitidez", nitidez, 0, 0, 95.0),
        ("nitidez", nitidez, 0, 4, 153.0),
        ("nitidez", nitidez, 1, 1, 32.0),
        ("laplaciano", laplaciano, 0, 0, -60.0),
        ("laplaciano", laplaciano, 1, 1, 8.0),
        ("laplaciano", laplaciano, 4, 4, -170.0),
        ("media3", media, 0, 0, 155.0 / 9),
        ("media3", media, 1, 1, 376.0 / 9),
    ]


# Ejecuta los casos manuales con una implementación e imprime una fila por caso.
# Parámetros:
#   gpu  objeto GpuConvolucion para usar PyCUDA, o None para usar Python secuencial
def ejecutar_casos_manuales(gpu):
    for nombre, mascara, fila, col, esperado in casos_manuales():
        if gpu is not None:
            salida, _ = gpu.convolucion(MATRIZ_FIGURA, mascara, 5, 5, 3, 128)
        else:
            salida = convolucion_python(MATRIZ_FIGURA, 5, 5, mascara, 3)
        obtenido = float(salida[fila * 5 + col])
        correcto = abs(obtenido - esperado) < 1e-4
        print(f"{'pycuda' if gpu else 'python'}\t{nombre}\t{fila}\t{col}\t"
              f"{esperado:.4f}\t{obtenido:.4f}\t{'OK' if correcto else 'FALLO'}")


# Prueba pequeña: comprueba a mano Python secuencial y PyCUDA.
# Parámetros:
#   gpu  objeto GpuConvolucion ya inicializado
def prueba_manual(gpu):
    print("=== PRUEBA MANUAL ===")
    print("impl\tprueba\tfila\tcol\tesperado\tobtenido\testado")
    ejecutar_casos_manuales(None)
    ejecutar_casos_manuales(gpu)


# Prueba de límites: imagen de 1003x997, cuyos píxeles no son múltiplo de los hilos por
# bloque. Compara PyCUDA contra Python secuencial con 128, 256 y 512 hilos.
# Parámetros:
#   gpu  objeto GpuConvolucion ya inicializado
def prueba_no_multiplo(gpu):
    print("=== TAMANO NO MULTIPLO ===")
    ancho, alto = 1003, 997
    imagen = np.array([(i * 37) % 256 for i in range(ancho * alto)], dtype=np.float32)
    mascara = crear_mascara("media", 3)

    print("(calculando la referencia en Python secuencial...)", flush=True)
    referencia = convolucion_python(imagen, ancho, alto, mascara, 3)

    print("imagen\thilos\tbloques\terror_maximo\terror_medio\tpixeles_fuera\testado")
    for hilos in (128, 256, 512):
        salida, tiempos = gpu.convolucion(imagen, mascara, ancho, alto, 3, hilos)
        error = comparar(referencia, salida)
        estado = "OK" if error.fuera_de_tolerancia == 0 else "FALLO"
        print(f"{ancho}x{alto}\t{hilos}\t{tiempos.bloques}\t{error.maximo:.6f}\t"
              f"{error.medio:.6f}\t{error.fuera_de_tolerancia}\t{estado}")
