"""Convolución secuencial en Python (equivale a convolucionCPU de filtros_cpu.cpp).

La versión evaluada está escrita a mano con bucles explícitos sobre píxeles y
coeficientes: no usa pandas, ni funciones de filtrado de NumPy/SciPy/OpenCV, ni JIT, ni
multiprocesamiento, ni llama a la versión en C. NumPy solo almacena los datos y aporta
el tipo np.float32, para que la acumulación tenga la misma precisión que el float de C.
"""

import time

import numpy as np


# Convolución secuencial con extensión con ceros: un vecino fuera de la imagen aportaría
# 0, así que se omite. Recorre los mismos cuatro bucles que la versión en C y acumula en
# np.float32 (suma, entrada y máscara son escalares float32), por lo que cada producto y
# cada suma se redondean a 32 bits igual que en C.
#
# Como en Python esto puede tardar horas en imágenes grandes, se admite un límite de
# tiempo. Se revisa una vez por fila (un costo despreciable frente a ancho*k*k
# operaciones). Si se supera, la ejecución se marca como no completada.
#
# Parámetros:
#   entrada   arreglo plano np.float32 de ancho*alto (fila tras fila)
#   ancho, alto  dimensiones de la imagen
#   mascara   arreglo np.float32 de k*k coeficientes
#   k         lado de la máscara (impar)
#   limite_s  segundos máximos de ejecución; None = sin límite
# Retorna: arreglo np.float32 de ancho*alto, o None si se superó el límite.
def convolucion_python(entrada, ancho, alto, mascara, k, limite_s=None):
    pixeles = list(entrada.astype(np.float32))
    coeficientes = list(mascara.astype(np.float32))
    salida = [np.float32(0.0)] * (ancho * alto)
    radio = k // 2
    inicio = time.perf_counter()

    for fila in range(alto):
        if limite_s is not None and time.perf_counter() - inicio > limite_s:
            return None

        for col in range(ancho):
            suma = np.float32(0.0)

            for i in range(k):
                for j in range(k):
                    f = fila + i - radio
                    c = col + j - radio
                    if 0 <= f < alto and 0 <= c < ancho:
                        suma += pixeles[f * ancho + c] * coeficientes[i * k + j]

            salida[fila * ancho + col] = suma

    return np.array(salida, dtype=np.float32)


# Referencia de VALIDACIÓN, no evaluada ni medida. Se usa solo para comprobar PyCUDA en
# las configuraciones donde Python secuencial no alcanzó a terminar (por el límite de
# tiempo). Hace la misma operación sumando ventanas desplazadas de la imagen con bordes
# en cero, en el mismo orden (i, j) y en float32, así que coincide con la versión en C.
# Parámetros: los mismos que convolucion_python (sin límite).
# Retorna: arreglo plano np.float32 de ancho*alto.
def convolucion_referencia_numpy(entrada, ancho, alto, mascara, k):
    radio = k // 2
    imagen = entrada.reshape(alto, ancho).astype(np.float32)
    con_bordes = np.pad(imagen, radio, mode="constant", constant_values=0)
    suma = np.zeros((alto, ancho), dtype=np.float32)
    for i in range(k):
        for j in range(k):
            suma += con_bordes[i:i + alto, j:j + ancho] * mascara[i * k + j]
    return suma.reshape(-1)
