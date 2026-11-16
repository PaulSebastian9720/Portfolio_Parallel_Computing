"""Convolución secuencial en Python, con bucles explícitos (equivale a filtros_cpu.cpp)."""

import numpy as np


# Mismos cuatro bucles que convolucionCPU en C. No se usan funciones de filtrado de
# NumPy/OpenCV, JIT ni multiprocesamiento: NumPy solo guarda los datos. La suma y los
# operandos son escalares np.float32, así cada producto y cada suma se redondean a 32 bits
# igual que el float de C. Vecino fuera de la imagen: vale 0 y se omite.
#
# fila_inicio y fila_fin permiten calcular solo un bloque de filas (para estimar el tiempo
# en imágenes grandes). Las filas del bloque leen a sus vecinos de la imagen completa, así
# que dan los mismos valores que en la convolución entera.
def convolucion_python(entrada, ancho, alto, mascara, k, fila_inicio=0, fila_fin=None):
    if fila_fin is None:
        fila_fin = alto
    radio = k // 2

    # Solo se pasan a lista las filas que el bloque lee: las suyas y `radio` filas de cada
    # lado. Convertir la imagen completa en cada llamada costaría lo mismo para 7 filas que
    # para 4000 y falsearía la estimación.
    desde = max(0, fila_inicio - radio)
    hasta = min(alto, fila_fin + radio)
    pixeles = list(entrada[desde * ancho:hasta * ancho].astype(np.float32))
    coeficientes = list(mascara.astype(np.float32))
    salida = [np.float32(0.0)] * ((fila_fin - fila_inicio) * ancho)

    for fila in range(fila_inicio, fila_fin):
        for col in range(ancho):
            suma = np.float32(0.0)

            for i in range(k):
                for j in range(k):
                    f = fila + i - radio
                    c = col + j - radio
                    if 0 <= f < alto and 0 <= c < ancho:
                        suma += pixeles[(f - desde) * ancho + c] * coeficientes[i * k + j]

            salida[(fila - fila_inicio) * ancho + col] = suma

    return np.array(salida, dtype=np.float32)
