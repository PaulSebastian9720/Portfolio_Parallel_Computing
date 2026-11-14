"""Constantes, estructuras de datos y utilidades compartidas (equivale a comun.h y a
las funciones de lectura/escritura de main.cu de la práctica 01).

Tipos usados en todo el programa: las imágenes y máscaras son arreglos NumPy de tipo
np.float32 (float de 32 bits, igual que el float de C/CUDA) y contiguos en memoria
(orden C, fila tras fila), que es la disposición que espera el kernel.
"""

from dataclasses import dataclass

import numpy as np

# Error máximo admitido entre implementaciones, en escala 0-255. Cota del peor caso
# (media 21x21): 441 * 255 * 1.19e-7 = 0.013; y 0.02 es muy inferior a un nivel de gris.
TOLERANCIA = 0.02


# Tiempos de una ejecución en GPU, en milisegundos.
#   cpu_gpu  copia de la imagen CPU -> GPU
#   kernel   ejecución del kernel
#   gpu_cpu  copia del resultado GPU -> CPU
#   total    las tres etapas juntas
#   bloques  bloques lanzados en la dimensión X
@dataclass
class TiemposGpu:
    cpu_gpu: float
    kernel: float
    gpu_cpu: float
    total: float
    bloques: int


# Resultado de comparar dos salidas float.
#   maximo               error absoluto máximo
#   medio                error absoluto medio
#   fuera_de_tolerancia  píxeles con error > TOLERANCIA (incluye NaN)
@dataclass
class ErrorNumerico:
    maximo: float = 0.0
    medio: float = 0.0
    fuera_de_tolerancia: int = 0


# Devuelve los k*k coeficientes de la máscara, fila tras fila.
# Parámetros:
#   filtro  "media", "nitidez" o "laplaciano"
#   k       lado de la máscara; solo se usa en "media" (nitidez y laplaciano son 3x3)
# Retorna: arreglo np.float32 contiguo de k*k elementos.
def crear_mascara(filtro, k):
    if filtro == "nitidez":
        valores = [0, -1, 0,
                   -1, 5, -1,
                   0, -1, 0]
    elif filtro == "laplaciano":
        valores = [0, 1, 0,
                   1, -4, 1,
                   0, 1, 0]
    else:
        valores = [np.float32(1.0) / np.float32(k * k)] * (k * k)
    return np.ascontiguousarray(valores, dtype=np.float32)


# Lee un PGM binario (P5) de 8 bits generado por preparar_imagenes.py (o preparar.cpp).
# Parámetros:
#   ruta  ruta del archivo .pgm
# Retorna: (imagen, ancho, alto); imagen es un arreglo plano np.float32 de ancho*alto.
def leer_pgm(ruta):
    with open(ruta, "rb") as archivo:
        cabecera = []
        while len(cabecera) < 4:
            linea = archivo.readline()
            if not linea:
                break
            linea = linea.split(b"#")[0]
            cabecera.extend(linea.split())
        datos = archivo.read()

    if len(cabecera) < 4 or cabecera[0] != b"P5":
        raise ValueError(f"No se pudo leer {ruta}. Genere las imagenes con preparar_imagenes.py")

    ancho, alto = int(cabecera[1]), int(cabecera[2])
    bytes_imagen = np.frombuffer(datos, dtype=np.uint8, count=ancho * alto)
    return bytes_imagen.astype(np.float32), ancho, alto


# Convierte el resultado numérico a 8 bits y lo guarda como PGM. Media y nitidez se
# recortan a [0, 255]. En el laplaciano se usa el valor absoluto, porque sus respuestas
# negativas también indican cambios de intensidad. Se hace después de validar los floats.
# Parámetros:
#   ruta      archivo de destino
#   valores   arreglo plano float32 de ancho*alto
#   ancho, alto  dimensiones
#   absoluto  True para aplicar valor absoluto antes de recortar
def guardar_pgm(ruta, valores, ancho, alto, absoluto):
    datos = np.abs(valores) if absoluto else valores
    bytes_imagen = np.clip(np.rint(datos), 0, 255).astype(np.uint8)
    with open(ruta, "wb") as archivo:
        archivo.write(f"P5\n{ancho} {alto}\n255\n".encode("ascii"))
        archivo.write(bytes_imagen.tobytes())


# Compara dos salidas float antes de convertirlas a 8 bits. NumPy solo se usa aquí para
# calcular el error, no para la convolución. Las diferencias NaN (píxeles que ningún
# hilo calculó) cuentan como fuera de tolerancia y no entran en máximo ni medio.
# Parámetros:
#   referencia, otra  arreglos float32 del mismo tamaño
# Retorna: ErrorNumerico.
def comparar(referencia, otra):
    diferencia = np.abs(referencia.astype(np.float64) - otra.astype(np.float64))
    validos = ~np.isnan(diferencia)
    error = ErrorNumerico()
    error.fuera_de_tolerancia = int(np.count_nonzero(~(diferencia <= TOLERANCIA)))
    if np.any(validos):
        error.maximo = float(diferencia[validos].max())
        error.medio = float(diferencia[validos].sum() / diferencia.size)
    return error
