"""Tipos, máscaras, lectura/escritura de PGM y comparación de salidas."""

from dataclasses import dataclass

import numpy as np

# Error admitido entre versiones, en la escala 0-255. El peor caso es la media 21x21:
# 441 sumas * 255 * épsilon de float32 (1.19e-7) da 0.013, y 0.02 sigue muy por debajo de
# un nivel de gris.
TOLERANCIA = 0.02


# Tiempos de una ejecución en GPU, en ms; total abarca las tres etapas.
@dataclass
class TiemposGpu:
    cpu_gpu: float
    kernel: float
    gpu_cpu: float
    total: float
    bloques: int


@dataclass
class ErrorNumerico:
    maximo: float = 0.0
    medio: float = 0.0
    fuera_de_tolerancia: int = 0


# Coeficientes de la máscara fila tras fila, en float32 como el float de C. k solo se usa
# en la media; nitidez y laplaciano son siempre 3x3.
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


# Lee un PGM de 8 bits y lo devuelve como arreglo plano float32 (fila tras fila), que es
# la forma en que el kernel recorre la imagen.
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


# Pasa el resultado a 8 bits para verlo: redondea y recorta a [0, 255]. En el laplaciano
# (absoluto=True) se toma antes el valor absoluto, porque un valor negativo también es un
# borde. Solo se usa para guardar imágenes; la validación compara los float sin recortar.
def guardar_pgm(ruta, valores, ancho, alto, absoluto):
    datos = np.abs(valores) if absoluto else valores
    bytes_imagen = np.clip(np.rint(datos), 0, 255).astype(np.uint8)
    with open(ruta, "wb") as archivo:
        archivo.write(f"P5\n{ancho} {alto}\n255\n".encode("ascii"))
        archivo.write(bytes_imagen.tobytes())


# Error absoluto máximo y medio entre dos salidas float, y píxeles por encima de la
# tolerancia. Un NaN es un píxel que ningún hilo escribió: cuenta como fuera.
def comparar(referencia, otra):
    diferencia = np.abs(referencia.astype(np.float64) - otra.astype(np.float64))
    validos = ~np.isnan(diferencia)
    error = ErrorNumerico()
    error.fuera_de_tolerancia = int(np.count_nonzero(~(diferencia <= TOLERANCIA)))
    if np.any(validos):
        error.maximo = float(diferencia[validos].max())
        error.medio = float(diferencia[validos].sum() / diferencia.size)
    return error
