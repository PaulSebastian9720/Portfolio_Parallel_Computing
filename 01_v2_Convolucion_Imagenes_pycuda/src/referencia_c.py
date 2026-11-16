"""Llama al ejecutable referencia_c/puente, que enlaza sin cambios filtros_cpu.cpp y
filtros_gpu.cu de la práctica 01. Da la salida de referencia C/C++ y mide C/C++ y CUDA C
donde la práctica 01 no lo hizo. Python nunca usa estas salidas como su convolución."""

import subprocess
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
DIR_TEMPORAL = RAIZ / "salidas" / "referencia"


def ruta_puente():
    for ruta in (RAIZ / "referencia_c" / "build" / "Release" / "puente.exe",
                 RAIZ / "referencia_c" / "build" / "puente"):
        if ruta.exists():
            return ruta
    raise FileNotFoundError("Compile referencia_c (ver README) antes de ejecutar")


# Devuelve las salidas float de C/C++ y CUDA C y las filas de tiempos que imprime el
# puente. Con repeticiones=0 solo calcula, sin medir.
def ejecutar_puente(filtro, k, ruta_pgm, hilos, repeticiones):
    DIR_TEMPORAL.mkdir(parents=True, exist_ok=True)
    base = DIR_TEMPORAL / f"{filtro}_k{k}_{Path(ruta_pgm).stem}_{hilos}"
    proceso = subprocess.run(
        [str(ruta_puente()), filtro, str(k), str(ruta_pgm), str(hilos), str(base),
         str(repeticiones)],
        capture_output=True, text=True, check=True)

    filas = [linea.split("\t") for linea in proceso.stdout.strip().splitlines()[1:]]
    salidas = []
    for sufijo in ("_c.f32", "_cuda_c.f32"):
        archivo = Path(str(base) + sufijo)
        salidas.append(np.fromfile(archivo, dtype=np.float32))
        archivo.unlink()
    return salidas[0], salidas[1], filas
