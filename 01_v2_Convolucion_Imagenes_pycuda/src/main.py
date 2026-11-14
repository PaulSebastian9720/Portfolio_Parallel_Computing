"""Ejecuta una prueba o un experimento de la práctica 02 (equivale a main.cu).

Imprime tablas separadas por tabuladores y guarda los registros en resultados/*.csv.

Uso:
    uv run src/main.py entorno | manual | no_multiplo | filtro | resolucion | mascara | hilos
                   [--repeticiones N] [--limite-python SEG] [--sin-python]
"""

import argparse
import csv
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from comun import TOLERANCIA, comparar, crear_mascara, guardar_pgm, leer_pgm
from filtros_cpu import convolucion_python, convolucion_referencia_numpy
from filtros_gpu import GpuConvolucion
from pruebas import prueba_manual, prueba_no_multiplo

RAIZ = Path(__file__).resolve().parent.parent
DIR_IMAGENES = RAIZ / "imagenes"
DIR_SALIDAS = RAIZ / "salidas"
DIR_RESULTADOS = RAIZ / "resultados"

COLUMNAS_MEDICIONES = ["version", "filtro", "mascara", "resolucion", "hilos", "bloques",
                       "repeticion", "cpu_a_gpu", "kernel", "gpu_a_cpu", "total",
                       "unidad", "estado"]
COLUMNAS_VALIDACION = ["filtro", "mascara", "resolucion", "hilos", "referencia",
                       "error_maximo", "error_medio", "pixeles_fuera", "estado"]


# Una configuración de un experimento (un solo factor cambia entre configuraciones).
#   filtro      "media", "nitidez" o "laplaciano"
#   k           lado de la máscara
#   resolucion  lado de la imagen cuadrada (2000, 4000 o 6000)
#   hilos       hilos por bloque de PyCUDA
@dataclass
class Configuracion:
    filtro: str
    k: int
    resolucion: int
    hilos: int


# Mismas configuraciones que main.cu de la práctica 01. El segundo valor indica si se
# incluye Python secuencial (el experimento de hilos solo aplica a la GPU).
EXPERIMENTOS = {
    "filtro": ("FILTRO", True, [
        Configuracion("media", 3, 4000, 256),
        Configuracion("nitidez", 3, 4000, 256),
        Configuracion("laplaciano", 3, 4000, 256),
    ]),
    "resolucion": ("RESOLUCION", True, [
        Configuracion("media", 9, 2000, 256),
        Configuracion("media", 9, 4000, 256),
        Configuracion("media", 9, 6000, 256),
    ]),
    "mascara": ("MASCARA", True, [
        Configuracion("media", 3, 4000, 256),
        Configuracion("media", 9, 4000, 256),
        Configuracion("media", 21, 4000, 256),
    ]),
    "hilos": ("HILOS POR BLOQUE", False, [
        Configuracion("media", 9, 4000, 128),
        Configuracion("media", 9, 4000, 256),
        Configuracion("media", 9, 4000, 512),
    ]),
}


# Ejecuta y cronometra una vez la convolución secuencial en Python con perf_counter.
# Parámetros:
#   imagen, ancho, alto, mascara, k  datos de la convolución
#   limite_s  segundos máximos antes de interrumpir
# Retorna: (salida o None si no terminó, milisegundos transcurridos).
def medir_python(imagen, ancho, alto, mascara, k, limite_s):
    inicio = time.perf_counter()
    salida = convolucion_python(imagen, ancho, alto, mascara, k, limite_s)
    return salida, (time.perf_counter() - inicio) * 1000


# Imprime una fila de medición en pantalla y la agrega a la lista del CSV.
# Parámetros:
#   filas  lista donde se acumulan los diccionarios del CSV
#   los demás: valores de cada columna (None = celda vacía)
def registrar(filas, version, c, bloques, repeticion, cpu_gpu, kernel, gpu_cpu, total, estado):
    fila = {
        "version": version, "filtro": c.filtro, "mascara": c.k, "resolucion": c.resolucion,
        "hilos": c.hilos if version == "pycuda" else "", "bloques": bloques or "",
        "repeticion": repeticion,
        "cpu_a_gpu": "" if cpu_gpu is None else f"{cpu_gpu:.4f}",
        "kernel": "" if kernel is None else f"{kernel:.4f}",
        "gpu_a_cpu": "" if gpu_cpu is None else f"{gpu_cpu:.4f}",
        "total": f"{total:.4f}", "unidad": "ms", "estado": estado,
    }
    filas.append(fila)
    print("\t".join(str(fila[col]) for col in COLUMNAS_MEDICIONES), flush=True)


# Escribe una lista de diccionarios en un CSV dentro de resultados/.
# Parámetros:
#   nombre    nombre del archivo
#   columnas  orden de las columnas
#   filas     lista de diccionarios
def escribir_csv(nombre, columnas, filas):
    DIR_RESULTADOS.mkdir(exist_ok=True)
    with open(DIR_RESULTADOS / nombre, "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=columnas)
        escritor.writeheader()
        escritor.writerows(filas)


# Mide cada configuración en Python secuencial (si corresponde) y en PyCUDA: una
# ejecución de calentamiento descartada y `repeticiones` medidas. Después valida PyCUDA
# contra Python secuencial (o contra la referencia NumPy si Python no terminó), guarda la
# imagen filtrada y escribe los CSV.
#
# Python secuencial: si el calentamiento supera el límite, la configuración se registra
# como "no_completada" con el tiempo transcurrido y no se hacen repeticiones; no se
# estiman tiempos.
#
# Parámetros:
#   clave         nombre corto del experimento (para los archivos CSV)
#   gpu           GpuConvolucion inicializada
#   repeticiones  ejecuciones medidas por configuración
#   limite_s      límite de tiempo por ejecución de Python secuencial
#   con_python    False para saltar Python secuencial
def ejecutar_experimento(clave, gpu, repeticiones, limite_s, con_python):
    nombre, incluye_python, configuraciones = EXPERIMENTOS[clave]
    con_python = con_python and incluye_python
    DIR_SALIDAS.mkdir(exist_ok=True)

    print(f"=== EXPERIMENTO {nombre}: MEDICIONES (milisegundos) ===")
    print("\t".join(COLUMNAS_MEDICIONES))

    mediciones, validaciones = [], []
    for c in configuraciones:
        imagen, ancho, alto = leer_pgm(DIR_IMAGENES / f"img_{c.resolucion}.pgm")
        mascara = crear_mascara(c.filtro, c.k)
        salida_python = None

        if con_python:
            salida_python, ms = medir_python(imagen, ancho, alto, mascara, c.k, limite_s)
            if salida_python is None:
                registrar(mediciones, "python", c, None, "calentamiento", None, None, None,
                          ms, f"no_completada (limite {limite_s:.0f} s)")
            else:
                for r in range(1, repeticiones + 1):
                    salida_python, ms = medir_python(imagen, ancho, alto, mascara, c.k, limite_s)
                    if salida_python is None:
                        registrar(mediciones, "python", c, None, r, None, None, None,
                                  ms, f"no_completada (limite {limite_s:.0f} s)")
                        break
                    registrar(mediciones, "python", c, None, r, None, None, None, ms, "ok")

        gpu.convolucion(imagen, mascara, ancho, alto, c.k, c.hilos)
        for r in range(1, repeticiones + 1):
            salida_gpu, t = gpu.convolucion(imagen, mascara, ancho, alto, c.k, c.hilos)
            registrar(mediciones, "pycuda", c, t.bloques, r, t.cpu_gpu, t.kernel,
                      t.gpu_cpu, t.total, "ok")

        if salida_python is not None:
            referencia, tipo_referencia = salida_python, "python_secuencial"
        else:
            referencia = convolucion_referencia_numpy(imagen, ancho, alto, mascara, c.k)
            tipo_referencia = "numpy_validacion"

        error = comparar(referencia, salida_gpu)
        validaciones.append({
            "filtro": c.filtro, "mascara": c.k, "resolucion": c.resolucion,
            "hilos": c.hilos, "referencia": tipo_referencia,
            "error_maximo": f"{error.maximo:.6f}", "error_medio": f"{error.medio:.6f}",
            "pixeles_fuera": error.fuera_de_tolerancia,
            "estado": "OK" if error.fuera_de_tolerancia == 0 else "FALLO",
        })

        archivo = f"{c.filtro}_k{c.k}_{c.resolucion}.pgm"
        guardar_pgm(DIR_SALIDAS / archivo, referencia, ancho, alto, c.filtro == "laplaciano")

    print(f"\n=== EXPERIMENTO {nombre}: VALIDACION PyCUDA vs referencia "
          f"(tolerancia {TOLERANCIA:.2f}) ===")
    print("\t".join(COLUMNAS_VALIDACION))
    for v in validaciones:
        print("\t".join(str(v[col]) for col in COLUMNAS_VALIDACION))

    escribir_csv(f"{clave}_mediciones.csv", COLUMNAS_MEDICIONES, mediciones)
    escribir_csv(f"{clave}_validacion.csv", COLUMNAS_VALIDACION, validaciones)
    print(f"\nCSV guardados en {DIR_RESULTADOS}")


# Devuelve el nombre del procesador (Linux: /proc/cpuinfo; otros: platform).
def nombre_cpu():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as archivo:
            for linea in archivo:
                if linea.startswith("model name"):
                    return linea.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "desconocido"


# Devuelve la RAM total en GiB, o "desconocida" si el sistema no la expone.
def ram_total():
    try:
        bytes_ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        return f"{bytes_ram / 1024 ** 3:.1f} GiB"
    except (ValueError, OSError, AttributeError):
        return "desconocida"


# Devuelve la versión de nvcc (CUDA Toolkit con el que PyCUDA compila el kernel).
def version_nvcc():
    try:
        salida = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout
        return salida.strip().splitlines()[-1]
    except (OSError, IndexError):
        return "nvcc no encontrado"


# Imprime el equipo y las versiones que la guía pide registrar, más los tiempos de
# preparación de la GPU (contexto y compilación del kernel).
# Parámetros:
#   gpu  GpuConvolucion inicializada
def mostrar_entorno(gpu):
    import pycuda
    import pycuda.driver as cuda

    driver = cuda.get_driver_version()
    dispositivo = gpu.dispositivo
    capacidad = dispositivo.compute_capability()
    print("=== ENTORNO ===")
    print(f"cpu\t{nombre_cpu()}")
    print(f"nucleos_logicos\t{os.cpu_count()}")
    print(f"ram\t{ram_total()}")
    print(f"gpu\t{dispositivo.name()} (sm_{capacidad[0]}{capacidad[1]})")
    print(f"memoria_gpu\t{dispositivo.total_memory() / 1024 ** 3:.1f} GiB")
    print(f"max_hilos_por_bloque\t{dispositivo.get_attribute(cuda.device_attribute.MAX_THREADS_PER_BLOCK)}")
    print(f"sistema_operativo\t{platform.platform()}")
    print(f"python\t{sys.version.split()[0]}")
    print(f"numpy\t{np.__version__}")
    print(f"pycuda\t{pycuda.VERSION_TEXT}")
    print(f"cuda_driver\t{driver // 1000}.{(driver % 1000) // 10}")
    print(f"cuda_toolkit_pycuda\t{'.'.join(map(str, cuda.get_version()))}")
    print(f"nvcc\t{version_nvcc()}")
    print(f"ms_inicializar_contexto\t{gpu.ms_contexto:.4f}")
    print(f"ms_compilar_cargar_kernel\t{gpu.ms_compilacion:.4f}")


# Lee los argumentos, inicializa la GPU una sola vez y ejecuta la opción pedida.
def main():
    analizador = argparse.ArgumentParser(description="Práctica 02: convolución con PyCUDA")
    analizador.add_argument("opcion", choices=["entorno", "manual", "no_multiplo", *EXPERIMENTOS])
    analizador.add_argument("--repeticiones", type=int, default=5,
                            help="ejecuciones medidas tras el calentamiento (por defecto 5)")
    analizador.add_argument("--limite-python", type=float, default=600,
                            help="segundos máximos por ejecución de Python secuencial (por defecto 600)")
    analizador.add_argument("--sin-python", action="store_true",
                            help="no ejecutar Python secuencial (solo PyCUDA)")
    args = analizador.parse_args()

    gpu = GpuConvolucion()
    try:
        if args.opcion == "entorno":
            mostrar_entorno(gpu)
        elif args.opcion == "manual":
            prueba_manual(gpu)
        elif args.opcion == "no_multiplo":
            prueba_no_multiplo(gpu)
        else:
            ejecutar_experimento(args.opcion, gpu, args.repeticiones,
                                 args.limite_python, not args.sin_python)
    finally:
        gpu.cerrar()


if __name__ == "__main__":
    main()
