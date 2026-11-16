"""Ejecuta una prueba o un experimento de la práctica 02 (equivale a main.cu).

Imprime tablas separadas por tabuladores y guarda los registros en resultados/*.csv.

Uso:
    uv run src/main.py entorno | manual | no_multiplo | comparacion | filtro | resolucion
                       | mascara | hilos  [--repeticiones N] [--segundos-muestra S]
"""

import argparse
import csv
import ctypes
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from comun import TOLERANCIA, comparar, crear_mascara, guardar_pgm, leer_pgm
from filtros_cpu import convolucion_python
from filtros_gpu import GpuConvolucion
from pruebas import prueba_manual, prueba_no_multiplo
from referencia_c import ejecutar_puente

RAIZ = Path(__file__).resolve().parent.parent
DIR_IMAGENES = RAIZ / "imagenes"
DIR_SALIDAS = RAIZ / "salidas"
DIR_RESULTADOS = RAIZ / "resultados"

COLUMNAS_MEDICIONES = ["version", "filtro", "mascara", "resolucion", "hilos", "bloques",
                       "repeticion", "filas_medidas", "ms_muestra", "cpu_a_gpu", "kernel",
                       "gpu_a_cpu", "total", "unidad", "estado"]
COLUMNAS_VALIDACION = ["filtro", "mascara", "resolucion", "hilos", "version",
                       "filas_comparadas", "error_maximo", "error_medio", "pixeles_fuera",
                       "estado"]


# Un solo factor cambia entre las configuraciones de un experimento.
@dataclass
class Configuracion:
    filtro: str
    k: int
    resolucion: int
    hilos: int


# python: "completo", "estimado" (bloque de filas) o None.
# c_medido: False cuando los tiempos de C/C++ y CUDA C se reutilizan de la práctica 01 y
# el puente solo aporta la salida de referencia.
@dataclass
class Experimento:
    nombre: str
    python: str | None
    c_medido: bool
    configuraciones: list


# Las configuraciones de alta resolución son las de main.cu de la práctica 01; en 500x500
# las cuatro versiones terminan completas.
EXPERIMENTOS = {
    "comparacion": Experimento("COMPARACION 500x500", "completo", True, [
        Configuracion("media", 3, 500, 256),
        Configuracion("nitidez", 3, 500, 256),
        Configuracion("laplaciano", 3, 500, 256),
        Configuracion("media", 9, 500, 256),
        Configuracion("media", 21, 500, 256),
    ]),
    "filtro": Experimento("FILTRO", "estimado", False, [
        Configuracion("media", 3, 4000, 256),
        Configuracion("nitidez", 3, 4000, 256),
        Configuracion("laplaciano", 3, 4000, 256),
    ]),
    "resolucion": Experimento("RESOLUCION", "estimado", False, [
        Configuracion("media", 9, 2000, 256),
        Configuracion("media", 9, 4000, 256),
        Configuracion("media", 9, 6000, 256),
    ]),
    "mascara": Experimento("MASCARA", "estimado", False, [
        Configuracion("media", 3, 4000, 256),
        Configuracion("media", 9, 4000, 256),
        Configuracion("media", 21, 4000, 256),
    ]),
    "hilos": Experimento("HILOS POR BLOQUE", None, False, [
        Configuracion("media", 9, 4000, 128),
        Configuracion("media", 9, 4000, 256),
        Configuracion("media", 9, 4000, 512),
    ]),
}


def texto(valor):
    if valor is None or valor == "":
        return ""
    return f"{float(valor):.4f}"


def registrar(filas, c, **datos):
    fila = {col: "" for col in COLUMNAS_MEDICIONES}
    fila.update(filtro=c.filtro, mascara=c.k, resolucion=c.resolucion, unidad="ms")
    for col in ("ms_muestra", "cpu_a_gpu", "kernel", "gpu_a_cpu", "total"):
        datos[col] = texto(datos.get(col))
    fila.update(datos)
    filas.append(fila)
    print("\t".join(str(fila[col]) for col in COLUMNAS_MEDICIONES), flush=True)


def escribir_csv(nombre, columnas, filas):
    DIR_RESULTADOS.mkdir(exist_ok=True)
    with open(DIR_RESULTADOS / nombre, "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=columnas)
        escritor.writeheader()
        escritor.writerows(filas)


# Python completo: calentamiento y repeticiones; perf_counter rodea solo la convolución.
def medir_python_completo(mediciones, c, imagen, ancho, alto, mascara, repeticiones):
    convolucion_python(imagen, ancho, alto, mascara, c.k)
    salida = None
    for r in range(1, repeticiones + 1):
        inicio = time.perf_counter()
        salida = convolucion_python(imagen, ancho, alto, mascara, c.k)
        ms = (time.perf_counter() - inicio) * 1000
        registrar(mediciones, c, version="python", repeticion=r, filas_medidas=alto,
                  total=ms, estado="ok")
    return salida


# Python estimado para alta resolución, donde una ejecución completa tarda de minutos a
# horas. El calentamiento calcula 2 filas centrales y de paso da el costo por fila; con él
# se eligen n filas para que cada repetición dure unos `segundos`. Cada repetición mide esas
# n filas y se extrapola: (ms / n) * alto. Las filas de borde leen menos vecinos, así que
# la estimación queda un poco por encima. El CSV guarda la muestra medida (ms_muestra) y la
# estimación (total) con estado "estimado".
def medir_python_estimado(mediciones, c, imagen, ancho, alto, mascara, repeticiones,
                          segundos):
    centro = alto // 2
    inicio = time.perf_counter()
    convolucion_python(imagen, ancho, alto, mascara, c.k, centro, centro + 2)
    ms_fila = (time.perf_counter() - inicio) * 1000 / 2

    n = min(alto, max(1, math.ceil(segundos * 1000 / ms_fila)))
    fila_inicio = max(0, centro - n // 2)
    fila_fin = fila_inicio + n

    salida = None
    for r in range(1, repeticiones + 1):
        inicio = time.perf_counter()
        salida = convolucion_python(imagen, ancho, alto, mascara, c.k, fila_inicio, fila_fin)
        ms = (time.perf_counter() - inicio) * 1000
        registrar(mediciones, c, version="python", repeticion=r, filas_medidas=n,
                  ms_muestra=ms, total=ms / n * alto,
                  estado="ok" if n == alto else "estimado")
    return salida, fila_inicio, fila_fin


# Calentamiento de la GPU: al menos 1 s y 3 ejecuciones. Con una sola no bastaba: tras
# varios segundos solo de CPU, la GPU del portátil baja su consumo y las primeras
# repeticiones salían hasta 3 veces más lentas.
def calentar_gpu(gpu, imagen, mascara, ancho, alto, c):
    inicio, ejecuciones = time.perf_counter(), 0
    while ejecuciones < 3 or time.perf_counter() - inicio < 1.0:
        gpu.convolucion(imagen, mascara, ancho, alto, c.k, c.hilos)
        ejecuciones += 1


def validar(validaciones, c, version, referencia, salida, filas_comparadas):
    error = comparar(referencia, salida)
    validaciones.append({
        "filtro": c.filtro, "mascara": c.k, "resolucion": c.resolucion, "hilos": c.hilos,
        "version": version, "filas_comparadas": filas_comparadas,
        "error_maximo": f"{error.maximo:.6f}", "error_medio": f"{error.medio:.6f}",
        "pixeles_fuera": error.fuera_de_tolerancia,
        "estado": "OK" if error.fuera_de_tolerancia == 0 else "FALLO",
    })


# Por configuración: C/C++ y CUDA C (puente), PyCUDA y Python, en ese orden para que las
# dos versiones GPU corran seguidas. Luego se validan Python, CUDA C y PyCUDA contra la
# salida float de C/C++ y se guardan las imágenes de C/C++ y PyCUDA.
def ejecutar_experimento(clave, gpu, repeticiones, segundos):
    experimento = EXPERIMENTOS[clave]
    DIR_SALIDAS.mkdir(exist_ok=True)

    print(f"=== EXPERIMENTO {experimento.nombre}: MEDICIONES (milisegundos) ===")
    print("\t".join(COLUMNAS_MEDICIONES), flush=True)

    mediciones, validaciones = [], []
    for c in experimento.configuraciones:
        ruta = DIR_IMAGENES / f"img_{c.resolucion}.pgm"
        imagen, ancho, alto = leer_pgm(ruta)
        mascara = crear_mascara(c.filtro, c.k)

        reps_c = repeticiones if experimento.c_medido else 0
        salida_c, salida_cuda_c, filas_c = ejecutar_puente(c.filtro, c.k, ruta, c.hilos, reps_c)
        for version, rep, bloques, cpu_gpu, kernel, gpu_cpu, total in filas_c:
            registrar(mediciones, c, version=version, repeticion=rep, filas_medidas=alto,
                      hilos=c.hilos if version == "cuda_c" else "", bloques=bloques,
                      cpu_a_gpu=cpu_gpu, kernel=kernel, gpu_a_cpu=gpu_cpu, total=total,
                      estado="ok")

        calentar_gpu(gpu, imagen, mascara, ancho, alto, c)
        for r in range(1, repeticiones + 1):
            salida_gpu, t = gpu.convolucion(imagen, mascara, ancho, alto, c.k, c.hilos)
            registrar(mediciones, c, version="pycuda", hilos=c.hilos, bloques=t.bloques,
                      repeticion=r, filas_medidas=alto, cpu_a_gpu=t.cpu_gpu,
                      kernel=t.kernel, gpu_a_cpu=t.gpu_cpu, total=t.total, estado="ok")

        salida_python, filas = None, (0, alto)
        if experimento.python == "completo":
            salida_python = medir_python_completo(mediciones, c, imagen, ancho, alto,
                                                  mascara, repeticiones)
        elif experimento.python == "estimado":
            salida_python, *filas = medir_python_estimado(mediciones, c, imagen, ancho, alto,
                                                          mascara, repeticiones, segundos)

        if salida_python is not None:
            desde, hasta = filas
            validar(validaciones, c, "python", salida_c[desde * ancho:hasta * ancho],
                    salida_python, hasta - desde)
        validar(validaciones, c, "cuda_c", salida_c, salida_cuda_c, alto)
        validar(validaciones, c, "pycuda", salida_c, salida_gpu, alto)

        base = f"{c.filtro}_k{c.k}_{c.resolucion}"
        absoluto = c.filtro == "laplaciano"
        guardar_pgm(DIR_SALIDAS / f"{base}_c.pgm", salida_c, ancho, alto, absoluto)
        guardar_pgm(DIR_SALIDAS / f"{base}_pycuda.pgm", salida_gpu, ancho, alto, absoluto)

    print(f"\n=== EXPERIMENTO {experimento.nombre}: VALIDACION contra C/C++ secuencial "
          f"(tolerancia {TOLERANCIA:.2f}) ===")
    print("\t".join(COLUMNAS_VALIDACION))
    for v in validaciones:
        print("\t".join(str(v[col]) for col in COLUMNAS_VALIDACION))

    escribir_csv(f"{clave}_mediciones.csv", COLUMNAS_MEDICIONES, mediciones)
    escribir_csv(f"{clave}_validacion.csv", COLUMNAS_VALIDACION, validaciones)
    print(f"\nCSV guardados en {DIR_RESULTADOS}")


def nombre_cpu():
    if sys.platform == "win32":
        import winreg
        try:
            clave = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                   r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(clave, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as archivo:
            for linea in archivo:
                if linea.startswith("model name"):
                    return linea.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "desconocido"


def ram_total():
    if sys.platform == "win32":
        class Estado(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        estado = Estado()
        estado.dwLength = ctypes.sizeof(Estado)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(estado)):
            return f"{estado.ullTotalPhys / 1024 ** 3:.1f} GiB"
    try:
        bytes_ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        return f"{bytes_ram / 1024 ** 3:.1f} GiB"
    except (ValueError, OSError, AttributeError):
        return "desconocida"


def version_nvcc():
    try:
        salida = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout
        return salida.strip().splitlines()[-1]
    except (OSError, IndexError):
        return "nvcc no encontrado"


# Equipo, versiones y tiempos de preparación de la GPU (contexto y compilación).
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


# La GPU se inicializa una sola vez, antes de cualquier medición.
def main():
    analizador = argparse.ArgumentParser(description="Práctica 02: convolución con PyCUDA")
    analizador.add_argument("opcion", choices=["entorno", "manual", "no_multiplo", *EXPERIMENTOS])
    analizador.add_argument("--repeticiones", type=int, default=5,
                            help="ejecuciones medidas tras el calentamiento (por defecto 5)")
    analizador.add_argument("--segundos-muestra", type=float, default=5,
                            help="duración aproximada de cada muestra de Python estimado")
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
            ejecutar_experimento(args.opcion, gpu, args.repeticiones, args.segundos_muestra)
    finally:
        gpu.cerrar()


if __name__ == "__main__":
    main()
