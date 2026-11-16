"""Pruebas de corrección: matriz calculada a mano y tamaño no múltiplo de los hilos."""

import numpy as np

from comun import comparar, crear_mascara, guardar_pgm
from filtros_cpu import convolucion_python
from referencia_c import DIR_TEMPORAL, ejecutar_puente

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


# Valores calculados a mano sobre la matriz de la Figura 1, incluidos píxeles de borde.
# Ejemplo, nitidez en (0,0): 5*35 - 40 (derecha) - 40 (abajo) = 95; los vecinos de fuera
# valen 0.
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


# gpu=None prueba Python secuencial; con un GpuConvolucion prueba PyCUDA (128 hilos).
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


def prueba_manual(gpu):
    print("=== PRUEBA MANUAL ===")
    print("impl\tprueba\tfila\tcol\tesperado\tobtenido\testado")
    ejecutar_casos_manuales(None)
    ejecutar_casos_manuales(gpu)


# Prueba de límites con una imagen de 1003x997: sus píxeles no son múltiplo de 128, 256 ni
# 512, así que al último bloque siempre le sobran hilos. La salida de la GPU empieza en NaN;
# si un píxel queda sin calcular, la comparación lo cuenta como fuera de tolerancia.
# Python, CUDA C y PyCUDA se comparan contra C/C++.
def prueba_no_multiplo(gpu):
    print("=== TAMANO NO MULTIPLO ===")
    ancho, alto = 1003, 997
    imagen = np.array([(i * 37) % 256 for i in range(ancho * alto)], dtype=np.float32)
    mascara = crear_mascara("media", 3)

    ruta = DIR_TEMPORAL / "no_multiplo.pgm"
    DIR_TEMPORAL.mkdir(parents=True, exist_ok=True)
    guardar_pgm(ruta, imagen, ancho, alto, False)

    print("(calculando Python secuencial completo...)", flush=True)
    salida_python = convolucion_python(imagen, ancho, alto, mascara, 3)

    print("imagen	version	hilos	bloques	error_maximo	error_medio	pixeles_fuera	estado")
    for hilos in (128, 256, 512):
        salida_c, salida_cuda_c, _ = ejecutar_puente("media", 3, ruta, hilos, 0)
        salida_gpu, tiempos = gpu.convolucion(imagen, mascara, ancho, alto, 3, hilos)
        casos = [("cuda_c", salida_cuda_c), ("pycuda", salida_gpu)]
        if hilos == 128:
            casos.insert(0, ("python", salida_python))
        for version, salida in casos:
            error = comparar(salida_c, salida)
            estado = "OK" if error.fuera_de_tolerancia == 0 else "FALLO"
            columna_hilos = "" if version == "python" else hilos
            columna_bloques = "" if version == "python" else tiempos.bloques
            print(f"{ancho}x{alto}	{version}	{columna_hilos}	{columna_bloques}	"
                  f"{error.maximo:.6f}	{error.medio:.6f}	{error.fuera_de_tolerancia}	{estado}")
    ruta.unlink()
