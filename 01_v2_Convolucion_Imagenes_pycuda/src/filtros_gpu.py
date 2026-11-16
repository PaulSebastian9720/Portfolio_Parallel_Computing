"""Convolución en GPU con PyCUDA: anfitrión en Python, kernel en CUDA C."""

import time

import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from comun import TiemposGpu

# Mismo kernel de filtros_gpu.cu (práctica 01), como cadena para SourceModule.
# Cada hilo calcula un píxel. Con bloques e hilos solo en X, el índice global
# blockIdx.x * blockDim.x + threadIdx.x recorre la imagen como un arreglo plano, y de él
# salen la fila (indice / ancho) y la columna (indice % ancho). Los hilos sobrantes del
# último bloque caen fuera de la imagen y salen sin escribir. Vecino fuera de la imagen:
# vale 0, así que se omite.
KERNEL_CONVOLUCION = """
__global__ void kernelConvolucion(const float* entrada, float* salida, const float* mascara,
                                  int ancho, int alto, int k) {
    const int indice = blockIdx.x * blockDim.x + threadIdx.x;
    if (indice >= ancho * alto) {
        return;
    }

    const int fila = indice / ancho;
    const int col = indice % ancho;
    const int radio = k / 2;

    float suma = 0.0f;
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            const int f = fila + i - radio;
            const int c = col + j - radio;
            if (f >= 0 && f < alto && c >= 0 && c < ancho) {
                suma += entrada[f * ancho + c] * mascara[i * k + j];
            }
        }
    }
    salida[indice] = suma;
}
"""


# Prepara la GPU una sola vez: contexto y kernel compilado. No se usa pycuda.autoinit
# porque así se puede medir aparte cuánto tarda crear el contexto y cuánto compilar.
class GpuConvolucion:

    def __init__(self, numero_gpu=0):
        inicio = time.perf_counter()
        cuda.init()
        self.dispositivo = cuda.Device(numero_gpu)
        self.contexto = self.dispositivo.make_context()
        self.ms_contexto = (time.perf_counter() - inicio) * 1000

        # SourceModule llama a nvcc; si el kernel ya se compiló antes, lo toma de la caché.
        try:
            inicio = time.perf_counter()
            modulo = SourceModule(KERNEL_CONVOLUCION)
            self.funcion = modulo.get_function("kernelConvolucion")
            self.ms_compilacion = (time.perf_counter() - inicio) * 1000
        except Exception:
            self.cerrar()
            raise

    def cerrar(self):
        self.contexto.pop()
        self.contexto.detach()

    # Convolución en la GPU con las tres etapas medidas por separado.
    #
    # Fuera de la medición: reservar memoria, llenar la salida con 0xFF (NaN en float, para
    # detectar píxeles sin calcular) y copiar la máscara. Dentro: copia CPU -> GPU, kernel
    # y copia GPU -> CPU, cada una entre dos eventos. El kernel es asíncrono; por eso se
    # espera al último evento antes de leer los tiempos.
    #
    # Los arreglos van como float32 contiguos y los enteros como np.int32, que es lo que
    # espera la firma del kernel; PyCUDA no convierte tipos por su cuenta.
    def convolucion(self, entrada, mascara, ancho, alto, k, hilos):
        entrada = np.ascontiguousarray(entrada, dtype=np.float32)
        mascara = np.ascontiguousarray(mascara, dtype=np.float32)
        salida = np.empty(ancho * alto, dtype=np.float32)
        bloques = (ancho * alto + hilos - 1) // hilos

        gpu_entrada = cuda.mem_alloc(entrada.nbytes)
        gpu_salida = cuda.mem_alloc(salida.nbytes)
        gpu_mascara = cuda.mem_alloc(mascara.nbytes)
        cuda.memset_d8(gpu_salida, 0xFF, salida.nbytes)
        cuda.memcpy_htod(gpu_mascara, mascara)

        inicio, tras_entrada, tras_kernel, fin = (cuda.Event() for _ in range(4))

        # CPU -> GPU
        inicio.record()
        cuda.memcpy_htod(gpu_entrada, entrada)
        tras_entrada.record()

        self.funcion(gpu_entrada, gpu_salida, gpu_mascara,
                     np.int32(ancho), np.int32(alto), np.int32(k),
                     block=(hilos, 1, 1), grid=(bloques, 1))
        tras_kernel.record()

        # GPU -> CPU
        cuda.memcpy_dtoh(salida, gpu_salida)
        fin.record()
        fin.synchronize()

        tiempos = TiemposGpu(
            cpu_gpu=inicio.time_till(tras_entrada),
            kernel=tras_entrada.time_till(tras_kernel),
            gpu_cpu=tras_kernel.time_till(fin),
            total=inicio.time_till(fin),
            bloques=bloques,
        )

        gpu_entrada.free()
        gpu_salida.free()
        gpu_mascara.free()
        return salida, tiempos
