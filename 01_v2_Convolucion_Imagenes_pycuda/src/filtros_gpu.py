"""Convolución en GPU con PyCUDA (equivale a convolucionGPU de filtros_gpu.cu).

El programa anfitrión está en Python y el kernel sigue en CUDA C, escrito como cadena
(KERNEL_CONVOLUCION) y compilado con SourceModule.
Correspondencia con la API de CUDA C usada en la práctica 01:

    cudaMalloc               -> cuda.mem_alloc(bytes)
    cudaMemset               -> cuda.memset_d8(puntero, valor, bytes)
    cudaMemcpy HostToDevice  -> cuda.memcpy_htod(destino_gpu, arreglo_numpy)
    cudaMemcpy DeviceToHost  -> cuda.memcpy_dtoh(arreglo_numpy, origen_gpu)
    cudaFree                 -> puntero.free()
    cudaEvent_t / Record     -> cuda.Event() / evento.record()
    cudaEventSynchronize     -> evento.synchronize()
    cudaEventElapsedTime     -> inicio.time_till(fin)   (milisegundos)
    kernel<<<b, h>>>(...)    -> funcion(..., block=(h, 1, 1), grid=(b, 1))

PyCUDA también ofrece cuda.In / cuda.Out / cuda.InOut, que copian los arreglos
automáticamente al lanzar el kernel. No se usan aquí porque las copias quedarían dentro
de la misma llamada que el kernel, y la práctica exige medir por separado la
transferencia CPU->GPU, el kernel y la transferencia GPU->CPU.
"""

import time

import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from comun import TiemposGpu

# Kernel de la práctica 01 (filtros_gpu.cu), sin cambios. SourceModule envuelve el código
# en extern "C" para que el nombre "kernelConvolucion" no se altere y se pueda obtener
# con get_function().
#
# Tipos: float (32 bits) = np.float32 en Python; int = np.int32.
#
# Cada hilo calcula un píxel de salida. Bloques e hilos se distribuyen solo en X:
#   indice = blockIdx.x * blockDim.x + threadIdx.x   (posición en el arreglo plano)
#   fila = indice / ancho,  col = indice % ancho
# Si los píxeles no son múltiplo de los hilos por bloque, el último bloque tiene hilos
# sobrantes: su índice queda fuera de la imagen y terminan sin escribir.
# Extensión con ceros: un vecino fuera de la imagen aportaría 0, así que se omite.
#
# Parámetros del kernel:
#   entrada      imagen original en memoria global (ancho*alto floats, fila tras fila)
#   salida       arreglo separado donde se escribe el resultado (ancho*alto floats)
#   mascara      k*k coeficientes, fila tras fila
#   ancho, alto  dimensiones de la imagen
#   k            lado de la máscara (impar)
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


# Contexto CUDA y kernel compilado, preparados una sola vez antes de las mediciones.
#
# No se usa pycuda.autoinit para poder medir por separado la inicialización del contexto
# y la compilación/carga del kernel (la guía pide registrar ambos y no repetirlos dentro
# de cada medición).
#
# Atributos:
#   dispositivo         cuda.Device usado (GPU 0)
#   ms_contexto         tiempo de cuda.init() + creación del contexto, en ms
#   ms_compilacion      tiempo de compilar y cargar el kernel con SourceModule, en ms
#   funcion             kernel "kernelConvolucion" listo para lanzar
class GpuConvolucion:

    # Inicializa el driver, crea el contexto en la GPU indicada y compila el kernel.
    # Si la compilación falla (por ejemplo, nvcc no soporta la arquitectura de la GPU),
    # se libera el contexto antes de propagar el error.
    # Parámetros:
    #   numero_gpu  índice del dispositivo (0 por defecto)
    def __init__(self, numero_gpu=0):
        inicio = time.perf_counter()
        cuda.init()
        self.dispositivo = cuda.Device(numero_gpu)
        self.contexto = self.dispositivo.make_context()
        self.ms_contexto = (time.perf_counter() - inicio) * 1000

        try:
            inicio = time.perf_counter()
            modulo = SourceModule(KERNEL_CONVOLUCION)
            self.funcion = modulo.get_function("kernelConvolucion")
            self.ms_compilacion = (time.perf_counter() - inicio) * 1000
        except Exception:
            self.cerrar()
            raise

    # Libera el contexto CUDA. Debe llamarse al terminar (main.py lo hace en un finally).
    def cerrar(self):
        self.contexto.pop()
        self.contexto.detach()

    # Ejecuta la convolución en la GPU y mide cada etapa con eventos CUDA.
    #
    # La reserva de memoria y la copia de la máscara se hacen antes de medir, igual que
    # en la práctica 01. La salida se llena con bytes 0xFF, que como float son NaN: si
    # algún hilo no escribe su píxel, la comparación lo detecta.
    #
    # Los eventos marcan el momento en que la GPU terminó todo lo anterior. El kernel es
    # asíncrono: Python no lo espera, pero el evento siguiente solo se completa cuando
    # acaba. fin.synchronize() espera al último evento antes de leer los tiempos.
    #
    # Parámetros:
    #   entrada  arreglo plano de ancho*alto; se convierte a np.float32 contiguo
    #   mascara  arreglo de k*k coeficientes; se convierte a np.float32 contiguo
    #   ancho, alto  dimensiones de la imagen
    #   k        lado de la máscara
    #   hilos    hilos por bloque (dimensión X)
    # Retorna: (salida np.float32 de ancho*alto, TiemposGpu).
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

        inicio.record()
        cuda.memcpy_htod(gpu_entrada, entrada)
        tras_entrada.record()

        self.funcion(gpu_entrada, gpu_salida, gpu_mascara,
                     np.int32(ancho), np.int32(alto), np.int32(k),
                     block=(hilos, 1, 1), grid=(bloques, 1))
        tras_kernel.record()

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
