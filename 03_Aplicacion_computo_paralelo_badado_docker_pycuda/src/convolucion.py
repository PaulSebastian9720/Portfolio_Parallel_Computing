"""Convolución con PyCUDA de la práctica 02, sin nada de HTTP.

    python src/convolucion.py <imagen> <filtro> <k> <hilos> [salida.npy]
"""

import sys
import time
from dataclasses import asdict, dataclass

import numpy as np
import pycuda
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# Error admitido entre la ejecución directa y el servicio, en la escala 0-255. Es el mismo
# de la práctica 02: el peor caso (media 21x21) acumula 441 * 255 * 1.19e-7 = 0.013.
TOLERANCIA = 0.02

# Mismo kernel de la práctica 02. Cada hilo calcula un píxel. Con bloques e hilos solo en
# X, blockIdx.x * blockDim.x + threadIdx.x recorre la imagen como un arreglo plano; de ahí
# salen la fila (indice / ancho) y la columna (indice % ancho). Los hilos sobrantes del
# último bloque salen sin escribir. Un vecino fuera de la imagen vale 0 y se omite.
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


# Tiempos de una ejecución en GPU, en ms; total abarca las tres etapas.
@dataclass
class TiemposGpu:
    cpu_gpu: float
    kernel: float
    gpu_cpu: float
    total: float
    bloques: int


# Coeficientes fila tras fila, en float32 como el float de C. k solo se usa en la media;
# nitidez y laplaciano son siempre 3x3.
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


# Prepara la GPU una sola vez: contexto y kernel compilado. El contexto queda activo en el
# hilo que crea el objeto, así que todas las llamadas deben hacerse desde ese mismo hilo.
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

    def memoria(self):
        libre, total = cuda.mem_get_info()
        return libre, total

    def info(self):
        libre, total = self.memoria()
        driver = cuda.get_driver_version()
        compilado = cuda.get_version()
        return {
            "nombre": self.dispositivo.name(),
            "capacidad": "%d.%d" % self.dispositivo.compute_capability(),
            "memoria_total_mb": round(total / 2**20),
            "memoria_libre_mb": round(libre / 2**20),
            "driver_cuda": f"{driver // 1000}.{driver % 1000 // 10}",
            "cuda_pycuda": ".".join(map(str, compilado)),
            "pycuda": pycuda.VERSION_TEXT,
            "ms_contexto": self.ms_contexto,
            "ms_compilacion": self.ms_compilacion,
        }

    # Convolución en la GPU con las tres etapas medidas por separado.
    #
    # Fuera de la medición: reservar memoria, llenar la salida con 0xFF (NaN en float, para
    # detectar píxeles sin calcular) y copiar la máscara. Dentro: copia CPU -> GPU, kernel
    # y copia GPU -> CPU, cada una entre dos eventos. La memoria del anfitrión es paginable.
    # El kernel es asíncrono; por eso se espera al último evento antes de leer los tiempos.
    #
    # Los arreglos van como float32 contiguos y los enteros como np.int32, que es lo que
    # espera la firma del kernel; PyCUDA no convierte tipos por su cuenta.
    def convolucion(self, entrada, mascara, ancho, alto, k, hilos):
        entrada = np.ascontiguousarray(entrada, dtype=np.float32)
        mascara = np.ascontiguousarray(mascara, dtype=np.float32)
        salida = np.empty(ancho * alto, dtype=np.float32)
        bloques = (ancho * alto + hilos - 1) // hilos

        gpu_entrada = gpu_salida = gpu_mascara = None
        try:
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
        finally:
            # Se libera también cuando falla una reserva o el kernel.
            for memoria in (gpu_entrada, gpu_salida, gpu_mascara):
                if memoria is not None:
                    memoria.free()
        return salida, tiempos


# Lee un PGM de 8 bits como arreglo plano float32, igual que en la práctica 02.
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
        raise ValueError(f"{ruta} no es un PGM binario (P5)")

    ancho, alto = int(cabecera[1]), int(cabecera[2])
    bytes_imagen = np.frombuffer(datos, dtype=np.uint8, count=ancho * alto)
    return bytes_imagen.astype(np.float32), ancho, alto


# Error absoluto máximo y medio entre dos salidas float. Un NaN es un píxel que ningún hilo
# escribió: cuenta como fuera de tolerancia.
def comparar(referencia, otra):
    diferencia = np.abs(referencia.astype(np.float64) - otra.astype(np.float64))
    validos = ~np.isnan(diferencia)
    return {
        "maximo": float(diferencia[validos].max()) if np.any(validos) else 0.0,
        "medio": float(diferencia[validos].sum() / diferencia.size),
        "fuera_de_tolerancia": int(np.count_nonzero(~(diferencia <= TOLERANCIA))),
    }


# Ejecución directa, sin servicio: sirve de referencia para comparar con lo que devuelve la
# API. Si se da una ruta .npy, guarda la salida float sin recortar.
def main(argv):
    if len(argv) not in (5, 6):
        print(__doc__.strip(), file=sys.stderr)
        return 1

    ruta, filtro, k, hilos = argv[1], argv[2], int(argv[3]), int(argv[4])
    entrada, ancho, alto = leer_pgm(ruta)
    mascara = crear_mascara(filtro, k)

    gpu = GpuConvolucion()
    try:
        print(gpu.info())
        salida, tiempos = gpu.convolucion(entrada, mascara, ancho, alto, k, hilos)
    finally:
        gpu.cerrar()

    print(f"{ancho}x{alto} {filtro} k={k} hilos={hilos}", asdict(tiempos))
    print(f"min={np.nanmin(salida):.4f} max={np.nanmax(salida):.4f} "
          f"sin calcular={int(np.isnan(salida).sum())}")
    if len(argv) == 6:
        np.save(argv[5], salida)
        print(f"Guardada {argv[5]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
