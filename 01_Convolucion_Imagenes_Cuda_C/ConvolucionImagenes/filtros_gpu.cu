#include "comun.h"

#include <cstdio>
#include <cstdlib>

// Termina el programa con un mensaje claro si una llamada a CUDA falla.
static void revisar(cudaError_t error, const char* operacion) {
    if (error != cudaSuccess) {
        printf("Error de CUDA en %s: %s\n", operacion, cudaGetErrorString(error));
        exit(1);
    }
}

// Cada hilo calcula un píxel de salida. Bloques e hilos se distribuyen solo en X:
//   indice = blockIdx.x * blockDim.x + threadIdx.x   (posición en el arreglo plano)
//   fila = indice / ancho,  col = indice % ancho
// Si los píxeles no son múltiplo de los hilos por bloque, el último bloque tiene hilos
// sobrantes: su índice queda fuera de la imagen y terminan sin escribir.
// Extensión con ceros: un vecino fuera de la imagen aportaría 0, así que se omite.
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

// Ejecuta la convolución en la GPU y mide cada etapa con eventos CUDA.
//
// La reserva de memoria y la copia de la máscara se hacen antes de medir.
// La salida se llena con bytes 0xFF, que como float son NaN: si algún hilo no escribe
// su píxel, la comparación con la CPU lo detecta.
//
// Los eventos marcan el momento en que la GPU terminó todo lo anterior. El kernel es
// asíncrono: el programa no lo espera, pero el evento siguiente solo se completa cuando
// acaba. cudaEventSynchronize espera al último evento antes de leer los tiempos.
TiemposGpu convolucionGPU(const float* entrada, float* salida, int ancho, int alto,
                          const float* mascara, int k, int hilosPorBloque) {
    const size_t bytesImagen = static_cast<size_t>(ancho) * alto * sizeof(float);
    const size_t bytesMascara = k * k * sizeof(float);
    const int bloques = (ancho * alto + hilosPorBloque - 1) / hilosPorBloque;

    float* gpuEntrada;
    float* gpuSalida;
    float* gpuMascara;
    revisar(cudaMalloc(&gpuEntrada, bytesImagen), "reservar entrada");
    revisar(cudaMalloc(&gpuSalida, bytesImagen), "reservar salida");
    revisar(cudaMalloc(&gpuMascara, bytesMascara), "reservar mascara");
    revisar(cudaMemset(gpuSalida, 0xFF, bytesImagen), "inicializar salida");
    revisar(cudaMemcpy(gpuMascara, mascara, bytesMascara, cudaMemcpyHostToDevice), "copiar mascara");

    cudaEvent_t inicio, trasEntrada, trasKernel, fin;
    cudaEventCreate(&inicio);
    cudaEventCreate(&trasEntrada);
    cudaEventCreate(&trasKernel);
    cudaEventCreate(&fin);

    cudaEventRecord(inicio);
    revisar(cudaMemcpy(gpuEntrada, entrada, bytesImagen, cudaMemcpyHostToDevice), "copiar CPU a GPU");
    cudaEventRecord(trasEntrada);

    kernelConvolucion<<<bloques, hilosPorBloque>>>(gpuEntrada, gpuSalida, gpuMascara, ancho, alto, k);
    revisar(cudaGetLastError(), "lanzar kernel");
    cudaEventRecord(trasKernel);

    revisar(cudaMemcpy(salida, gpuSalida, bytesImagen, cudaMemcpyDeviceToHost), "copiar GPU a CPU");
    cudaEventRecord(fin);
    revisar(cudaEventSynchronize(fin), "sincronizar");

    TiemposGpu t;
    t.bloques = bloques;
    cudaEventElapsedTime(&t.cpuGpu, inicio, trasEntrada);
    cudaEventElapsedTime(&t.kernel, trasEntrada, trasKernel);
    cudaEventElapsedTime(&t.gpuCpu, trasKernel, fin);
    cudaEventElapsedTime(&t.total, inicio, fin);

    cudaEventDestroy(inicio);
    cudaEventDestroy(trasEntrada);
    cudaEventDestroy(trasKernel);
    cudaEventDestroy(fin);
    cudaFree(gpuEntrada);
    cudaFree(gpuSalida);
    cudaFree(gpuMascara);

    return t;
}
