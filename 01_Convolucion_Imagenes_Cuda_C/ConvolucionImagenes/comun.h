#pragma once

#include <string>
#include <vector>

// Error máximo admitido entre CPU y GPU, en escala 0-255. Cota del peor caso
// (media 21x21): 441 * 255 * 1.19e-7 = 0.013; y 0.02 es muy inferior a un nivel de gris.
const double TOLERANCIA = 0.02;

// Tiempos de una ejecución en GPU, en milisegundos.
struct TiemposGpu {
    float cpuGpu;
    float kernel;
    float gpuCpu;
    float total;
    int bloques;
};

struct ErrorNumerico {
    double maximo = 0;
    double medio = 0;
    long fueraDeTolerancia = 0;
};

// filtros_cpu.cpp
std::vector<float> crearMascara(const std::string& filtro, int k);
void convolucionCPU(const float* entrada, float* salida, int ancho, int alto,
                    const float* mascara, int k);

// filtros_gpu.cu
TiemposGpu convolucionGPU(const float* entrada, float* salida, int ancho, int alto,
                          const float* mascara, int k, int hilosPorBloque);

// pruebas.cpp
ErrorNumerico comparar(const std::vector<float>& cpu, const std::vector<float>& gpu);
void pruebaManual();
void pruebaNoMultiplo();
