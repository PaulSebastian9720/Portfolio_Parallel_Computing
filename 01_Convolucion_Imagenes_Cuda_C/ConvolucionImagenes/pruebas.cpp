#include "comun.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

using std::string;
using std::vector;

// Compara las salidas float de CPU y GPU, antes de convertirlas a 8 bits.
// La condición negada también cuenta los NaN, que son píxeles que ningún hilo calculó.
ErrorNumerico comparar(const vector<float>& cpu, const vector<float>& gpu) {
    ErrorNumerico error;
    double suma = 0;
    for (size_t i = 0; i < cpu.size(); i++) {
        const double diferencia = std::fabs(static_cast<double>(cpu[i]) - gpu[i]);
        if (!(diferencia <= TOLERANCIA)) error.fueraDeTolerancia++;
        if (!std::isnan(diferencia)) {
            error.maximo = std::max(error.maximo, diferencia);
            suma += diferencia;
        }
    }
    error.medio = suma / cpu.size();
    return error;
}

// Comprueba valores calculados a mano sobre la matriz de la Figura 1 de la guía.
// En los bordes, los vecinos fuera de la imagen valen 0.
// Ejemplo, nitidez en (0,0): 5*35 - 40 (derecha) - 40 (abajo) = 95.
static void ejecutarCasosManuales(bool usarGpu) {
    struct Caso { string nombre; vector<float> mascara; int fila; int col; float esperado; };

    const vector<float> matriz = {
        35, 40, 41, 45, 50,
        40, 40, 42, 46, 52,
        42, 46, 50, 55, 55,
        48, 52, 56, 58, 60,
        56, 60, 65, 70, 75,
    };
    const vector<float> figura = {-2, -1, 0, -1, 1, 1, 0, 1, 2};
    const vector<float> nitidez = crearMascara("nitidez", 3);
    const vector<float> laplaciano = crearMascara("laplaciano", 3);
    const vector<float> media = crearMascara("media", 3);

    const vector<Caso> casos = {
        {"figura", figura, 1, 1, 78},
        {"figura", figura, 1, 2, 87},
        {"nitidez", nitidez, 0, 0, 95},
        {"nitidez", nitidez, 0, 4, 153},
        {"nitidez", nitidez, 1, 1, 32},
        {"laplaciano", laplaciano, 0, 0, -60},
        {"laplaciano", laplaciano, 1, 1, 8},
        {"laplaciano", laplaciano, 4, 4, -170},
        {"media3", media, 0, 0, 155.0f / 9},
        {"media3", media, 1, 1, 376.0f / 9},
    };

    for (const Caso& caso : casos) {
        vector<float> salida(matriz.size());
        if (usarGpu) {
            convolucionGPU(matriz.data(), salida.data(), 5, 5, caso.mascara.data(), 3, 128);
        } else {
            convolucionCPU(matriz.data(), salida.data(), 5, 5, caso.mascara.data(), 3);
        }
        const float obtenido = salida[caso.fila * 5 + caso.col];
        const bool correcto = std::fabs(obtenido - caso.esperado) < 1e-4f;
        printf("%s\t%s\t%d\t%d\t%.4f\t%.4f\t%s\n", usarGpu ? "gpu" : "cpu", caso.nombre.c_str(),
               caso.fila, caso.col, caso.esperado, obtenido, correcto ? "OK" : "FALLO");
    }
}

void pruebaManual() {
    printf("=== PRUEBA MANUAL ===\n");
    printf("impl\tprueba\tfila\tcol\tesperado\tobtenido\testado\n");
    ejecutarCasosManuales(false);
    fflush(stdout);
    ejecutarCasosManuales(true);
}

// Imagen de 1003x997: sus píxeles no son múltiplo de los hilos por bloque.
void pruebaNoMultiplo() {
    printf("=== TAMANO NO MULTIPLO ===\n");
    printf("imagen\thilos\tbloques\terror_maximo\terror_medio\tpixeles_fuera\testado\n");

    const int ancho = 1003, alto = 997;
    vector<float> imagen(ancho * alto);
    for (size_t i = 0; i < imagen.size(); i++) {
        imagen[i] = (i * 37) % 256;
    }

    const vector<float> mascara = crearMascara("media", 3);
    vector<float> cpu(imagen.size());
    convolucionCPU(imagen.data(), cpu.data(), ancho, alto, mascara.data(), 3);

    for (int hilos : {128, 256, 512}) {
        vector<float> gpu(imagen.size());
        const TiemposGpu t = convolucionGPU(imagen.data(), gpu.data(), ancho, alto, mascara.data(), 3, hilos);
        const ErrorNumerico error = comparar(cpu, gpu);
        printf("%dx%d\t%d\t%d\t%.6f\t%.6f\t%ld\t%s\n", ancho, alto, hilos, t.bloques, error.maximo,
               error.medio, error.fueraDeTolerancia, error.fueraDeTolerancia == 0 ? "OK" : "FALLO");
    }
}
