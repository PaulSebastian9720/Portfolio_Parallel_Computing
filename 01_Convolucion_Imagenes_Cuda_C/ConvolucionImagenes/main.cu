// Ejecuta una prueba o un experimento e imprime una tabla separada por tabuladores.
//   ./ConvolucionImagenes manual | no_multiplo | filtro | resolucion | mascara | hilos

#include "comun.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using std::string;
using std::vector;

const string DIR_IMAGENES = string(RAIZ_PROYECTO) + "/../imagenes/";
const string DIR_SALIDAS = string(RAIZ_PROYECTO) + "/salidas/";

const int REPETICIONES = 5;

struct Configuracion {
    string filtro;
    int k;
    int resolucion;
    int hilos;
};

// Lee un PGM binario (P5) de 8 bits generado por preparar_imagenes/preparar.cpp.
vector<float> leerPGM(const string& ruta, int& ancho, int& alto) {
    std::ifstream archivo(ruta, std::ios::binary);
    string tipo;
    int maximo;
    archivo >> tipo >> ancho >> alto >> maximo;
    archivo.get();  // Separador entre la cabecera y los datos

    vector<unsigned char> bytes(static_cast<size_t>(ancho) * alto);
    archivo.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    if (!archivo || tipo != "P5") {
        printf("No se pudo leer %s. Genere las imagenes con preparar.cpp\n", ruta.c_str());
        exit(1);
    }
    return vector<float>(bytes.begin(), bytes.end());
}

// Convierte el resultado numérico a 8 bits y lo guarda. Media y nitidez se recortan a
// [0, 255]. En el laplaciano se usa el valor absoluto, porque sus respuestas negativas
// también indican cambios de intensidad. Se hace después de validar los valores float.
void guardarPGM(const string& ruta, const vector<float>& valores, int ancho, int alto, bool absoluto) {
    std::ofstream archivo(ruta, std::ios::binary);
    archivo << "P5\n" << ancho << " " << alto << "\n255\n";
    for (float valor : valores) {
        if (absoluto) valor = std::fabs(valor);
        archivo.put(static_cast<char>(std::clamp(std::lround(valor), 0L, 255L)));
    }
}

double medirCPU(const vector<float>& imagen, vector<float>& salida, int ancho, int alto,
                const vector<float>& mascara, int k) {
    const auto inicio = std::chrono::steady_clock::now();
    convolucionCPU(imagen.data(), salida.data(), ancho, alto, mascara.data(), k);
    const auto fin = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(fin - inicio).count();
}

// Mide cada configuración en CPU y GPU: una ejecución de calentamiento descartada y
// REPETICIONES medidas. Después compara ambas salidas y guarda las dos imágenes filtradas.
void ejecutarExperimento(const string& nombre, const vector<Configuracion>& configuraciones) {
    std::filesystem::create_directories(DIR_SALIDAS);

    printf("=== EXPERIMENTO %s: MEDICIONES (milisegundos) ===\n", nombre.c_str());
    printf("impl\tfiltro\tmascara\tresolucion\thilos\tbloques\trepeticion\tcpu_a_gpu\tkernel\tgpu_a_cpu\ttotal\n");

    vector<ErrorNumerico> errores;
    for (const Configuracion& c : configuraciones) {
        int ancho, alto;
        const vector<float> imagen = leerPGM(DIR_IMAGENES + "img_" + std::to_string(c.resolucion) + ".pgm", ancho, alto);
        const vector<float> mascara = crearMascara(c.filtro, c.k);
        vector<float> cpu(imagen.size());
        vector<float> gpu(imagen.size());

        medirCPU(imagen, cpu, ancho, alto, mascara, c.k);
        for (int r = 1; r <= REPETICIONES; r++) {
            const double total = medirCPU(imagen, cpu, ancho, alto, mascara, c.k);
            printf("cpu\t%s\t%d\t%d\t\t\t%d\t\t\t\t%.4f\n", c.filtro.c_str(), c.k, c.resolucion, r, total);
        }
        fflush(stdout);

        convolucionGPU(imagen.data(), gpu.data(), ancho, alto, mascara.data(), c.k, c.hilos);
        for (int r = 1; r <= REPETICIONES; r++) {
            const TiemposGpu t = convolucionGPU(imagen.data(), gpu.data(), ancho, alto, mascara.data(), c.k, c.hilos);
            printf("gpu\t%s\t%d\t%d\t%d\t%d\t%d\t%.4f\t%.4f\t%.4f\t%.4f\n", c.filtro.c_str(), c.k, c.resolucion,
                   c.hilos, t.bloques, r, t.cpuGpu, t.kernel, t.gpuCpu, t.total);
        }
        fflush(stdout);

        errores.push_back(comparar(cpu, gpu));
        const string base = c.filtro + "_k" + std::to_string(c.k) + "_" + std::to_string(c.resolucion);
        guardarPGM(DIR_SALIDAS + base + ".pgm", cpu, ancho, alto, c.filtro == "laplaciano");
        guardarPGM(DIR_SALIDAS + base + "_gpu.pgm", gpu, ancho, alto, c.filtro == "laplaciano");
    }

    printf("\n=== EXPERIMENTO %s: VALIDACION CPU vs GPU (tolerancia %.2f) ===\n", nombre.c_str(), TOLERANCIA);
    printf("filtro\tmascara\tresolucion\thilos\terror_maximo\terror_medio\tpixeles_fuera\testado\n");
    for (size_t i = 0; i < configuraciones.size(); i++) {
        const Configuracion& c = configuraciones[i];
        printf("%s\t%d\t%d\t%d\t%.6f\t%.6f\t%ld\t%s\n", c.filtro.c_str(), c.k, c.resolucion, c.hilos,
               errores[i].maximo, errores[i].medio, errores[i].fueraDeTolerancia,
               errores[i].fueraDeTolerancia == 0 ? "OK" : "FALLO");
    }
}

int main(int argc, char** argv) {
    const string opcion = argc == 2 ? argv[1] : "";

    if (opcion == "manual") {
        pruebaManual();
    } else if (opcion == "no_multiplo") {
        pruebaNoMultiplo();
    } else if (opcion == "filtro") {
        ejecutarExperimento("FILTRO", {
            {"media", 3, 4000, 256},
            {"nitidez", 3, 4000, 256},
            {"laplaciano", 3, 4000, 256},
        });
    } else if (opcion == "resolucion") {
        ejecutarExperimento("RESOLUCION", {
            {"media", 9, 2000, 256},
            {"media", 9, 4000, 256},
            {"media", 9, 6000, 256},
        });
    } else if (opcion == "mascara") {
        ejecutarExperimento("MASCARA", {
            {"media", 3, 4000, 256},
            {"media", 9, 4000, 256},
            {"media", 21, 4000, 256},
        });
    } else if (opcion == "hilos") {
        ejecutarExperimento("HILOS POR BLOQUE", {
            {"media", 9, 4000, 128},
            {"media", 9, 4000, 256},
            {"media", 9, 4000, 512},
        });
    } else {
        printf("Uso: %s manual | no_multiplo | filtro | resolucion | mascara | hilos\n", argv[0]);
        return 1;
    }
    return 0;
}
