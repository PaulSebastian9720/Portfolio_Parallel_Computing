#include "comun.h"

// Devuelve los k*k coeficientes de la máscara, fila tras fila.
std::vector<float> crearMascara(const std::string& filtro, int k) {
    // Filtro de nitidez (3x3): resalta los detalles
    if (filtro == "nitidez") {
        return {0, -1, 0,
                -1, 5, -1,
                0, -1, 0};
    }

    // Filtro laplaciano (3x3): resalta los cambios de intensidad
    if (filtro == "laplaciano") {
        return {0, 1, 0,
                1, -4, 1,
                0, 1, 0};
    }

    // Filtro de media (k x k): suaviza la imagen; todos los coeficientes valen 1/(k*k)
    return std::vector<float>(k * k, 1.0f / (k * k));
}

// Convolución secuencial: referencia de corrección y de aceleración.
// Extensión con ceros: un vecino fuera de la imagen aportaría 0, así que se omite.
void convolucionCPU(const float* entrada, float* salida, int ancho, int alto,
                    const float* mascara, int k) {
    const int radio = k / 2;

    for (int fila = 0; fila < alto; fila++) {
        for (int col = 0; col < ancho; col++) {
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

            salida[fila * ancho + col] = suma;
        }
    }
}
