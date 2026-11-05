// Convierte una imagen a un PNG en escala de grises y de tamaño reducido.
//
// Ejecutar desde 01_Convolucion_Imagenes_Cuda_C/:
//   .\preparar_imagenes\build\Release\a_png.exe imagenes\img_6000.pgm Informe_LaTeX\img\entrada_6000.png 1000
//
// Con tres valores más recorta antes una misma zona de la escena, sin importar la resolución.
// Son fracciones de 0 a 1: posición en X, posición en Y y lado del recorte.
//   .\preparar_imagenes\build\Release\a_png.exe imagenes\img_6000.pgm Informe_LaTeX\img\recorte_6000.png 1000 0.55 0.65 0.0833

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc != 4 && argc != 7) {
        std::cerr << "Uso: a_png <entrada> <salida.png> <lado_maximo> [x y lado]\n";
        return 1;
    }

    cv::Mat entrada = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (entrada.empty()) {
        std::cerr << "No se pudo leer " << argv[1] << "\n";
        return 1;
    }

    if (argc == 7) {
        const int x = static_cast<int>(std::stod(argv[4]) * entrada.cols);
        const int y = static_cast<int>(std::stod(argv[5]) * entrada.rows);
        const int lado = static_cast<int>(std::stod(argv[6]) * entrada.cols);
        const cv::Rect zona(x, y, lado, lado);
        if ((zona & cv::Rect(0, 0, entrada.cols, entrada.rows)) != zona) {
            std::cerr << "El recorte queda fuera de la imagen\n";
            return 1;
        }
        entrada = entrada(zona);
    }

    const int ladoMaximo = std::stoi(argv[3]);
    const double escala = std::min(1.0, static_cast<double>(ladoMaximo) / std::max(entrada.cols, entrada.rows));

    cv::Mat salida;
    cv::resize(entrada, salida, cv::Size(), escala, escala, cv::INTER_AREA);

    if (!cv::imwrite(argv[2], salida)) {
        std::cerr << "No se pudo escribir " << argv[2] << "\n";
        return 1;
    }
    std::cout << argv[1] << " (" << entrada.cols << "x" << entrada.rows << ") -> "
              << argv[2] << " (" << salida.cols << "x" << salida.rows << ")\n";
    return 0;
}
