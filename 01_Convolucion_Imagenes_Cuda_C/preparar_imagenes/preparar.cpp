// Genera las imágenes de entrada de 2000, 4000 y 6000 píxeles.
//
// Compilar y ejecutar desde 01_Convolucion_Imagenes_Cuda_C/:
//   cmake -S preparar_imagenes -B preparar_imagenes/build -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
//   cmake --build preparar_imagenes/build --config Release
//   .\preparar_imagenes\build\Release\preparar.exe imagenes\foto.jpg imagenes
//
// Pasos: escala de grises -> recorte cuadrado central -> reducción con INTER_AREA,
// que promedia los píxeles de origen y evita artefactos al reducir.

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Uso: preparar <imagen_origen> <carpeta_salida>\n";
        return 1;
    }

    const cv::Mat gris = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (gris.empty()) {
        std::cerr << "No se pudo leer " << argv[1] << "\n";
        return 1;
    }

    const int lado = std::min(gris.cols, gris.rows);
    if (lado < 6000) {
        std::cerr << "La imagen debe medir al menos 6000x6000; su recorte mide "
                  << lado << "x" << lado << "\n";
        return 1;
    }

    const int x = (gris.cols - lado) / 2;
    const int y = (gris.rows - lado) / 2;
    const cv::Mat recorte = gris(cv::Rect(x, y, lado, lado));
    std::cout << "Origen: " << gris.cols << "x" << gris.rows << ", recorte: "
              << lado << "x" << lado << " en (" << x << ", " << y << ")\n";

    for (const int resolucion : {2000, 4000, 6000}) {
        cv::Mat salida;
        if (resolucion == lado) {
            salida = recorte;
        } else {
            cv::resize(recorte, salida, cv::Size(resolucion, resolucion), 0, 0, cv::INTER_AREA);
        }

        const std::string ruta = std::string(argv[2]) + "/img_" + std::to_string(resolucion) + ".pgm";
        if (!cv::imwrite(ruta, salida)) {
            std::cerr << "No se pudo escribir " << ruta << "\n";
            return 1;
        }
        std::cout << "Escrita " << ruta << "\n";
    }
    return 0;
}
