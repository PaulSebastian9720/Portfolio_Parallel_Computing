# Convolución de imágenes con CUDA C

Filtros de media, nitidez y laplaciano sobre imágenes de alta resolución, con una versión secuencial en C++ y otra en CUDA C. El programa comprueba que ambas den el mismo resultado y mide cuánto tarda cada una.

Autores: Paul Sebastian Naspud Vivar y Jennyfer Camila Ramirez Saeteros.

## Carpetas

| Carpeta | Contenido |
|---|---|
| `ConvolucionImagenes/` | Filtros en CPU y GPU, pruebas y experimentos |
| `preparar_imagenes/` | `preparar` genera las entradas y `a_png` convierte un PGM a PNG |
| `imagenes/` | `FUENTES.txt` indica de dónde descargar las fotos, que no se suben al repositorio |

## Dependencias

Se probó en Windows 11 con:

- NVIDIA CUDA Toolkit 13.5 y controlador 621.14
- Visual Studio Community 2022 con MSVC 14.44
- CMake 4.4.4
- OpenCV 4.14.0 instalado con vcpkg en `C:\vcpkg`, necesario solo en `preparar_imagenes/`

El proyecto compila para `sm_120`, la arquitectura de la RTX 5050. En otra GPU hay que agregar `-DCMAKE_CUDA_ARCHITECTURES=<valor>` al configurar.

Los comandos se ejecutan desde `01_Convolucion_Imagenes_Cuda_C/`.

## Preparar las imágenes

Descarga la fotografía 1 de [`imagenes/FUENTES.txt`](imagenes/FUENTES.txt) y guárdala como `imagenes/foto.jpg`. Con esto salen las versiones de 2000, 4000 y 6000 píxeles:

```powershell
cmake -S preparar_imagenes -B preparar_imagenes/build -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build preparar_imagenes/build --config Release
.\preparar_imagenes\build\Release\preparar.exe imagenes\foto.jpg imagenes
```

## Compilar

```powershell
cmake -S ConvolucionImagenes -B ConvolucionImagenes/build
cmake --build ConvolucionImagenes/build --config Release
```

Los tiempos solo valen en Release.

## Pruebas

```powershell
.\ConvolucionImagenes\build\Release\ConvolucionImagenes.exe manual
.\ConvolucionImagenes\build\Release\ConvolucionImagenes.exe no_multiplo
```

`manual` contrasta 10 valores calculados a mano sobre una matriz de 5×5, en CPU y en GPU. `no_multiplo` compara ambas versiones con una imagen de 1003×997 y 128, 256 y 512 hilos por bloque.

## Experimentos

```powershell
.\ConvolucionImagenes\build\Release\ConvolucionImagenes.exe filtro
.\ConvolucionImagenes\build\Release\ConvolucionImagenes.exe resolucion
.\ConvolucionImagenes\build\Release\ConvolucionImagenes.exe mascara
.\ConvolucionImagenes\build\Release\ConvolucionImagenes.exe hilos
```

| Experimento | Filtro | Imagen | Hilos por bloque |
|---|---|---|---|
| `filtro` | Media, nitidez y laplaciano de 3×3 | 4000×4000 | 256 |
| `resolucion` | Media de 9×9 | 2000, 4000 y 6000 | 256 |
| `mascara` | Media de 3×3, 9×9 y 21×21 | 4000×4000 | 256 |
| `hilos` | Media de 9×9 | 4000×4000 | 128, 256 y 512 |

Cada configuración hace un calentamiento y 5 repeticiones. Los tiempos, en milisegundos, y la comparación entre CPU y GPU se imprimen en la consola. Las imágenes filtradas quedan en `ConvolucionImagenes/salidas/`.
