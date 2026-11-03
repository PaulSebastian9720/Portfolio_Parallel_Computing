cmake_minimum_required(VERSION 4.2)

# Arquitectura de la GPU donde se ejecuta y mide (RTX 5050). Debe definirse antes
# de project(). Para otra GPU: cmake -DCMAKE_CUDA_ARCHITECTURES=<valor> ...
set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "Arquitectura CUDA de destino")

project(ConvolucionImagenes CUDA CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# El proyecto no usa módulos C++20; su escaneo falla con espacios en la ruta.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

add_executable(ConvolucionImagenes
        main.cu
        filtros_gpu.cu
        filtros_cpu.cpp
        pruebas.cpp)

# Las rutas de imagenes/ y salidas/ se resuelven desde la carpeta del proyecto.
target_compile_definitions(ConvolucionImagenes PRIVATE
        RAIZ_PROYECTO="${CMAKE_CURRENT_SOURCE_DIR}")
