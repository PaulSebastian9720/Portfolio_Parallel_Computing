# Convolución de imágenes con PyCUDA

Práctica 02 de Computación Paralela. Los filtros de media, nitidez y laplaciano de la práctica 01 pasan a un programa anfitrión en Python: Python secuencial con bucles explícitos y PyCUDA con el mismo kernel en CUDA C. Las cuatro versiones (C/C++, CUDA C, Python y PyCUDA) se validan contra C/C++ y se comparan en tiempo.

Autores: Paul Sebastian Naspud Vivar y Jennyfer Camila Ramirez Saeteros.

La práctica 01 se conserva sin cambios en [`../01_Convolucion_Imagenes_Cuda_C`](../01_Convolucion_Imagenes_Cuda_C) (commit `015ce85`).

## Carpetas

| Carpeta | Contenido |
|---|---|
| `src/` | Programa Python: filtros secuencial y PyCUDA, pruebas, experimentos, preparación de imágenes y conversión a PNG |
| `resultados/` | `resultados.xlsx` con todas las mediciones y validaciones, una hoja por prueba o experimento |

## Requisitos

Probado en Windows 11 con:

- Python 3.12.10 (sirve 3.12 o posterior) y [uv](https://docs.astral.sh/uv/) 0.13.8
- NVIDIA CUDA Toolkit 13.5, controlador 621.14, GPU RTX 5050 (`sm_120`)
- Visual Studio Community 2022 (MSVC 14.44) y CMake 4.4.4

Dependencias de Python, fijadas en `pyproject.toml` y `uv.lock`: numpy 2.5.5, pycuda 2026.1, opencv-python 5.0.0.93 y matplotlib 3.11.2.

PyCUDA se compila al instalarse, y `SourceModule` llama a `nvcc`, que en Windows necesita `cl.exe`. Por eso todos los comandos se ejecutan en una consola con el entorno de Visual Studio:

```bat
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5
```

## Instalación

Desde `01_v2_Convolucion_Imagenes_pycuda/`:

```bat
pip install uv
uv sync
```

La validación contra C/C++ usa el ejecutable `puente` (carpeta `referencia_c/`, que enlaza sin cambios `filtros_cpu.cpp` y `filtros_gpu.cu` de la práctica 01). Esa carpeta no se incluye en el repositorio.

## Imágenes

Se usa la misma fotografía de la práctica 01, la 1 de [`FUENTES.txt`](../01_Convolucion_Imagenes_Cuda_C/imagenes/FUENTES.txt). Guárdala como `imagenes/foto.jpg` y genera las entradas:

```bat
uv run src/preparar_imagenes.py imagenes/foto.jpg imagenes
```

Salen `img_500.pgm`, `img_2000.pgm`, `img_4000.pgm` e `img_6000.pgm`. Las tres grandes son idénticas byte a byte a las de la práctica 01.

## Pruebas y experimentos

```bat
uv run src/main.py entorno
uv run src/main.py manual
uv run src/main.py no_multiplo
uv run src/main.py comparacion
uv run src/main.py filtro
uv run src/main.py resolucion
uv run src/main.py mascara
uv run src/main.py hilos
```

| Opción | Configuración | Versiones medidas |
|---|---|---|
| `comparacion` | Media 3, 9 y 21, nitidez y laplaciano en 500×500 | Las cuatro, completas |
| `filtro` | Media, nitidez y laplaciano 3×3 en 4000×4000 | Python estimado y PyCUDA |
| `resolucion` | Media 9×9 en 2000, 4000 y 6000 | Python estimado y PyCUDA |
| `mascara` | Media 3, 9 y 21 en 4000×4000 | Python estimado y PyCUDA |
| `hilos` | Media 9×9 en 4000×4000, 128, 256 y 512 hilos | PyCUDA |

Cada versión hace un calentamiento y 5 repeticiones. En alta resolución los tiempos de C/C++ y CUDA C se reutilizan de la práctica 01, y `puente` solo aporta la salida de referencia. Python secuencial tardaría de minutos a horas en esas imágenes, así que se mide un bloque de filas centrales de unos 5 s y se extrapola a la imagen completa; esas filas quedan con `estado=estimado` en el CSV. `--segundos-muestra` cambia la duración de la muestra.

Las imágenes filtradas de C/C++ y PyCUDA quedan en `salidas/`. Para pasarlas a PNG:

```bat
uv run src/a_png.py salidas/media_k3_4000_pycuda.pgm salida.png 1000
uv run src/a_png.py salidas/media_k3_4000_pycuda.pgm recorte.png 1000 0.55 0.65 0.0833
```
