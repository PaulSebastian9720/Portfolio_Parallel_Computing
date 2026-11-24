# Servicio web de convolución con PyCUDA en Docker

Práctica 03 de Computación Paralela. Los kernels de convolución de la práctica 02 (media 3×3, 9×9 y 21×21, nitidez y laplaciano 3×3) pasan a un servicio HTTP en Python que corre dentro de un contenedor con acceso a la GPU. Una página web sube la imagen, elige el filtro y muestra el resultado con sus tiempos.

Autores: Paul Sebastian Naspud Vivar y Jennyfer Camila Ramirez Saeteros.

La práctica 02 está en [`../01_v2_Convolucion_Imagenes_pycuda`](../01_v2_Convolucion_Imagenes_pycuda) (commit `33832b4`).

## Estructura

| Archivo | Contenido |
|---|---|
| `src/convolucion.py` | Kernel CUDA, máscaras y clase `GpuConvolucion` (contexto, compilación, memoria, eventos). Sin HTTP; se puede ejecutar solo |
| `src/servicio.py` | Validación de entradas, hilo único de GPU con límite de cola, paso a 8 bits y archivos temporales |
| `src/main.py` | API REST (FastAPI, puerto 8000) e interfaz estática (puerto 3000) en un mismo proceso |
| `src/public/` | Interfaz: `index.html`, `estilos.css`, `app.js`, sin dependencias externas |
| `src/experimentos.py` | Validación y experimentos; escribe CSV en `resultados/` |
| `src/preparar_imagenes.py` | Imágenes de entrada en PNG y sus SHA-256 |
| `Dockerfile`, `.dockerignore` | Imagen del servicio |
| `resultados/resultados.xlsx` | Todas las mediciones y validaciones, una hoja por archivo |

## Requisitos del anfitrión

- GPU NVIDIA con controlador que admita CUDA 12.9 o posterior (probado: RTX 5050 Laptop, `sm_120`, controlador 621.14).
- Docker con acceso a GPU: Docker Desktop 29.9.3 con backend WSL2 en Windows 11, o Docker Engine con NVIDIA Container Toolkit en Linux.

El controlador es del anfitrión; la imagen trae el toolkit CUDA 12.9 (nvcc y cabeceras), que PyCUDA necesita para compilar el kernel al iniciar.

## Construir y ejecutar

Desde `03_Aplicacion_computo_paralelo_badado_docker_pycuda/`:

```powershell
docker pull nvidia/cuda:12.9.1-devel-ubuntu22.04
docker run --rm --gpus all nvidia/cuda:12.9.1-devel-ubuntu22.04 nvidia-smi   # comprobar acceso a GPU
docker build -t convolucion-pycuda:0.1 .
docker run -d --gpus all -p 3000:3000 -p 8000:8000 --name convolucion convolucion-pycuda:0.1
docker logs -f convolucion                                                    # "GPU lista: ..."
```

- Interfaz: <http://localhost:3000>
- Documentación de la API (OpenAPI): <http://localhost:8000/docs>

Desde otro equipo de la red se usa la IP del anfitrión en lugar de `localhost`; la página busca la API en el mismo host, puerto 8000.

Variables de entorno (`-e NOMBRE=valor`): `MAX_MB` (64), `MAX_LADO` (8000), `MAX_PIXELES` (36000000), `MAX_COLA` (4), `TTL_MIN` (10, minutos que se guardan los resultados).

Para detenerlo: `docker rm -f convolucion`.

## API

| Método y ruta | Uso |
|---|---|
| `GET /api/estado` | GPU, versiones, tiempos de inicio, límites y cola |
| `GET /api/filtros` | Filtros, tamaños de máscara, coeficientes, hilos y modos de salida |
| `POST /api/procesar` | Procesa una imagen (multipart) y devuelve métricas en JSON |
| `GET /api/resultados/{id}` | PNG del resultado; `{id}_o` original en gris (entrada PGM); `{id}_f` salida float32 `.npy` |

Campos de `POST /api/procesar`:

| Campo | Valores |
|---|---|
| `imagen` | PNG, JPEG, BMP o PGM de 8 bits |
| `filtro` | `media`, `nitidez`, `laplaciano`, `personalizado` |
| `k` | media: 3, 9 o 21; nitidez y laplaciano: 3; personalizado: impar de 1 a 15 |
| `hilos` | 64, 128, 256 (por defecto), 512 o 1024 |
| `mascara` | solo personalizado: lista JSON de k·k números |
| `salida` | `recorte`, `absoluto` (por defecto en laplaciano) o `umbral` |
| `umbral`, `umbral_abs` | solo con `salida=umbral`: 0 a 255 y si se compara \|valor\| |
| `guardar_float` | `true` para obtener también la salida float32 sin recortar |

Respuesta correcta (200): `id`, dimensiones, píxeles, máscara, hilos, bloques, hilos ociosos, rango de la salida float y `tiempos_ms` (decodificación, cola, htod, kernel, dtoh, gpu_total, postproceso, codificación, servidor), más las URL del resultado. Errores: `{"error", "detalle", "id"}` con 413 (archivo, lado o píxeles sobre el límite), 415 (no es una imagen admitida), 422 (parámetros no válidos), 503 (GPU no disponible o cola llena) y 507 (memoria GPU insuficiente). Todas las respuestas llevan la cabecera `X-Request-Id`.

Ejemplo:

```powershell
curl -F imagen=@imagenes/img_4000.png -F filtro=media -F k=9 -F hilos=256 http://localhost:8000/api/procesar
```

## Imágenes

Se usan las imágenes de la práctica 02, convertidas a PNG sin cambiar un píxel, y un juego de prueba con tamaños irregulares recortado de la misma fotografía (la 1 de [`FUENTES.txt`](../01_Convolucion_Imagenes_Cuda_C/imagenes/FUENTES.txt)):

```powershell
uv sync
uv run src/preparar_imagenes.py --desde-pgm ../01_v2_Convolucion_Imagenes_pycuda/imagenes imagenes
uv run src/preparar_imagenes.py --pruebas ../01_v2_Convolucion_Imagenes_pycuda/imagenes/foto.jpg imagenes
```

Los SHA-256 quedan en `imagenes/hashes.txt` y `imagenes/hashes_pruebas.txt`.

## Validación y experimentos

Con el contenedor en marcha y las carpetas montadas:

```powershell
docker run -d --gpus all -p 3000:3000 -p 8000:8000 -v "${PWD}\imagenes:/app/imagenes:ro" -v "${PWD}\resultados:/app/resultados" --name convolucion convolucion-pycuda:0.1
uv run src/experimentos.py arranque http://localhost:8000          # justo después de iniciar
uv run src/experimentos.py validacion http://localhost:8000        # necesita PyCUDA en el anfitrión
docker exec convolucion python src/experimentos.py directo contenedor
uv run src/experimentos.py directo anfitrion                       # necesita PyCUDA en el anfitrión
uv run src/experimentos.py servicio http://localhost:8000 contenedor
uv run src/experimentos.py concurrencia http://localhost:8000 contenedor
```

La prueba de cola llena usa otro contenedor con `-e MAX_COLA=1` y `uv run src/experimentos.py cola http://localhost:8000`.

PyCUDA en el anfitrión (Windows) necesita CUDA Toolkit 13.5 y el entorno de Visual Studio, porque `SourceModule` llama a `nvcc` y este a `cl.exe`:

```bat
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
uv run src/experimentos.py directo anfitrion
```

Cada orden escribe un CSV en `resultados/`; todos se reúnen en `resultados/resultados.xlsx`.
