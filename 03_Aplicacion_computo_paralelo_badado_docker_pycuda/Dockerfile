# Servicio de convolución con PyCUDA.
#   docker build -t convolucion-pycuda:0.1 .
#   docker run --rm --gpus all -p 3000:3000 -p 8000:8000 convolucion-pycuda:0.1
#
# El driver NVIDIA es del anfitrión y lo monta NVIDIA Container Toolkit con --gpus; la
# imagen solo trae el toolkit CUDA 12.9 (nvcc, cabeceras). CUDA 12.8+ es necesario para
# compilar a sm_120 (RTX serie 50); el driver del anfitrión debe admitir CUDA >= 12.9.

ARG CUDA_IMAGEN=nvidia/cuda:12.9.1-devel-ubuntu22.04

# Etapa 1: Python 3.12 y dependencias. pycuda se compila aquí contra las cabeceras CUDA,
# así que el compilador de C++ y las cabeceras de Python no pasan a la imagen final.
FROM ${CUDA_IMAGEN} AS construccion
COPY --from=ghcr.io/astral-sh/uv:0.13.8 /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_PREFERENCE=only-managed
WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install 3.12 && \
    uv sync --frozen --no-dev --no-install-project

# Etapa 2: misma base CUDA devel, porque SourceModule llama a nvcc al iniciar el servicio.
FROM ${CUDA_IMAGEN}
RUN apt-get update && \
    apt-get upgrade -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/* && \
    useradd --create-home --uid 1000 app
COPY --from=construccion /opt/python /opt/python
COPY --from=construccion --chown=app:app /app/.venv /app/.venv
WORKDIR /app
COPY --chown=app:app src ./src

ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    DIR_RESULTADOS=/tmp/resultados \
    PUERTO_WEB=3000 \
    PUERTO_API=8000
USER app
EXPOSE 3000 8000
CMD ["python", "src/main.py"]
