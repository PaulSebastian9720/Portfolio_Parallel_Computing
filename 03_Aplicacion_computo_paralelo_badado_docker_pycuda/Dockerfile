# Servicio de convolución con PyCUDA.
#   docker build -t convolucion-pycuda:0.1 .
#   docker run --rm --gpus all -p 3000:3000 -p 8000:8000 convolucion-pycuda:0.1
#
# El driver NVIDIA es del anfitrión y lo monta NVIDIA Container Toolkit con --gpus; la
# imagen solo trae el toolkit CUDA 12.9 (nvcc, cabeceras). CUDA 12.8+ es necesario para
# compilar a sm_120 (RTX serie 50); el driver del anfitrión debe admitir CUDA >= 12.9.
# Ubuntu 22.04 trae Python 3.10 y numpy 2.5 pide 3.12, por eso se usa el PPA deadsnakes.

ARG CUDA_IMAGEN=nvidia/cuda:12.9.1-devel-ubuntu22.04

# Etapa 1: entorno virtual con las dependencias de pyproject.toml. pycuda se compila aquí
# contra las cabeceras CUDA y de Python; esas cabeceras no pasan a la imagen final.
FROM ${CUDA_IMAGEN} AS construccion
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common gpg-agent && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends python3.12 python3.12-dev python3.12-venv && \
    rm -rf /var/lib/apt/lists/*

# pyproject.toml es la única lista de dependencias: se extrae con tomllib a un
# requirements.txt y se instala con pip, sin uv.
WORKDIR /app
COPY pyproject.toml ./
RUN python3.12 -m venv /opt/venv && \
    python3.12 -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']))" > requirements.txt && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# Etapa 2: misma base CUDA devel, porque SourceModule llama a nvcc al iniciar el servicio.
# Solo se instala el intérprete 3.12 (sin -dev) y se copia el entorno virtual.
FROM ${CUDA_IMAGEN}
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common gpg-agent && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends python3.12 && \
    apt-get purge -y --auto-remove software-properties-common gpg-agent && \
    apt-get upgrade -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/* && \
    useradd --create-home --uid 1000 app

COPY --from=construccion /opt/venv /opt/venv
WORKDIR /app
COPY --chown=app:app src ./src

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    DIR_RESULTADOS=/tmp/resultados \
    PUERTO_WEB=3000 \
    PUERTO_API=8000
USER app
EXPOSE 3000 8000
CMD ["python", "src/main.py"]
