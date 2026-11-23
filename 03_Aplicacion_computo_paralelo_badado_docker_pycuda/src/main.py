"""Servicio de convolución: API REST en el puerto 8000 e interfaz web en el 3000.

    python src/main.py
"""

import asyncio
import logging
import os
import platform
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

import servicio
from convolucion import crear_mascara
from servicio import ErrorSolicitud, TrabajadorGpu

HOST = os.environ.get("HOST", "0.0.0.0")
PUERTO_API = int(os.environ.get("PUERTO_API", "8000"))
PUERTO_WEB = int(os.environ.get("PUERTO_WEB", "3000"))
PUBLIC = Path(__file__).parent / "public"
ID_VALIDO = re.compile(r"^[0-9a-f]{32}$")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")


async def limpiar_periodicamente():
    while True:
        await asyncio.sleep(60)
        await run_in_threadpool(servicio.borrar_antiguos)


# La GPU se inicializa al arrancar, antes de aceptar solicitudes, para que la creación del
# contexto y la compilación no se sumen a la primera solicitud.
@asynccontextmanager
async def ciclo_de_vida(app):
    servicio.DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    app.state.trabajador = await run_in_threadpool(TrabajadorGpu)
    limpieza = asyncio.create_task(limpiar_periodicamente())
    yield
    limpieza.cancel()
    await run_in_threadpool(app.state.trabajador.cerrar)


api = FastAPI(title="Convolución PyCUDA", version="0.1.0", lifespan=ciclo_de_vida,
              description="Convolución de imágenes en GPU con kernels propios (práctica 03).")

# La interfaz se sirve desde otro puerto, así que el navegador pide permiso CORS. Se acepta
# cualquier host en el puerto de la interfaz para poder entrar desde otro equipo de la red.
api.add_middleware(CORSMiddleware, allow_origin_regex=rf"^https?://[^/]+:{PUERTO_WEB}$",
                   allow_methods=["GET", "POST"], expose_headers=["X-Request-Id"])


# Cada solicitud recibe un id que va en la cabecera X-Request-Id y nombra su resultado.
# El tamaño se revisa con Content-Length antes de leer el cuerpo.
@api.middleware("http")
async def identificar(request: Request, siguiente):
    request.state.id = uuid.uuid4().hex
    largo = request.headers.get("content-length")
    if request.url.path == "/api/procesar" and largo and int(largo) > servicio.MAX_MB * 2**20 + 2**16:
        respuesta = error_json(request, ErrorSolicitud(
            413, "archivo_muy_grande",
            f"El archivo pesa {int(largo) / 2**20:.1f} MB; el máximo es {servicio.MAX_MB} MB"))
    else:
        respuesta = await siguiente(request)
    respuesta.headers["X-Request-Id"] = request.state.id
    return respuesta


def error_json(request, error):
    log.warning("%s %d %s: %s", request.state.id, error.estado, error.error, error.detalle)
    return JSONResponse(status_code=error.estado,
                        content={"error": error.error, "detalle": error.detalle,
                                 "id": request.state.id})


@api.exception_handler(ErrorSolicitud)
async def manejar_error(request: Request, error: ErrorSolicitud):
    return error_json(request, error)


@api.exception_handler(RequestValidationError)
async def manejar_validacion(request: Request, error: RequestValidationError):
    detalle = "; ".join(f"{'.'.join(map(str, e['loc'][1:]))}: {e['msg']}" for e in error.errors())
    return error_json(request, ErrorSolicitud(422, "parametros_invalidos", detalle))


@api.exception_handler(Exception)
async def manejar_inesperado(request: Request, error: Exception):
    log.exception("%s error no controlado", request.state.id)
    return JSONResponse(status_code=500, headers={"X-Request-Id": request.state.id},
                        content={"error": "error_interno", "detalle": str(error),
                                 "id": request.state.id})


@api.get("/api/estado", summary="Estado de la GPU, versiones y cola")
async def estado(request: Request):
    return {**request.app.state.trabajador.estado(),
            "python": platform.python_version(), "numpy": np.__version__,
            "limites": {"max_mb": servicio.MAX_MB, "max_lado": servicio.MAX_LADO,
                        "max_pixeles": servicio.MAX_PIXELES, "max_cola": servicio.MAX_COLA,
                        "ttl_min": servicio.TTL_MIN}}


@api.get("/api/filtros", summary="Filtros, máscaras, tamaños, hilos y modos de salida")
async def filtros():
    mascaras = {f"media_{k}": crear_mascara("media", k).tolist() for k in servicio.TAMANOS["media"]}
    mascaras["nitidez_3"] = crear_mascara("nitidez", 3).tolist()
    mascaras["laplaciano_3"] = crear_mascara("laplaciano", 3).tolist()
    return {"filtros": servicio.FILTROS, "tamanos": servicio.TAMANOS, "mascaras": mascaras,
            "max_k_personalizada": servicio.MAX_K_PERSONALIZADA,
            "hilos": servicio.HILOS, "salidas": servicio.MODOS_SALIDA,
            "limites": {"max_mb": servicio.MAX_MB, "max_lado": servicio.MAX_LADO,
                        "max_pixeles": servicio.MAX_PIXELES}}


@api.post(
    "/api/procesar",
    summary="Aplica un filtro en la GPU y devuelve métricas y la URL del resultado",
    responses={413: {"description": "Archivo o dimensiones por encima del límite"},
               415: {"description": "El archivo no es una imagen admitida"},
               422: {"description": "Filtro, k, hilos, máscara o umbral no válidos"},
               503: {"description": "GPU no disponible o cola llena"},
               507: {"description": "Memoria GPU insuficiente"}},
)
async def procesar(
    request: Request,
    imagen: UploadFile = File(..., description="PNG, JPEG, BMP o PGM de 8 bits"),
    filtro: str = Form("media", description="media | nitidez | laplaciano | personalizado"),
    k: int | None = Form(None, description="Lado de la máscara. media: 3, 9 o 21; nitidez y "
                         "laplaciano: 3 (fijo); personalizado: impar de 1 a 15"),
    hilos: int = Form(256, description="Hilos por bloque: 64, 128, 256, 512 o 1024"),
    mascara: str | None = Form(None, description="Solo personalizado: lista JSON de k*k números"),
    salida: str | None = Form(None, description="recorte | absoluto | umbral. Por defecto "
                              "absoluto en laplaciano y recorte en los demás"),
    umbral: int | None = Form(None, description="Solo con salida=umbral: T de 0 a 255"),
    umbral_abs: bool | None = Form(None, description="Solo con salida=umbral: comparar |valor|"),
):
    inicio = time.perf_counter()
    id_solicitud = request.state.id
    coeficientes, k, salida = servicio.validar_parametros(
        filtro, k, hilos, mascara, salida, umbral, umbral_abs)

    datos = await imagen.read()
    if len(datos) > servicio.MAX_MB * 2**20:
        raise ErrorSolicitud(413, "archivo_muy_grande", f"El máximo es {servicio.MAX_MB} MB")

    # Decodificación y preparación: formato real, escala de grises y float32.
    t = time.perf_counter()
    entrada, ancho, alto, formato, gris = await run_in_threadpool(servicio.decodificar, datos)
    ms_decodificacion = (time.perf_counter() - t) * 1000

    resultado, tiempos, ms_cola = await request.app.state.trabajador.ejecutar(
        entrada, coeficientes, ancho, alto, k, hilos)

    # Paso a 8 bits según el modo y escritura del PNG con nombre único.
    t = time.perf_counter()
    bytes_salida = await run_in_threadpool(servicio.a_8_bits, resultado, salida, umbral, umbral_abs)
    ms_postproceso = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    await run_in_threadpool(servicio.guardar_png, bytes_salida, ancho, alto, f"{id_solicitud}.png")
    ms_codificacion = (time.perf_counter() - t) * 1000

    # El navegador no muestra PGM; en ese caso se guarda también la entrada en gris como PNG.
    url_original = None
    if formato == "PPM":
        await run_in_threadpool(gris.save, servicio.DIR_RESULTADOS / f"{id_solicitud}_o.png",
                                compress_level=1)
        url_original = f"/api/resultados/{id_solicitud}_o"

    pixeles = ancho * alto
    respuesta = {
        "id": id_solicitud, "formato": formato, "bytes": len(datos),
        "ancho": ancho, "alto": alto, "pixeles": pixeles,
        "filtro": filtro, "k": k, "mascara": coeficientes.tolist(),
        "hilos": hilos, "bloques": tiempos.bloques, "hilos_lanzados": tiempos.bloques * hilos,
        "hilos_ociosos": tiempos.bloques * hilos - pixeles,
        "salida": salida, "umbral": umbral, "umbral_abs": bool(umbral_abs),
        "rango_float": {"min": float(np.nanmin(resultado)), "max": float(np.nanmax(resultado)),
                        "sin_calcular": int(np.isnan(resultado).sum())},
        "tiempos_ms": {"decodificacion": ms_decodificacion, "cola": ms_cola,
                       "htod": tiempos.cpu_gpu, "kernel": tiempos.kernel,
                       "dtoh": tiempos.gpu_cpu, "gpu_total": tiempos.total,
                       "postproceso": ms_postproceso, "codificacion": ms_codificacion,
                       "servidor": (time.perf_counter() - inicio) * 1000},
        "url_resultado": f"/api/resultados/{id_solicitud}",
        "url_original": url_original,
    }
    log.info("%s %s k=%d hilos=%d %dx%d kernel=%.3f ms servidor=%.1f ms", id_solicitud, filtro,
             k, hilos, ancho, alto, tiempos.kernel, respuesta["tiempos_ms"]["servidor"])
    return respuesta


@api.get("/api/resultados/{nombre}", summary="Descarga el PNG de una solicitud")
async def resultado(nombre: str):
    base = nombre.removesuffix("_o")
    ruta = servicio.DIR_RESULTADOS / f"{nombre}.png"
    if not ID_VALIDO.match(base) or not ruta.is_file():
        raise ErrorSolicitud(404, "resultado_no_encontrado",
                             f"No existe el resultado {nombre} o ya se borró "
                             f"(se guardan {servicio.TTL_MIN} min)")
    return FileResponse(ruta, media_type="image/png", filename=f"convolucion_{nombre}.png")


web = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
web.mount("/", StaticFiles(directory=PUBLIC, html=True), name="public")


# Un solo proceso con dos servidores: así hay un único contexto CUDA para toda la API.
async def principal():
    servidores = [
        uvicorn.Server(uvicorn.Config(api, host=HOST, port=PUERTO_API, log_level="info")),
        uvicorn.Server(uvicorn.Config(web, host=HOST, port=PUERTO_WEB, log_level="warning")),
    ]
    log.info("API en http://%s:%d/docs | interfaz en http://%s:%d", HOST, PUERTO_API, HOST, PUERTO_WEB)
    await asyncio.gather(*(s.serve() for s in servidores))


if __name__ == "__main__":
    asyncio.run(principal())
