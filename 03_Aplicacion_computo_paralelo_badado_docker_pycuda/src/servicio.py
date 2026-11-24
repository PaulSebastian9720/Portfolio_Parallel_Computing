"""Gestión de solicitudes: validación, hilo de GPU con cola acotada y archivos de salida."""

import asyncio
import io
import json
import logging
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pycuda.driver as cuda
from PIL import Image

from convolucion import GpuConvolucion, crear_mascara

log = logging.getLogger("servicio")

MAX_MB = int(os.environ.get("MAX_MB", "64"))
# Se acepta cualquier ancho y alto (no cuadrada, impar, no múltiplo de los hilos) con dos
# límites: el lado y el total de píxeles. 6000x6000 es la imagen más grande de los
# experimentos; 8000 por lado deja pasar, por ejemplo, 8000x4000 con los mismos píxeles.
MAX_LADO = int(os.environ.get("MAX_LADO", "8000"))
MAX_PIXELES = int(os.environ.get("MAX_PIXELES", str(6000 * 6000)))
MAX_COLA = int(os.environ.get("MAX_COLA", "4"))
TTL_MIN = int(os.environ.get("TTL_MIN", "10"))
DIR_RESULTADOS = Path(os.environ.get(
    "DIR_RESULTADOS", Path(tempfile.gettempdir()) / "convolucion_resultados"))

FILTROS = ("media", "nitidez", "laplaciano", "personalizado")
# Solo la media elige k entre los tamaños de la práctica 02. La personalizada acepta
# cualquier k impar entre 1 y MAX_K_PERSONALIZADA, para que el píxel central exista.
MAX_K_PERSONALIZADA = 15
TAMANOS = {"media": (3, 9, 21), "nitidez": (3,), "laplaciano": (3,),
           "personalizado": tuple(range(1, MAX_K_PERSONALIZADA + 1, 2))}
HILOS = (64, 128, 256, 512, 1024)
MODOS_SALIDA = ("recorte", "absoluto", "umbral")
# Pillow llama PPM también al PGM, y MPO al JPEG con varias imágenes que guardan los
# celulares (se usa la primera).
FORMATOS = {"PNG", "JPEG", "MPO", "BMP", "PPM"}
MODOS_PIXEL = {"1", "L", "LA", "P", "RGB", "RGBA"}  # 8 bits por canal

# Los límites propios se revisan con la cabecera, antes de decodificar; se quita el de
# Pillow para que una imagen enorme reciba el mensaje de límite y no uno de formato.
Image.MAX_IMAGE_PIXELS = None


# Error con código HTTP y mensaje para el cliente. Lo lanzan la validación y el trabajador.
class ErrorSolicitud(Exception):
    def __init__(self, estado, error, detalle):
        super().__init__(detalle)
        self.estado = estado
        self.error = error
        self.detalle = detalle


# Devuelve la máscara (float32 plana) y el modo de salida ya resueltos.
def validar_parametros(filtro, k, hilos, mascara_json, salida, umbral, umbral_abs):
    if filtro not in FILTROS:
        raise ErrorSolicitud(422, "filtro_desconocido",
                             f"Filtro '{filtro}' no existe. Opciones: {', '.join(FILTROS)}")
    # Nitidez y laplaciano son siempre 3x3; si no se envía k se toma el primero permitido.
    if k is None:
        k = TAMANOS[filtro][0]
    if k not in TAMANOS[filtro]:
        permitidos = (f"impar de 1 a {MAX_K_PERSONALIZADA}" if filtro == "personalizado"
                      else ", ".join(map(str, TAMANOS[filtro])))
        raise ErrorSolicitud(422, "mascara_no_permitida",
                             f"k={k} no es válido para {filtro}. Permitido: {permitidos}")
    if hilos not in HILOS:
        raise ErrorSolicitud(422, "hilos_no_permitidos",
                             f"hilos={hilos} no es válido. Permitidos: {HILOS}")
    if salida is None:
        salida = "absoluto" if filtro == "laplaciano" else "recorte"
    if salida not in MODOS_SALIDA:
        raise ErrorSolicitud(422, "salida_desconocida",
                             f"Modo de salida '{salida}' no existe. Opciones: {MODOS_SALIDA}")

    # El umbral solo tiene sentido en la salida binaria; recorte y absoluto pasan los
    # valores directamente, así que ahí no se admite.
    if salida == "umbral":
        if umbral is None:
            raise ErrorSolicitud(422, "umbral_requerido", "salida=umbral necesita el campo umbral")
        if not 0 <= umbral <= 255:
            raise ErrorSolicitud(422, "umbral_fuera_de_rango",
                                 f"umbral={umbral}; debe estar entre 0 y 255")
    elif umbral is not None or umbral_abs is not None:
        raise ErrorSolicitud(422, "umbral_no_aplica",
                             f"El umbral solo se usa con salida=umbral, no con salida={salida}")

    if filtro != "personalizado":
        if mascara_json is not None:
            raise ErrorSolicitud(422, "mascara_no_aplica",
                                 f"El filtro {filtro} usa su máscara fija; mascara solo va con personalizado")
        return crear_mascara(filtro, k), k, salida

    try:
        valores = np.asarray(json.loads(mascara_json or ""), dtype=np.float64).ravel()
    except (ValueError, TypeError):
        raise ErrorSolicitud(422, "mascara_invalida",
                             "La máscara debe ser una lista JSON de números") from None
    if valores.size != k * k:
        raise ErrorSolicitud(422, "mascara_invalida",
                             f"La máscara {k}x{k} necesita {k * k} valores y llegaron {valores.size}")
    if not np.all(np.isfinite(valores)) or np.any(np.abs(valores) > 1e6):
        raise ErrorSolicitud(422, "mascara_invalida",
                             "Los coeficientes deben ser finitos y de magnitud menor a 1e6")
    return np.ascontiguousarray(valores, dtype=np.float32), k, salida


# Comprueba el formato real con Pillow (no la extensión) y las dimensiones antes de
# decodificar los píxeles; luego pasa a escala de grises y a float32 plano.
def decodificar(datos):
    try:
        imagen = Image.open(io.BytesIO(datos))
    except Exception:
        raise ErrorSolicitud(415, "formato_no_valido",
                             "El archivo no es una imagen PNG, JPEG, BMP o PGM") from None
    if imagen.format not in FORMATOS:
        raise ErrorSolicitud(415, "formato_no_valido",
                             f"Formato {imagen.format} no admitido. Use PNG, JPEG, BMP o PGM")
    if imagen.mode not in MODOS_PIXEL:
        raise ErrorSolicitud(415, "profundidad_no_admitida",
                             f"Modo de píxel {imagen.mode} no admitido; se esperan 8 bits por canal")

    ancho, alto = imagen.size
    if ancho < 1 or alto < 1:
        raise ErrorSolicitud(422, "dimensiones_invalidas", f"La imagen mide {ancho}x{alto}")
    if ancho > MAX_LADO or alto > MAX_LADO:
        raise ErrorSolicitud(413, "lado_excedido",
                             f"La imagen mide {ancho}x{alto}; ningún lado puede pasar de {MAX_LADO}")
    if ancho * alto > MAX_PIXELES:
        raise ErrorSolicitud(413, "pixeles_excedidos",
                             f"La imagen tiene {ancho * alto:,} píxeles ({ancho}x{alto}); "
                             f"el máximo es {MAX_PIXELES:,}")
    try:
        gris = imagen.convert("L")
    except Exception:
        raise ErrorSolicitud(415, "imagen_corrupta",
                             "No se pudieron leer los píxeles de la imagen") from None
    entrada = np.asarray(gris, dtype=np.uint8).astype(np.float32).ravel()
    return entrada, ancho, alto, imagen.format, gris


# Pasa la salida float a 8 bits según el modo elegido. Solo es para mostrar la imagen; la
# comparación numérica usa los float antes de este paso.
def a_8_bits(valores, salida, umbral, umbral_abs):
    if salida == "umbral":
        base = np.abs(valores) if umbral_abs else valores
        # El NaN (píxel sin calcular) compara falso y queda en 0.
        return np.where(base >= umbral, 255, 0).astype(np.uint8)
    if salida == "absoluto":
        valores = np.abs(valores)
    return np.clip(np.rint(valores), 0, 255).astype(np.uint8)


def guardar_png(arreglo, ancho, alto, nombre):
    ruta = DIR_RESULTADOS / nombre
    Image.fromarray(arreglo.reshape(alto, ancho), mode="L").save(ruta, compress_level=1)
    return ruta


def borrar_antiguos():
    limite = time.time() - TTL_MIN * 60
    borrados = 0
    for ruta in DIR_RESULTADOS.glob("*.*"):
        try:
            if ruta.stat().st_mtime < limite:
                ruta.unlink()
                borrados += 1
        except FileNotFoundError:
            pass
    if borrados:
        log.info("limpieza: %d archivos con más de %d min", borrados, TTL_MIN)


# Toda la GPU pasa por un único hilo: el contexto CUDA se crea en él y queda activo ahí, y
# las convoluciones se ejecutan una tras otra. Una solicitud ocupa un lugar desde que se
# acepta hasta que responde (decodificación, GPU y PNG); si ya hay MAX_COLA lugares
# ocupados, la siguiente se rechaza con 503. Así se acota también la memoria del anfitrión.
class TrabajadorGpu:

    def __init__(self):
        self.ejecutor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")
        self.gpu = None
        self.error_inicio = None
        self.info = None
        self.pendientes = 0
        self.atendidas = 0
        self.rechazadas = 0
        self.candado = threading.Lock()
        self.ejecutor.submit(self._iniciar).result()

    def _iniciar(self):
        try:
            self.gpu = GpuConvolucion()
            # El primer lanzamiento carga el módulo en la GPU (carga diferida de CUDA) y
            # tarda cientos de ms. Se hace aquí con una imagen pequeña y se registra
            # aparte, para que no caiga en la primera solicitud.
            inicio = time.perf_counter()
            self.gpu.convolucion(np.zeros(64 * 64, np.float32), crear_mascara("media", 3),
                                 64, 64, 3, 256)
            ms_calentamiento = (time.perf_counter() - inicio) * 1000
            self.info = self.gpu.info()
            self.info["ms_calentamiento"] = ms_calentamiento
            log.info("GPU lista: %s | contexto %.1f ms | compilación %.1f ms | primer "
                     "lanzamiento %.1f ms", self.info["nombre"], self.info["ms_contexto"],
                     self.info["ms_compilacion"], ms_calentamiento)
        except Exception as error:
            self.error_inicio = f"{type(error).__name__}: {error}"
            log.error("No se pudo iniciar CUDA: %s", self.error_inicio)

    def cerrar(self):
        if self.gpu is not None:
            self.ejecutor.submit(self.gpu.cerrar).result()
        self.ejecutor.shutdown()

    def estado(self):
        return {"disponible": self.gpu is not None, "error": self.error_inicio,
                "gpu": self.info, "pendientes": self.pendientes, "max_cola": MAX_COLA,
                "atendidas": self.atendidas, "rechazadas": self.rechazadas}

    @contextmanager
    def turno(self):
        if self.gpu is None:
            raise ErrorSolicitud(503, "gpu_no_disponible",
                                 f"No hay GPU CUDA disponible: {self.error_inicio}")
        with self.candado:
            if self.pendientes >= MAX_COLA:
                self.rechazadas += 1
                raise ErrorSolicitud(503, "cola_llena",
                                     f"Hay {self.pendientes} solicitudes en curso; máximo {MAX_COLA}")
            self.pendientes += 1
        try:
            yield
        finally:
            with self.candado:
                self.pendientes -= 1

    # La espera en cola es el tiempo entre pedir el hilo de la GPU y empezar a usarlo.
    async def ejecutar(self, entrada, mascara, ancho, alto, k, hilos):
        encolada = time.perf_counter()
        return await asyncio.get_running_loop().run_in_executor(
            self.ejecutor, self._convolucion, encolada, entrada, mascara, ancho, alto, k, hilos)

    # Corre en el hilo de la GPU. Antes de reservar comprueba que entrada y salida (dos
    # arreglos float32 del tamaño de la imagen) quepan en la memoria libre.
    def _convolucion(self, encolada, entrada, mascara, ancho, alto, k, hilos):
        ms_cola = (time.perf_counter() - encolada) * 1000
        necesaria = 2 * entrada.nbytes + mascara.nbytes
        libre, _ = self.gpu.memoria()
        if necesaria > libre * 0.9:
            raise ErrorSolicitud(507, "memoria_gpu_insuficiente",
                                 f"Se necesitan {necesaria / 2**20:.0f} MB en la GPU y hay "
                                 f"{libre / 2**20:.0f} MB libres")
        try:
            salida, tiempos = self.gpu.convolucion(entrada, mascara, ancho, alto, k, hilos)
        except cuda.MemoryError as error:
            raise ErrorSolicitud(507, "memoria_gpu_insuficiente", str(error)) from None
        except cuda.Error as error:
            raise ErrorSolicitud(500, "error_cuda", f"{type(error).__name__}: {error}") from None
        self.atendidas += 1
        self.info["memoria_libre_mb"] = round(libre / 2**20)
        return salida, tiempos, ms_cola
