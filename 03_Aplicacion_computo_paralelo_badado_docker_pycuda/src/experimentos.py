"""Validación y experimentos de la práctica 03. Cada orden escribe un CSV en resultados/.

    python src/experimentos.py directo <entorno>          PyCUDA directo, sin HTTP
    python src/experimentos.py servicio <url> <entorno>   integración, resolución, filtros
    python src/experimentos.py concurrencia <url> <entorno>
    python src/experimentos.py validacion <url>           requiere GPU local para la referencia
    python src/experimentos.py cola <url>                 con el servicio en MAX_COLA=1
    python src/experimentos.py arranque <url>             justo después de iniciar el contenedor
"""

import csv
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import numpy as np

IMAGENES = Path(os.environ.get("IMAGENES", "imagenes"))
RESULTADOS = Path(os.environ.get("RESULTADOS", "resultados"))
FOTO = Path(os.environ.get("FOTO", "../01_v2_Convolucion_Imagenes_pycuda/imagenes/foto.jpg"))
REPETICIONES = 5
HILOS = 256
TOLERANCIA = 0.02

COLUMNAS = ["id_solicitud", "experimento", "modo", "entorno", "imagen", "filtro", "resolucion",
            "k", "hilos", "bloques", "repeticion", "concurrencia", "ronda",
            "decodificacion_ms", "cola_ms", "htod_ms", "kernel_ms", "dtoh_ms", "gpu_ms",
            "postproceso_ms", "codificacion_ms", "servidor_ms", "cliente_ms", "intervalo_ronda_ms",
            "http", "estado", "detalle"]

# Experimentos de la guía: (experimento, imagen, filtro, k). Se cambia un factor a la vez.
INTEGRACION = [("integracion", "img_4000.png", "media", 9)]
RESOLUCION = [("resolucion", f"img_{n}.png", "media", 9) for n in (2000, 4000, 6000)]
FILTROS = ([("filtros", "img_4000.png", "media", k) for k in (3, 9, 21)]
           + [("filtros", "img_4000.png", f, 3) for f in ("nitidez", "laplaciano")])


class Registro:
    def __init__(self, nombre, columnas=COLUMNAS):
        RESULTADOS.mkdir(parents=True, exist_ok=True)
        self.ruta = RESULTADOS / f"{nombre}.csv"
        self.archivo = open(self.ruta, "w", newline="", encoding="utf-8")
        self.escritor = csv.DictWriter(self.archivo, fieldnames=columnas, extrasaction="ignore")
        self.escritor.writeheader()
        self.columnas = columnas

    # Escribe la fila en el CSV y la muestra en consola con los decimales completos.
    def fila(self, **valores):
        self.escritor.writerow(valores)
        self.archivo.flush()
        print("\t".join(f"{c}={valores[c]}" for c in self.columnas
                        if valores.get(c) not in (None, "")), flush=True)

    def cerrar(self):
        self.archivo.close()
        print(f"Escrito {self.ruta}")



def multipart(campos, ruta_imagen=None, datos_imagen=None, nombre="imagen.png"):
    limite = uuid.uuid4().hex
    cuerpo = io.BytesIO()
    for clave, valor in campos.items():
        cuerpo.write(f"--{limite}\r\nContent-Disposition: form-data; name=\"{clave}\"\r\n\r\n"
                     f"{valor}\r\n".encode())
    if ruta_imagen is not None:
        datos_imagen = Path(ruta_imagen).read_bytes()
        nombre = Path(ruta_imagen).name
    cuerpo.write(f"--{limite}\r\nContent-Disposition: form-data; name=\"imagen\"; "
                 f"filename=\"{nombre}\"\r\nContent-Type: application/octet-stream\r\n\r\n".encode())
    cuerpo.write(datos_imagen)
    cuerpo.write(f"\r\n--{limite}--\r\n".encode())
    return cuerpo.getvalue(), f"multipart/form-data; boundary={limite}"


def obtener(url):
    with urllib.request.urlopen(url, timeout=300) as r:
        return r.read()


# Una solicitud completa vista desde el cliente: envía el formulario, lee la respuesta y
# descarga el PNG (y el .npy si se pidió). El tiempo del cliente cubre todo eso, con red.
# El cuerpo se arma antes de medir, para no contar la lectura del archivo de disco.
def solicitar(url, campos, ruta_imagen=None, datos_imagen=None, descargar=True):
    cuerpo, tipo = multipart(campos, ruta_imagen, datos_imagen)
    peticion = urllib.request.Request(f"{url}/api/procesar", data=cuerpo, method="POST",
                                      headers={"Content-Type": tipo})
    inicio = time.perf_counter()
    try:
        with urllib.request.urlopen(peticion, timeout=300) as r:
            codigo, respuesta = r.status, json.loads(r.read())
    except urllib.error.HTTPError as error:
        codigo, respuesta = error.code, json.loads(error.read() or b"{}")
    flotante = None
    if codigo == 200 and descargar:
        obtener(url + respuesta["url_resultado"])
        if respuesta.get("url_float"):
            flotante = np.load(io.BytesIO(obtener(url + respuesta["url_float"])))
    cliente_ms = (time.perf_counter() - inicio) * 1000
    return codigo, respuesta, cliente_ms, flotante


# Para el archivo que pasa el límite de MB: el servicio responde 413 al ver Content-Length y
# cierra sin leer el cuerpo. curl envía "Expect: 100-continue", recibe el 413 antes de
# mandar los datos y puede leer la respuesta; urllib seguiría enviando y vería un reset.
def solicitar_curl(url, campos, datos):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as archivo:
        archivo.write(datos)
    try:
        argumentos = ["curl", "-s", "-w", "\n%{http_code}", "-F", f"imagen=@{archivo.name}"]
        for clave, valor in campos.items():
            argumentos += ["-F", f"{clave}={valor}"]
        salida = subprocess.run(argumentos + [f"{url}/api/procesar"], capture_output=True,
                                encoding="utf-8", timeout=300).stdout
    finally:
        os.unlink(archivo.name)
    cuerpo, _, codigo = salida.rpartition("\n")
    return int(codigo), json.loads(cuerpo or "{}"), None, None


def campos_de(filtro, k, hilos=HILOS, **extra):
    campos = {"filtro": filtro, "k": k, "hilos": hilos}
    campos.update(extra)
    return campos


def fila_servicio(registro, experimento, entorno, imagen, filtro, k, repeticion, codigo,
                  r, cliente_ms, concurrencia=1, ronda="", intervalo=""):
    t = r.get("tiempos_ms", {})
    registro.fila(id_solicitud=r.get("id", ""), experimento=experimento, modo="servicio",
                  entorno=entorno, imagen=imagen, filtro=filtro,
                  resolucion=f"{r.get('ancho', '')}x{r.get('alto', '')}", k=k, hilos=HILOS,
                  bloques=r.get("bloques", ""), repeticion=repeticion,
                  concurrencia=concurrencia, ronda=ronda,
                  decodificacion_ms=t.get("decodificacion"), cola_ms=t.get("cola"),
                  htod_ms=t.get("htod"), kernel_ms=t.get("kernel"), dtoh_ms=t.get("dtoh"),
                  gpu_ms=t.get("gpu_total"), postproceso_ms=t.get("postproceso"),
                  codificacion_ms=t.get("codificacion"), servidor_ms=t.get("servidor"),
                  cliente_ms=cliente_ms, intervalo_ronda_ms=intervalo, http=codigo,
                  estado="OK" if codigo == 200 else "ERROR",
                  detalle="" if codigo == 200 else r.get("error", ""))


# Calentamiento como en la práctica 02: repetir hasta sumar al menos 1 s y tres
# ejecuciones, para que la GPU del portátil salga del estado de bajo consumo.
def calentar(funcion):
    inicio, veces = time.perf_counter(), 0
    while veces < 3 or time.perf_counter() - inicio < 1.0:
        funcion()
        veces += 1



# El mismo procesamiento del servicio, llamado sin HTTP: decodificar, GPU, 8 bits y PNG.
# Usa las funciones de servicio.py, que no dependen de FastAPI.
def procesar_directo(gpu, ruta, filtro, k):
    import servicio
    from convolucion import crear_mascara

    datos = Path(ruta).read_bytes()
    inicio = time.perf_counter()
    entrada, ancho, alto, _, _ = servicio.decodificar(datos)
    decodificacion = (time.perf_counter() - inicio) * 1000
    salida, tiempos = gpu.convolucion(entrada, crear_mascara(filtro, k), ancho, alto, k, HILOS)
    t = time.perf_counter()
    bytes_salida = servicio.a_8_bits(salida, "absoluto" if filtro == "laplaciano" else "recorte",
                                     None, None)
    postproceso = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    servicio.guardar_png(bytes_salida, ancho, alto, "directo.png")
    codificacion = (time.perf_counter() - t) * 1000
    total = (time.perf_counter() - inicio) * 1000
    return salida, tiempos, ancho, alto, decodificacion, postproceso, codificacion, total


def directo(entorno):
    import servicio
    from convolucion import GpuConvolucion

    servicio.DIR_RESULTADOS = Path(tempfile.mkdtemp())
    gpu = GpuConvolucion()
    registro = Registro(f"directo_{entorno}")
    try:
        print(gpu.info())
        for experimento, imagen, filtro, k in INTEGRACION + RESOLUCION + FILTROS:
            ruta = IMAGENES / imagen
            calentar(lambda: procesar_directo(gpu, ruta, filtro, k))
            for repeticion in range(1, REPETICIONES + 1):
                _, t, ancho, alto, dec, post, cod, total = procesar_directo(gpu, ruta, filtro, k)
                registro.fila(id_solicitud=uuid.uuid4().hex, experimento=experimento,
                              modo="directo", entorno=entorno, imagen=imagen, filtro=filtro,
                              resolucion=f"{ancho}x{alto}", k=k, hilos=HILOS, bloques=t.bloques,
                              repeticion=repeticion, concurrencia=1, decodificacion_ms=dec,
                              htod_ms=t.cpu_gpu, kernel_ms=t.kernel, dtoh_ms=t.gpu_cpu,
                              gpu_ms=t.total, postproceso_ms=post, codificacion_ms=cod,
                              servidor_ms=total, http="", estado="OK")
    finally:
        gpu.cerrar()
        registro.cerrar()



def servicio_experimentos(url, entorno):
    registro = Registro(f"servicio_{entorno}")
    for experimento, imagen, filtro, k in INTEGRACION + RESOLUCION + FILTROS:
        ruta = IMAGENES / imagen
        calentar(lambda: solicitar(url, campos_de(filtro, k), ruta))
        for repeticion in range(1, REPETICIONES + 1):
            codigo, r, cliente, _ = solicitar(url, campos_de(filtro, k), ruta)
            fila_servicio(registro, experimento, entorno, imagen, filtro, k, repeticion,
                          codigo, r, cliente)
    registro.cerrar()


# Uno y dos clientes simultáneos, una solicitud por cliente y cinco rondas. Una barrera
# hace que salgan a la vez; el intervalo de la ronda va del primer envío a la última
# respuesta completa, medido con el reloj de este mismo proceso.
def concurrencia(url, entorno):
    registro = Registro(f"concurrencia_{entorno}")
    ruta, filtro, k = IMAGENES / "img_4000.png", "media", 9
    calentar(lambda: solicitar(url, campos_de(filtro, k), ruta))
    for clientes in (1, 2):
        for ronda in range(0, REPETICIONES + 1):  # la ronda 0 es de calentamiento
            barrera = threading.Barrier(clientes + 1)
            resultados = [None] * clientes

            def cliente(i):
                barrera.wait()
                resultados[i] = solicitar(url, campos_de(filtro, k), ruta)

            hilos = [threading.Thread(target=cliente, args=(i,)) for i in range(clientes)]
            for h in hilos:
                h.start()
            barrera.wait()
            inicio = time.perf_counter()
            for h in hilos:
                h.join()
            intervalo = (time.perf_counter() - inicio) * 1000
            if ronda == 0:
                continue
            for i, (codigo, r, cliente_ms, _) in enumerate(resultados):
                fila_servicio(registro, "concurrencia", entorno, ruta.name, filtro, k, i + 1,
                              codigo, r, cliente_ms, concurrencia=clientes, ronda=ronda,
                              intervalo=intervalo)
    registro.cerrar()



COLUMNAS_VALIDACION = ["prueba", "caso", "imagen", "filtro", "k", "hilos", "bloques",
                       "esperado", "obtenido", "error_maximo", "error_medio", "pixeles_fuera",
                       "sin_calcular", "http", "error", "id_solicitud", "estado"]


# Referencia independiente de la GPU para imágenes pequeñas: convolución con extensión
# de ceros en float64, sumando la imagen desplazada por cada coeficiente.
def referencia_numpy(imagen, mascara, k):
    alto, ancho = imagen.shape
    r = k // 2
    relleno = np.zeros((alto + 2 * r, ancho + 2 * r))
    relleno[r:r + alto, r:r + ancho] = imagen
    salida = np.zeros((alto, ancho))
    for i in range(k):
        for j in range(k):
            salida += relleno[i:i + alto, j:j + ancho] * float(mascara[i * k + j])
    return salida


def errores(referencia, obtenido):
    diferencia = np.abs(referencia.astype(np.float64) - obtenido.astype(np.float64))
    validos = ~np.isnan(diferencia)
    return (float(diferencia[validos].max()) if validos.any() else 0.0,
            float(diferencia[validos].sum() / diferencia.size),
            int(np.count_nonzero(~(diferencia <= TOLERANCIA))),
            int(np.isnan(obtenido).sum()))


def leer_gris(ruta):
    from PIL import Image
    with Image.open(ruta) as imagen:
        return np.asarray(imagen.convert("L"), dtype=np.float32)


def validacion(url):
    from convolucion import GpuConvolucion, crear_mascara
    registro = Registro("validacion", COLUMNAS_VALIDACION)

    def anotar(**valores):
        valores.setdefault("estado", "OK")
        registro.fila(**valores)

    # Prueba manual: matriz de 5x5 de la guía, mismos diez casos de la práctica 02.
    figura = [-2, -1, 0, -1, 1, 1, 0, 1, 2]
    casos = [("personalizado", figura, 1, 1, 78.0), ("personalizado", figura, 1, 2, 87.0),
             ("nitidez", None, 0, 0, 95.0), ("nitidez", None, 0, 4, 153.0),
             ("nitidez", None, 1, 1, 32.0), ("laplaciano", None, 0, 0, -60.0),
             ("laplaciano", None, 1, 1, 8.0), ("laplaciano", None, 4, 4, -170.0),
             ("media", None, 0, 0, 155.0 / 9), ("media", None, 1, 1, 376.0 / 9)]
    ruta = IMAGENES / "prueba_5x5.png"
    for filtro, mascara, fila, col, esperado in casos:
        extra = {"mascara": json.dumps(mascara)} if mascara else {}
        codigo, r, _, f = solicitar(url, campos_de(filtro, 3, 64, guardar_float="true", **extra),
                                    ruta)
        obtenido = float(f[fila, col])
        anotar(prueba="manual", caso=f"{filtro} ({fila},{col})", imagen=ruta.name,
               filtro=filtro, k=3, hilos=64, bloques=r["bloques"], esperado=f"{esperado:.4f}",
               obtenido=f"{obtenido:.4f}", http=codigo, id_solicitud=r["id"],
               estado="OK" if abs(obtenido - esperado) < 1e-4 else "FALLO")

    # Límites: imágenes cuyo número de píxeles no es múltiplo de los hilos, con todos los
    # tamaños de bloque, frente a la referencia en float64.
    for nombre, filtro, k in (("prueba_37x23.png", "media", 9),
                              ("prueba_1003x997.png", "media", 3),
                              ("prueba_1921x1081.png", "laplaciano", 3)):
        imagen = leer_gris(IMAGENES / nombre)
        referencia = referencia_numpy(imagen, crear_mascara(filtro, k), k)
        for hilos in (64, 128, 256, 512, 1024):
            codigo, r, _, f = solicitar(url, campos_de(filtro, k, hilos, guardar_float="true"),
                                        IMAGENES / nombre)
            maximo, medio, fuera, nan = errores(referencia, f)
            anotar(prueba="limites", caso=f"{r['ancho']}x{r['alto']} ociosos={r['hilos_ociosos']}",
                   imagen=nombre, filtro=filtro, k=k, hilos=hilos, bloques=r["bloques"],
                   error_maximo=f"{maximo:.6f}", error_medio=f"{medio:.6f}", pixeles_fuera=fuera,
                   sin_calcular=nan, http=codigo, id_solicitud=r["id"],
                   estado="OK" if fuera == 0 and nan == 0 else "FALLO")

    # Equivalencia directa-servicio en 4000x4000 y en la imagen ancha de 7680x4320: la
    # referencia es PyCUDA directo validado, ejecutado aquí con la GPU local.
    gpu = GpuConvolucion()
    try:
        for nombre in ("img_4000.png", "prueba_7680x4320.png"):
            imagen = leer_gris(IMAGENES / nombre)
            alto, ancho = imagen.shape
            for filtro, k in (("media", 3), ("media", 9), ("media", 21),
                              ("nitidez", 3), ("laplaciano", 3)):
                referencia, _ = gpu.convolucion(imagen.ravel(), crear_mascara(filtro, k),
                                                ancho, alto, k, HILOS)
                codigo, r, _, f = solicitar(url, campos_de(filtro, k, guardar_float="true"),
                                            IMAGENES / nombre)
                maximo, medio, fuera, nan = errores(referencia.reshape(alto, ancho), f)
                anotar(prueba="equivalencia", caso="servicio frente a directo", imagen=nombre,
                       filtro=filtro, k=k, hilos=HILOS, bloques=r["bloques"],
                       error_maximo=f"{maximo:.6f}", error_medio=f"{medio:.6f}",
                       pixeles_fuera=fuera, sin_calcular=nan, http=codigo, id_solicitud=r["id"],
                       estado="OK" if fuera == 0 and nan == 0 else "FALLO")

        # Aislamiento: dos solicitudes simultáneas sobre la misma imagen con filtros
        # distintos. Cada una debe coincidir con su referencia y no con la de la otra.
        imagen = leer_gris(IMAGENES / "img_4000.png")
        alto, ancho = imagen.shape
        pedidos = [("media", 9), ("laplaciano", 3)]
        referencias = [gpu.convolucion(imagen.ravel(), crear_mascara(f, k), ancho, alto, k,
                                       HILOS)[0].reshape(alto, ancho) for f, k in pedidos]
    finally:
        gpu.cerrar()

    barrera = threading.Barrier(2)
    respuestas = [None, None]

    def enviar(i):
        filtro, k = pedidos[i]
        barrera.wait()
        respuestas[i] = solicitar(url, campos_de(filtro, k, guardar_float="true"),
                                  IMAGENES / "img_4000.png")

    hilos = [threading.Thread(target=enviar, args=(i,)) for i in range(2)]
    for h in hilos:
        h.start()
    for h in hilos:
        h.join()
    for i, (filtro, k) in enumerate(pedidos):
        codigo, r, _, f = respuestas[i]
        propio = errores(referencias[i], f)
        ajeno = errores(referencias[1 - i], f)
        anotar(prueba="aislamiento", caso=f"solicitud {i + 1}, cola {r['tiempos_ms']['cola']:.3f} ms",
               imagen="img_4000.png", filtro=filtro, k=k, hilos=HILOS, bloques=r["bloques"],
               error_maximo=f"{propio[0]:.6f}", error_medio=f"{propio[1]:.6f}",
               pixeles_fuera=propio[2], obtenido=f"fuera frente a la otra: {ajeno[2]}",
               http=codigo, id_solicitud=r["id"],
               estado="OK" if propio[2] == 0 and ajeno[2] > 0 else "FALLO")
    if respuestas[0][1]["id"] == respuestas[1][1]["id"]:
        print("FALLO: las dos solicitudes recibieron el mismo id")

    # Entradas inválidas: cada una debe recibir su código y el servicio debe seguir
    # atendiendo; tras cada caso se envía una solicitud válida de recuperación.
    pequena = IMAGENES / "img_500.png"
    invalidas = [
        ("archivo no válido", campos_de("media", 3), None, b"esto no es una imagen", 415),
        ("archivo sobre el límite de MB", campos_de("media", 3), None,
         os.urandom(65 * 2**20), 413),
        ("lado sobre el límite", campos_de("media", 3), FOTO, None, 413),
        ("píxeles sobre el límite", campos_de("media", 3),
         IMAGENES / "excede_8000x5000.png", None, 413),
        ("filtro desconocido", campos_de("sobel", 3), pequena, None, 422),
        ("máscara de media no permitida", campos_de("media", 5), pequena, None, 422),
        ("k fijo de nitidez cambiado", campos_de("nitidez", 9), pequena, None, 422),
        ("hilos no permitidos", campos_de("media", 3, 100), pequena, None, 422),
        ("máscara personalizada incompleta",
         campos_de("personalizado", 3, mascara="[1, 2]"), pequena, None, 422),
        ("umbral sin salida binaria", campos_de("media", 3, salida="recorte", umbral=100),
         pequena, None, 422),
    ]
    for caso, campos, ruta, datos, esperado in invalidas:
        if datos is not None and len(datos) > 2**20:
            codigo, r, _, _ = solicitar_curl(url, campos, datos)
        else:
            codigo, r, _, _ = solicitar(url, campos, ruta, datos)
        recuperacion, rr, _, _ = solicitar(url, campos_de("media", 3), pequena)
        anotar(prueba="invalidas", caso=caso, imagen=Path(ruta).name if ruta else "",
               filtro=campos["filtro"], k=campos["k"], hilos=campos["hilos"],
               esperado=esperado, obtenido=f"recuperación {recuperacion}", http=codigo,
               error=r.get("detalle", r.get("error", "")), id_solicitud=r.get("id", ""),
               estado="OK" if codigo == esperado and recuperacion == 200 else "FALLO")
    registro.cerrar()


# Con el servicio iniciado con MAX_COLA=1, cuatro clientes envían a la vez la imagen de
# 6000x6000; uno debe entrar y el resto recibir 503. Después se comprueba la recuperación.
def cola(url):
    registro = Registro("cola")
    ruta = IMAGENES / "img_6000.png"
    for ronda in range(1, REPETICIONES + 1):
        barrera = threading.Barrier(4)
        resultados = [None] * 4

        def cliente(i):
            barrera.wait()
            resultados[i] = solicitar(url, campos_de("media", 21), ruta)

        hilos = [threading.Thread(target=cliente, args=(i,)) for i in range(4)]
        for h in hilos:
            h.start()
        for h in hilos:
            h.join()
        for i, (codigo, r, cliente_ms, _) in enumerate(resultados):
            fila_servicio(registro, "cola", "contenedor_max_cola_1", ruta.name, "media", 21,
                          i + 1, codigo, r, cliente_ms, concurrencia=4, ronda=ronda)
    codigo, r, cliente_ms, _ = solicitar(url, campos_de("media", 21), ruta)
    fila_servicio(registro, "cola_recuperacion", "contenedor_max_cola_1", ruta.name, "media",
                  21, 1, codigo, r, cliente_ms)
    registro.cerrar()


# Arranque en frío: espera a que la API responda y registra lo que el servicio midió al
# iniciar (contexto, compilación y primer lanzamiento) y la primera solicitud real.
def arranque(url):
    inicio = time.perf_counter()
    while True:
        try:
            estado = json.loads(obtener(f"{url}/api/estado"))
            break
        except (urllib.error.URLError, ConnectionError):
            time.sleep(0.2)
    espera = (time.perf_counter() - inicio) * 1000
    codigo, r, cliente_ms, _ = solicitar(url, campos_de("media", 9), IMAGENES / "img_4000.png")
    with open(RESULTADOS / "arranque.csv", "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["medida", "valor_ms", "alcance"])
        g = estado["gpu"]
        escritor.writerow(["contexto", g["ms_contexto"], "cuda.init y make_context"])
        escritor.writerow(["compilacion", g["ms_compilacion"], "SourceModule (nvcc o caché)"])
        escritor.writerow(["primer_lanzamiento", g["ms_calentamiento"],
                           "carga del módulo y kernel 64x64"])
        escritor.writerow(["espera_cliente_hasta_api", espera, "desde que se llama a este script"])
        escritor.writerow(["primera_solicitud_kernel", r["tiempos_ms"]["kernel"], "media 9x9, 4000"])
        escritor.writerow(["primera_solicitud_servidor", r["tiempos_ms"]["servidor"], ""])
        escritor.writerow(["primera_solicitud_cliente", cliente_ms, "incluye descarga del PNG"])
    print(json.dumps(estado, indent=1), f"\nprimera solicitud {codigo}: kernel "
          f"{r['tiempos_ms']['kernel']:.3f} ms, servidor {r['tiempos_ms']['servidor']:.1f} ms, "
          f"cliente {cliente_ms:.1f} ms")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    orden, argumentos = sys.argv[1] if len(sys.argv) > 1 else "", sys.argv[2:]
    ordenes = {"directo": directo, "servicio": servicio_experimentos,
               "concurrencia": concurrencia, "validacion": validacion, "cola": cola,
               "arranque": arranque}
    if orden not in ordenes:
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(1)
    ordenes[orden](*argumentos)
