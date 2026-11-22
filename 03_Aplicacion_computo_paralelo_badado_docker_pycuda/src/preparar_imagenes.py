"""Genera las imágenes PNG de los experimentos, desde una foto o desde los PGM de la práctica 02.

    uv run src/preparar_imagenes.py <foto_origen> imagenes [--jpg]
    uv run src/preparar_imagenes.py --desde-pgm <carpeta_pgm> imagenes [--jpg]
"""

import hashlib
import sys
from pathlib import Path

from PIL import Image, ImageOps

RESOLUCIONES = (500, 2000, 4000, 6000)
CALIDAD_JPG = 95


# Gris de 8 bits, recorte cuadrado central y reducción con BOX, que promedia por
# área como INTER_AREA. El decodificador JPEG, la conversión a gris y el remuestreo
# no son los de OpenCV, así que los píxeles no coinciden con los PGM de la práctica
# 02; para reproducirlos exactamente se usa --desde-pgm.
def desde_foto(origen):
    with Image.open(origen) as foto:
        gris = ImageOps.exif_transpose(foto).convert("L")

    ancho, alto = gris.size
    lado = min(ancho, alto)
    if lado < max(RESOLUCIONES):
        raise SystemExit(f"La imagen debe medir al menos 6000x6000; su recorte mide {lado}x{lado}")

    x = (ancho - lado) // 2
    y = (alto - lado) // 2
    recorte = gris.crop((x, y, x + lado, y + lado))
    print(f"Origen: {ancho}x{alto}, recorte: {lado}x{lado} en ({x}, {y})")

    for resolucion in RESOLUCIONES:
        if resolucion == lado:
            yield resolucion, recorte
        else:
            yield resolucion, recorte.resize((resolucion, resolucion), Image.Resampling.BOX)


# Conversión directa de img_<n>.pgm a PNG, sin redimensionar: los píxeles quedan
# idénticos a los de la práctica 02.
def desde_pgm(carpeta):
    for resolucion in RESOLUCIONES:
        ruta = Path(carpeta) / f"img_{resolucion}.pgm"
        if not ruta.exists():
            print(f"No existe {ruta}, se omite", file=sys.stderr)
            continue
        with Image.open(ruta) as pgm:
            if pgm.mode != "L":
                raise SystemExit(f"{ruta} no es un PGM de 8 bits (modo {pgm.mode})")
            yield resolucion, pgm.copy()


def sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def escribir(ruta, imagen, **opciones):
    imagen.save(ruta, **opciones)
    ancho, alto = imagen.size
    return f"{ruta.name}  {ancho}x{alto}  {ruta.stat().st_size} bytes  sha256={sha256(ruta)}"


def preparar(imagenes, destino, con_jpg):
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    lineas = []
    for resolucion, imagen in imagenes:
        # PNG en gris de 8 bits: sin pérdida, sirve para la validación numérica.
        lineas.append(escribir(destino / f"img_{resolucion}.png", imagen))
        if con_jpg:
            # JPEG tiene pérdida: solo para probar la subida desde el navegador,
            # no para comparar resultados con la ejecución directa.
            lineas.append(escribir(destino / f"img_{resolucion}.jpg", imagen,
                                   quality=CALIDAD_JPG, subsampling=0))

    for linea in lineas:
        print(linea)
    (destino / "hashes.txt").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"Escrito {destino / 'hashes.txt'}")


if __name__ == "__main__":
    argumentos = sys.argv[1:]
    con_jpg = "--jpg" in argumentos
    argumentos = [a for a in argumentos if a != "--jpg"]

    if len(argumentos) == 3 and argumentos[0] == "--desde-pgm":
        preparar(desde_pgm(argumentos[1]), argumentos[2], con_jpg)
    elif len(argumentos) == 2 and not argumentos[0].startswith("--"):
        preparar(desde_foto(argumentos[0]), argumentos[1], con_jpg)
    else:
        print("Uso: preparar_imagenes.py <foto_origen> <carpeta_salida> [--jpg]\n"
              "     preparar_imagenes.py --desde-pgm <carpeta_pgm> <carpeta_salida> [--jpg]",
              file=sys.stderr)
        sys.exit(1)
