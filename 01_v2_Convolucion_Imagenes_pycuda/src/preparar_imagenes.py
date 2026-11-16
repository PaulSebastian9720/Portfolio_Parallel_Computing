"""Genera las imágenes de entrada a partir de la foto (equivale a preparar.cpp).

    uv run src/preparar_imagenes.py imagenes/foto.jpg imagenes
"""

import sys
from pathlib import Path

import cv2

# 2000, 4000 y 6000 son las de la práctica 01. 500 sirve para la comparación completa
# de las cuatro versiones, donde Python secuencial sí alcanza a terminar.
RESOLUCIONES = (500, 2000, 4000, 6000)


# Escala de grises, recorte cuadrado central y reducción a cada resolución con
# INTER_AREA, que promedia los píxeles de origen. Se ejecuta una vez, fuera de las
# mediciones; OpenCV solo se usa aquí y en a_png.py.
def preparar(origen, destino):
    gris = cv2.imread(str(origen), cv2.IMREAD_GRAYSCALE)
    if gris is None:
        print(f"No se pudo leer {origen}", file=sys.stderr)
        return 1

    alto, ancho = gris.shape
    lado = min(ancho, alto)
    if lado < max(RESOLUCIONES):
        print(f"La imagen debe medir al menos 6000x6000; su recorte mide {lado}x{lado}",
              file=sys.stderr)
        return 1

    x = (ancho - lado) // 2
    y = (alto - lado) // 2
    recorte = gris[y:y + lado, x:x + lado]
    print(f"Origen: {ancho}x{alto}, recorte: {lado}x{lado} en ({x}, {y})")

    Path(destino).mkdir(parents=True, exist_ok=True)
    for resolucion in RESOLUCIONES:
        if resolucion == lado:
            salida = recorte
        else:
            salida = cv2.resize(recorte, (resolucion, resolucion), interpolation=cv2.INTER_AREA)

        ruta = Path(destino) / f"img_{resolucion}.pgm"
        if not cv2.imwrite(str(ruta), salida):
            print(f"No se pudo escribir {ruta}", file=sys.stderr)
            return 1
        print(f"Escrita {ruta}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: preparar_imagenes.py <imagen_origen> <carpeta_salida>", file=sys.stderr)
        sys.exit(1)
    sys.exit(preparar(sys.argv[1], sys.argv[2]))
