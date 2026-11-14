"""Prepara las imágenes de entrada (equivale a preparar_imagenes/preparar.cpp).

Se ejecuta una vez, antes de los experimentos, y no forma parte de las mediciones. Es el
único lugar donde se usa OpenCV (solo para leer, convertir y escribir imágenes).

Uso:
    uv run src/preparar_imagenes.py imagenes/foto.jpg imagenes

Pasos: escala de grises -> recorte cuadrado central -> reducción con INTER_AREA, que
promedia los píxeles de origen y evita artefactos al reducir.
"""

import sys
from pathlib import Path

import cv2

RESOLUCIONES = (2000, 4000, 6000)


# Genera img_2000.pgm, img_4000.pgm e img_6000.pgm a partir de una foto.
# Parámetros:
#   origen   ruta de la imagen original (debe medir al menos 6000x6000 tras el recorte)
#   destino  carpeta donde se escriben los PGM
# Retorna: 0 si todo salió bien, 1 si hubo un error.
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
