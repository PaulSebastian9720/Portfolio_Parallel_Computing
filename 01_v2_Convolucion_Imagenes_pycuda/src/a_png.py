"""Convierte un PGM a PNG reducido para el informe (equivale a a_png.cpp)."""

import sys

import cv2


# Lee la imagen, opcionalmente recorta una zona y la reduce hasta lado_maximo.
# La zona se da en fracciones de 0 a 1 (x, y, lado) para que el mismo recorte caiga en
# el mismo lugar de la escena sin importar la resolución.
def a_png(entrada, salida, lado_maximo, zona=None):
    imagen = cv2.imread(entrada, cv2.IMREAD_GRAYSCALE)
    if imagen is None:
        print(f"No se pudo leer {entrada}", file=sys.stderr)
        return 1

    if zona is not None:
        alto, ancho = imagen.shape
        x, y, lado = int(zona[0] * ancho), int(zona[1] * alto), int(zona[2] * ancho)
        if x + lado > ancho or y + lado > alto:
            print("El recorte queda fuera de la imagen", file=sys.stderr)
            return 1
        imagen = imagen[y:y + lado, x:x + lado]

    escala = min(1.0, lado_maximo / max(imagen.shape))
    reducida = cv2.resize(imagen, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
    if not cv2.imwrite(salida, reducida):
        print(f"No se pudo escribir {salida}", file=sys.stderr)
        return 1
    print(f"{entrada} ({imagen.shape[1]}x{imagen.shape[0]}) -> {salida} "
          f"({reducida.shape[1]}x{reducida.shape[0]})")
    return 0


if __name__ == "__main__":
    if len(sys.argv) not in (4, 7):
        print("Uso: a_png.py <entrada> <salida.png> <lado_maximo> [x y lado]", file=sys.stderr)
        sys.exit(1)
    zona = tuple(map(float, sys.argv[4:7])) if len(sys.argv) == 7 else None
    sys.exit(a_png(sys.argv[1], sys.argv[2], int(sys.argv[3]), zona))
