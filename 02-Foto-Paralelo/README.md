# 02-Foto-Paralelo

Detección de bordes con el filtro Sobel sobre una imagen de 6000x6000 píxeles, comparando el procesamiento secuencial (1 hilo) contra el paralelo (2, 4 y 6 hilos). Al final se muestra una tabla con los tiempos promedio y la aceleración (speedup).

Imagen usada (Unsplash):
https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=6000&h=6000&fit=crop&fm=jpg&q=90

## Requisitos

- Java 21
- Maven

## Cómo ejecutar

Desde la carpeta del proyecto:

```bash
mvn compile exec:java
```

La imagen de entrada debe estar en `src/imgs/foto.jpg`. Las imágenes de bordes resultantes se guardan en `src/imgs/salida/` y el informe está en `informe/informe.html`.
