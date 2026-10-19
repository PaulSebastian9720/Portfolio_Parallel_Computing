package ups.edu.ec;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Pipeline de detección de bordes
 * Misma interfaz que ImageProcessor, pero la máscara es un byte[] (1 byte por píxel: 0 o 255).
 *
 * PASO 1 — Carga: imagen cuadrada de al menos 6000x6000 y lado divisible entre DIVISIONES.
 * PASO 2 — Descomposición: ladoBloque = lado / DIVISIONES (6000 / 8 = 750 px). Cada bloque se
 *          lee con HALO px extra de sus vecinos (Gauss 5x5 = 2 px + Sobel 3x3 = 1 px) para que
 *          no aparezcan líneas falsas en las uniones.
 * PASO 3 — Por bloque: grises -> Gauss 5x5 -> Sobel (Gx, Gy) -> umbral.
 * PASO 4 — Reconstrucción: cada bloque escribe solo su núcleo (sin halo) en su posición de la máscara.
 */
public class ImageProcessorJava {

    public static final int LADO_MINIMO = 6000;
    public static final int DIVISIONES = 8;
    public static final int HALO = 3;
    public static final int UMBRAL = 50;

    /** Kernel Gaussiano 1D [1 4 6 4 1] / 16  */
    private static final int[] GAUSS = {1, 4, 6, 4, 1};

    /** Bloque lógico: índice, esquina superior izquierda y lado  */
    public record Block(int indice, int x, int y, int lado) {}

    private final byte[] bgr;   // píxeles originales: B, G, R por píxel (solo lectura)
    private final int lado;
    private final List<Block> bloques = new ArrayList<>();

    public ImageProcessorJava(String ruta) throws IOException {
        BufferedImage imagen = ImageIO.read(new File(ruta));
        if (imagen == null) {
            throw new IllegalArgumentException("No se pudo leer la imagen: " + ruta);
        }
        int ancho = imagen.getWidth(), alto = imagen.getHeight();
        if (ancho != alto || ancho < LADO_MINIMO || ancho % DIVISIONES != 0) {
            throw new IllegalArgumentException("Se esperaba una imagen cuadrada de al menos "
                    + LADO_MINIMO + "x" + LADO_MINIMO + " con lado divisible entre " + DIVISIONES
                    + "; se recibió " + ancho + "x" + alto);
        }
        // Formato 3 bytes por píxel (B, G, R) para leer los píxeles directamente del arreglo
        if (imagen.getType() != BufferedImage.TYPE_3BYTE_BGR) {
            BufferedImage copia = new BufferedImage(ancho, alto, BufferedImage.TYPE_3BYTE_BGR);
            copia.getGraphics().drawImage(imagen, 0, 0, null);
            imagen = copia;
        }
        bgr = ((DataBufferByte) imagen.getRaster().getDataBuffer()).getData();
        lado = ancho;

        int ladoBloque = lado / DIVISIONES;
        for (int fila = 0; fila < DIVISIONES; fila++)
            for (int col = 0; col < DIVISIONES; col++)
                bloques.add(new Block(bloques.size(), col * ladoBloque, fila * ladoBloque, ladoBloque));
    }

    public int getLado()            { return lado; }
    public int getLadoBloque()      { return lado / DIVISIONES; }
    public List<Block> getBloques() { return bloques; }

    /** Máscara vacía (1 byte por píxel) donde se ensamblan los bloques procesados. */
    public byte[] crearMascara() {
        return new byte[lado * lado];
    }

    /** Aplica el pipeline a un bloque (con halo) y escribe el núcleo en la máscara final. */
    public void procesarBloque(Block b, byte[] salida) {
        int x0 = Math.max(0, b.x() - HALO);
        int y0 = Math.max(0, b.y() - HALO);
        int x1 = Math.min(lado, b.x() + b.lado() + HALO);
        int y1 = Math.min(lado, b.y() + b.lado() + HALO);
        int w = x1 - x0, h = y1 - y0;

        int[] gris = escalaDeGrises(x0, y0, w, h);
        int[] suave = suavizar(gris, w, h);
        sobelYUmbral(suave, w, h, b, x0, y0, salida);
    }

    /** Luminancia: 0.299 R + 0.587 G + 0.114 B */
    private int[] escalaDeGrises(int x0, int y0, int w, int h) {
        int[] gris = new int[w * h];
        for (int y = 0; y < h; y++) {
            int i = ((y0 + y) * lado + x0) * 3;
            for (int x = 0; x < w; x++, i += 3) {
                int azul = bgr[i] & 0xFF;
                int verde = bgr[i + 1] & 0xFF;
                int rojo = bgr[i + 2] & 0xFF;
                gris[y * w + x] = (299 * rojo + 587 * verde + 114 * azul) / 1000;
            }
        }
        return gris;
    }

    /** Suavizado Gaussiano 5x5 separable: primero horizontal, luego vertical (total / 256). */
    private static int[] suavizar(int[] gris, int w, int h) {
        int[] temp = new int[w * h];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int suma = 0;
                for (int k = -2; k <= 2; k++) {
                    suma += GAUSS[k + 2] * gris[y * w + limitar(x + k, w)];
                }
                temp[y * w + x] = suma;
            }
        }
        int[] suave = new int[w * h];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int suma = 0;
                for (int k = -2; k <= 2; k++) {
                    suma += GAUSS[k + 2] * temp[limitar(y + k, h) * w + x];
                }
                suave[y * w + x] = suma / 256;
            }
        }
        return suave;
    }

    /**
     * Sobel 3x3 sobre el núcleo del bloque (sin halo):
     *   Gx = [-1 0 1; -2 0 2; -1 0 1]   (cambio horizontal)
     *   Gy = [-1 -2 -1; 0 0 0; 1 2 1]   (cambio vertical)
     *   magnitud = 0.5 |Gx| + 0.5 |Gy|;  borde (255) si magnitud > UMBRAL, si no 0.
     */
    private void sobelYUmbral(int[] s, int w, int h, Block b, int x0, int y0, byte[] salida) {
        for (int y = b.y(); y < b.y() + b.lado(); y++) {
            int ly = y - y0;
            int arriba = limitar(ly - 1, h) * w;
            int centro = ly * w;
            int abajo = limitar(ly + 1, h) * w;

            for (int x = b.x(); x < b.x() + b.lado(); x++) {
                int lx = x - x0;
                int izq = limitar(lx - 1, w);
                int der = limitar(lx + 1, w);

                int gx = -s[arriba + izq] + s[arriba + der]
                        - 2 * s[centro + izq] + 2 * s[centro + der]
                        - s[abajo + izq] + s[abajo + der];
                int gy = -s[arriba + izq] - 2 * s[arriba + lx] - s[arriba + der]
                        + s[abajo + izq] + 2 * s[abajo + lx] + s[abajo + der];

                int magnitud = (Math.abs(gx) + Math.abs(gy)) / 2;
                salida[y * lado + x] = magnitud > UMBRAL ? (byte) 255 : 0;
            }
        }
    }

    /** Mantiene el índice dentro de [0, max-1] (replica el borde de la región). */
    private static int limitar(int v, int max) {
        return v < 0 ? 0 : (v >= max ? max - 1 : v);
    }

    /** PASO 4 — Guarda la máscara ensamblada como PNG en escala de grises. */
    public void guardar(byte[] mascara, String archivo) throws IOException {
        BufferedImage img = new BufferedImage(lado, lado, BufferedImage.TYPE_BYTE_GRAY);
        byte[] destino = ((DataBufferByte) img.getRaster().getDataBuffer()).getData();
        System.arraycopy(mascara, 0, destino, 0, mascara.length);
        new File(archivo).getParentFile().mkdirs();
        ImageIO.write(img, "png", new File(archivo));
    }
}
