package ups.edu.ec;

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.Size;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

/**
 * Pipeline de detección de bordes con OpenCV (JavaCV).
 *
 * PASO 1 — Carga: imagen cuadrada de al menos 6000x6000 y lado divisible entre DIVISIONES.
 * PASO 2 — Descomposición: ladoBloque = lado / DIVISIONES (6000 / 8 = 750 px). Cada bloque se
 *          lee con HALO px extra de sus vecinos (Gauss 5x5 = 2 px + Sobel 3x3 = 1 px) para que
 *          no aparezcan líneas falsas en las uniones.
 * PASO 3 — Por bloque: grises -> GaussianBlur -> Sobel (Gx, Gy) -> threshold.
 * PASO 4 — Reconstrucción: cada bloque copia su núcleo (sin halo) a su posición en la máscara final.
 */
public class ImageProcessor {

    public static final int LADO_MINIMO = 6000;
    public static final int DIVISIONES = 8;
    public static final int HALO = 3;
    public static final int UMBRAL = 50;

    /** Bloque lógico: índice, esquina superior izquierda y lado (sin halo). */
    public record Block(int indice, int x, int y, int lado) {}

    private final Mat imagen;
    private final int lado;
    private final List<Block> bloques = new ArrayList<>();

    public ImageProcessor(String ruta) {
        imagen = imread(ruta);
        if (imagen.empty()) {
            throw new IllegalArgumentException("No se pudo leer la imagen: " + ruta);
        }
        int ancho = imagen.cols(), alto = imagen.rows();
        if (ancho != alto || ancho < LADO_MINIMO || ancho % DIVISIONES != 0) {
            throw new IllegalArgumentException("Se esperaba una imagen cuadrada de al menos "
                    + LADO_MINIMO + "x" + LADO_MINIMO + " con lado divisible entre " + DIVISIONES
                    + "; se recibió " + ancho + "x" + alto);
        }
        lado = ancho;

        int ladoBloque = lado / DIVISIONES;
        for (int fila = 0; fila < DIVISIONES; fila++)
            for (int col = 0; col < DIVISIONES; col++)
                bloques.add(new Block(bloques.size(), col * ladoBloque, fila * ladoBloque, ladoBloque));
    }

    public int getLado()            { return lado; }
    public int getLadoBloque()      { return lado / DIVISIONES; }
    public List<Block> getBloques() { return bloques; }

    /** Máscara vacía (1 canal) donde se ensamblan los bloques procesados. */
    public Mat crearMascara() {
        return new Mat(lado, lado, CV_8UC1);
    }

    /** Aplica el pipeline a un bloque (con halo) y copia el núcleo a la máscara final. */
    public void procesarBloque(Block b, Mat salida) {
        int x0 = Math.max(0, b.x() - HALO);
        int y0 = Math.max(0, b.y() - HALO);
        int x1 = Math.min(lado, b.x() + b.lado() + HALO);
        int y1 = Math.min(lado, b.y() + b.lado() + HALO);

        // PointerScope libera la memoria nativa de todas las Mat temporales al cerrar
        try (PointerScope scope = new PointerScope()) {
            Mat region = new Mat(imagen, new Rect(x0, y0, x1 - x0, y1 - y0));
            Mat gris = new Mat(), gx = new Mat(), gy = new Mat();
            Mat absX = new Mat(), absY = new Mat(), magnitud = new Mat(), bordes = new Mat();

            cvtColor(region, gris, COLOR_BGR2GRAY);
            GaussianBlur(gris, gris, new Size(5, 5), 0);
            Sobel(gris, gx, CV_16S, 1, 0);                    // cambio de intensidad horizontal
            Sobel(gris, gy, CV_16S, 0, 1);                    // cambio de intensidad vertical
            convertScaleAbs(gx, absX);
            convertScaleAbs(gy, absY);
            addWeighted(absX, 0.5, absY, 0.5, 0, magnitud);
            threshold(magnitud, bordes, UMBRAL, 255, THRESH_BINARY);

            // Descartar el halo: copiar solo el núcleo del bloque a su lugar en la salida
            Mat nucleo = new Mat(bordes, new Rect(b.x() - x0, b.y() - y0, b.lado(), b.lado()));
            nucleo.copyTo(new Mat(salida, new Rect(b.x(), b.y(), b.lado(), b.lado())));
        }
    }

    public void guardar(Mat mascara, String archivo) {
        new File(archivo).getParentFile().mkdirs();
        imwrite(archivo, mascara);
    }
}
