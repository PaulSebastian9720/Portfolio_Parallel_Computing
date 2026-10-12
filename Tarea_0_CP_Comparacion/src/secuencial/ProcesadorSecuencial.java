package secuencial;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public class ProcesadorSecuencial {
    public long totalPalabras = 0;
    public long contadorPalabra = 0;
    public long contadorSecuencia = 0;

    public void procesar(String texto, String palabra, String secuencia) {
        totalPalabras = 0;
        contadorPalabra = 0;
        contadorSecuencia = 0;

        String secMin = secuencia.toLowerCase();
        int n = texto.length();
        int i = 0;

        while (i < n) {
            // Saltar espacios y signos de puntuación
            while (i < n && !Character.isLetterOrDigit(texto.charAt(i))) {
                i++;
            }
            if (i >= n) break;

            int inicio = i;
            // Avanzar hasta el final de la palabra
            while (i < n && Character.isLetterOrDigit(texto.charAt(i))) {
                i++;
            }

            String p = texto.substring(inicio, i);
            totalPalabras++;

            // Comparar palabra exacta ignorando mayúsculas
            if (p.equalsIgnoreCase(palabra)) {
                contadorPalabra++;
            }

            // Verificar si contiene la secuencia ignorando mayúsculas
            if (p.toLowerCase().contains(secMin)) {
                contadorSecuencia++;
            }
        }
    }

    public static void main(String[] args) throws IOException {
        String rutaArchivo = "artifacts/don_quijote.txt";
        String palabra = "Quijote";
        String secuencia = "ción";

        System.out.println("=================================================");
        System.out.println("       PROCESAMIENTO SECUENCIAL (1 HILO)");
        System.out.println("=================================================");

        System.out.println("Cargando archivo en memoria...");
        String textoOriginal = Files.readString(Path.of(rutaArchivo));
        if (textoOriginal.startsWith("\uFEFF")) {
            textoOriginal = textoOriginal.substring(1);
        }

        ProcesadorSecuencial procesador = new ProcesadorSecuencial();

        // 1. Texto Original
        System.out.println("\n--- 1. TEXTO ORIGINAL ---");
        long t0 = System.nanoTime();
        procesador.procesar(textoOriginal, palabra, secuencia);
        long t1 = System.nanoTime();
        double tiempoOriginalMs = (t1 - t0) / 1_000_000.0;

        System.out.println("Total de palabras: " + procesador.totalPalabras);
        System.out.println("Apariciones de '" + palabra + "': " + procesador.contadorPalabra);
        System.out.println("Palabras con '" + secuencia + "': " + procesador.contadorSecuencia);
        System.out.printf("Tiempo de procesamiento: %.2f ms\n", tiempoOriginalMs);

        // 2. Texto Ampliado (50x)
        System.out.println("\n--- 2. TEXTO AMPLIADO (50x) ---");
        String textoAmpliado = textoOriginal.repeat(50);
        long t2 = System.nanoTime();
        procesador.procesar(textoAmpliado, palabra, secuencia);
        long t3 = System.nanoTime();
        double tiempoAmpliadoMs = (t3 - t2) / 1_000_000.0;

        System.out.println("Total de palabras: " + procesador.totalPalabras);
        System.out.println("Apariciones de '" + palabra + "': " + procesador.contadorPalabra);
        System.out.println("Palabras con '" + secuencia + "': " + procesador.contadorSecuencia);
        System.out.printf("Tiempo de procesamiento: %.2f ms\n", tiempoAmpliadoMs);
    }
}
