package parallel;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public class ProcesadorParalelo {
    public long totalPalabras = 0;
    public long contadorPalabra = 0;
    public long contadorSecuencia = 0;

    public void procesar(String texto, String palabra, String secuencia, int numHilos) {
        totalPalabras = 0;
        contadorPalabra = 0;
        contadorSecuencia = 0;

        int n = texto.length();
        int[] cortes = new int[numHilos + 1];
        cortes[0] = 0;
        cortes[numHilos] = n;

        //  Dividir el documento en partes evitando cortar palabras entre bloques
        int tamanoBloque = n / numHilos;
        for (int i = 1; i < numHilos; i++) {
            int corte = i * tamanoBloque;
            while (corte < n && Character.isLetterOrDigit(texto.charAt(corte))) {
                corte++;
            }
            cortes[i] = corte;
        }

        AnalizadorThread[] hilos = new AnalizadorThread[numHilos];
        for (int i = 0; i < numHilos; i++) {
            hilos[i] = new AnalizadorThread(texto, cortes[i], cortes[i + 1], palabra, secuencia);
            hilos[i].start(); // Inicia el hilo
        }

        //  Esperar que terminen (join) y combinar resultados parciales
        for (int i = 0; i < numHilos; i++) {
            try {
                hilos[i].join(); // Espera la finalización
                totalPalabras += hilos[i].totalPalabras;
                contadorPalabra += hilos[i].contadorPalabra;
                contadorSecuencia += hilos[i].contadorSecuencia;
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }

    public static void main(String[] args) throws IOException {
        String rutaArchivo = "artifacts/don_quijote.txt";
        String palabra = "Quijote";
        String secuencia = "ción";
        int[] casosHilos = {2, 4};

        System.out.println("=================================================");
        System.out.println("     PROCESAMIENTO PARALELO (JAVA THREADS)");
        System.out.println("=================================================");

        System.out.println("Cargando archivo en memoria...");
        String textoOriginal = Files.readString(Path.of(rutaArchivo));
        if (textoOriginal.startsWith("\uFEFF")) {
            textoOriginal = textoOriginal.substring(1);
        }

        ProcesadorParalelo procesador = new ProcesadorParalelo();

        // 1. Texto Original
        System.out.println("\n--- 1. TEXTO ORIGINAL ---");
        for (int h : casosHilos) {
            long t0 = System.nanoTime();
            procesador.procesar(textoOriginal, palabra, secuencia, h);
            long t1 = System.nanoTime();
            double tiempoMs = (t1 - t0) / 1_000_000.0;

            System.out.println("\n[Con " + h + " hilos]:");
            System.out.println("Total de palabras: " + procesador.totalPalabras);
            System.out.println("Apariciones de '" + palabra + "': " + procesador.contadorPalabra);
            System.out.println("Palabras con '" + secuencia + "': " + procesador.contadorSecuencia);
            System.out.printf("Tiempo de procesamiento: %.2f ms\n", tiempoMs);
        }

        // 2. Texto Ampliado (50x)
        System.out.println("\n--- 2. TEXTO AMPLIADO (50x) ---");
        String textoAmpliado = textoOriginal.repeat(50);
        for (int h : casosHilos) {
            long t0 = System.nanoTime();
            procesador.procesar(textoAmpliado, palabra, secuencia, h);
            long t1 = System.nanoTime();
            double tiempoMs = (t1 - t0) / 1_000_000.0;

            System.out.println("\n[Con " + h + " hilos]:");
            System.out.println("Total de palabras: " + procesador.totalPalabras);
            System.out.println("Apariciones de '" + palabra + "': " + procesador.contadorPalabra);
            System.out.println("Palabras con '" + secuencia + "': " + procesador.contadorSecuencia);
            System.out.printf("Tiempo de procesamiento: %.2f ms\n", tiempoMs);
        }
    }
}
