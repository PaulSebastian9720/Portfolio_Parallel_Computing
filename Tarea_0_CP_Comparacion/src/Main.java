import parallel.ProcesadorParalelo;
import secuencial.ProcesadorSecuencial;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public class Main {

    public static void main(String[] args) throws IOException {

        String palabra = "Quijote";
        String secuencia = "ción";

        // Cargar archivo completo en memoria antes de medir tiempos
        System.out.println("Cargando archivo en memoria...");
        String textoOriginal = Files.readString(Path.of("artifacts/don_quijote.txt"));
        if (textoOriginal.startsWith("\uFEFF")) {
            textoOriginal = textoOriginal.substring(1);
        }

        // -------------------------------------------------------------
        // ACTIVIDAD 1: Procesamiento Secuencial (1 hilo)
        // -------------------------------------------------------------
        System.out.println("\n=== 1. IMPLEMENTACIÓN SECUENCIAL ===");
        ProcesadorSecuencial sec = new ProcesadorSecuencial();
        long t0Sec = System.nanoTime();
        sec.procesar(textoOriginal, palabra, secuencia);
        long t1Sec = System.nanoTime();

        System.out.println("Total de palabras: " + sec.totalPalabras);
        System.out.println("Apariciones de '" + palabra + "': " + sec.contadorPalabra);
        System.out.println("Palabras con '" + secuencia + "': " + sec.contadorSecuencia);
        System.out.printf("Tiempo: %.2f ms\n", (t1Sec - t0Sec) / 1_000_000.0);

        // -------------------------------------------------------------
        // ACTIVIDAD 2: Procesamiento Paralelo (2 y 4 hilos)
        // -------------------------------------------------------------
        System.out.println("\n=== 2. IMPLEMENTACIÓN PARALELA ===");
        int[] hilosCasos = {2, 4};
        for (int h : hilosCasos) {
            ProcesadorParalelo par = new ProcesadorParalelo();
            long t0 = System.nanoTime();
            par.procesar(textoOriginal, palabra, secuencia, h);
            long t1 = System.nanoTime();

            System.out.println("\nResultados con " + h + " hilos:");
            System.out.println("Total de palabras: " + par.totalPalabras);
            System.out.println("Apariciones de '" + palabra + "': " + par.contadorPalabra);
            System.out.println("Palabras con '" + secuencia + "': " + par.contadorSecuencia);
            System.out.printf("Tiempo: %.2f ms\n", (t1 - t0) / 1_000_000.0);

            // Comprobar coincidencia con la versión secuencial
            boolean coincide = (par.totalPalabras == sec.totalPalabras &&
                                par.contadorPalabra == sec.contadorPalabra &&
                                par.contadorSecuencia == sec.contadorSecuencia);
            System.out.println("¿Coincide con secuencial?: " + (coincide ? "SÍ [OK]" : "NO [ERROR]"));
        }

        // -------------------------------------------------------------
        // ACTIVIDAD 3: Evaluación del Rendimiento (5 repeticiones)
        // -------------------------------------------------------------
        System.out.println("\n=== 3. EVALUACIÓN DEL RENDIMIENTO (5 repeticiones promedio) ===");

        // Versión ampliada repitiendo el texto (ej. 50 veces)
        System.out.println("Generando versión ampliada del texto (50x)...");
        String textoAmpliado = textoOriginal.repeat(50);

        // Medir tiempos para Original y Ampliado
        double tOriginalSec = medirTiempoPromedio(textoOriginal, palabra, secuencia, 1);
        double tOriginal2   = medirTiempoPromedio(textoOriginal, palabra, secuencia, 2);
        double tOriginal4   = medirTiempoPromedio(textoOriginal, palabra, secuencia, 4);

        double tAmpliadoSec = medirTiempoPromedio(textoAmpliado, palabra, secuencia, 1);
        double tAmpliado2   = medirTiempoPromedio(textoAmpliado, palabra, secuencia, 2);
        double tAmpliado4   = medirTiempoPromedio(textoAmpliado, palabra, secuencia, 4);

        // Imprimir la tabla solicitada en la consigna
        System.out.println("\n" + "=".repeat(68));
        System.out.printf("| %-18s | %-14s | %-14s | %-11s |\n",
                "Conjunto de datos", "Hilos", "Tiempo prom(ms)", "Aceleración");
        System.out.println("|" + "-".repeat(20) + "|" + "-".repeat(16) + "|" + "-".repeat(16) + "|" + "-".repeat(13) + "|");

        imprimirFila("Original", "1 (secuencial)", tOriginalSec, 1.00);
        imprimirFila("Original", "2",              tOriginal2,   tOriginalSec / tOriginal2);
        imprimirFila("Original", "4",              tOriginal4,   tOriginalSec / tOriginal4);

        imprimirFila("Ampliado", "1 (secuencial)", tAmpliadoSec, 1.00);
        imprimirFila("Ampliado", "2",              tAmpliado2,   tAmpliadoSec / tAmpliado2);
        imprimirFila("Ampliado", "4",              tAmpliado4,   tAmpliadoSec / tAmpliado4);
        System.out.println("=".repeat(68));
    }

    private static double medirTiempoPromedio(String texto, String palabra, String secuencia, int hilos) {
        long sumaNanos = 0;
        int repeticiones = 5;

        for (int r = 0; r < repeticiones; r++) {
            long inicio = System.nanoTime();
            if (hilos == 1) {
                new ProcesadorSecuencial().procesar(texto, palabra, secuencia);
            } else {
                new ProcesadorParalelo().procesar(texto, palabra, secuencia, hilos);
            }
            long fin = System.nanoTime();
            sumaNanos += (fin - inicio);
        }

        return (sumaNanos / (double) repeticiones) / 1_000_000.0;
    }

    private static void imprimirFila(String dataset, String hilos, double tiempoMs, double aceleracion) {
        System.out.printf("| %-18s | %-14s | %11.2f ms | %11.2f |\n",
                dataset, hilos, tiempoMs, aceleracion);
    }
}