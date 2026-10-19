package ups.edu.ec;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Detección de bordes (Sobel) por bloques: comparativa secuencial vs paralelo con hilos de Java.
 *
 * Conjuntos de datos:
 *   - Original:                 la imagen una vez (64 bloques).
 *   - Ampliado (Lote mayor/5x): la misma imagen 5 veces como lote (320 bloques).
 *
 * Cada configuración (1, 2 y 4 hilos) se ejecuta REPETICIONES veces y se promedia.
 * Solo se mide el procesamiento de los bloques; el guardado de la imagen va fuera de la medición.
 */
public class Main {

    // Imagen de Unsplash (6000x6000, recortada a cuadrado por el CDN de Unsplash). Enlace exacto de descarga:
    //   https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=6000&h=6000&fit=crop&fm=jpg&q=90
    private static final String RUTA_IMAGEN = "src/imgs/foto.jpg";
    private static final String CARPETA_SALIDA = "src/imgs/salida";

    private static final int[] HILOS = {1, 2, 4, 6};
    private static final int REPETICIONES = 3;
    private static final int CALENTAMIENTO = 3;
    private static final int COPIAS_LOTE = 25;

    record Corrida(double totalMs, TaskWorker[] hilos, byte[][] mascaras) {}
    record Fila(String conjunto, int hilos, double promedioMs) {}

    public static void main(String[] args) throws Exception {
        Locale.setDefault(Locale.of("es", "EC"));   // decimales con coma

        ImageProcessorJava procesador = new ImageProcessorJava(RUTA_IMAGEN);

        // Calentamiento de la JVM (corridas no medidas)
        for (int i = 0; i < CALENTAMIENTO; i++) {
            ejecutar(procesador, crearMascaras(procesador, 1), 1);
        }

        String res = procesador.getLado() + "x" + procesador.getLado();
        List<Fila> filas = new ArrayList<>();
        filas.addAll(medir(procesador, "Original (" + res + ")", "original", 1));
        filas.addAll(medir(procesador, "Ampliado (Lote mayor / " + COPIAS_LOTE + "x)", "lote5x", COPIAS_LOTE));
        imprimirTabla(filas);
    }

    /**
     * Ejecuta todas las tareas (copias x bloques):
     *   - Secuencial (1 hilo): run() se llama directamente en el hilo principal.
     *   - Paralelo  (N hilos): se lanzan N TaskWorker con start() y se esperan con join().
     * Las máscaras se reutilizan entre corridas: cada corrida sobrescribe todos sus píxeles.
     */
    private static Corrida ejecutar(ImageProcessorJava procesador, byte[][] mascaras, int numHilos) throws Exception {
        TaskWorker[] hilos = new TaskWorker[numHilos];
        for (int i = 0; i < numHilos; i++) {
            hilos[i] = new TaskWorker(i, numHilos, procesador, mascaras);
        }

        System.gc();   // evita que basura de corridas previas interrumpa esta medición
        long inicio = System.nanoTime();
        if (numHilos == 1) {
            hilos[0].run();
        } else {
            for (TaskWorker h : hilos) h.start();
            for (TaskWorker h : hilos) h.join();
        }
        long fin = System.nanoTime();

        return new Corrida(aMs(fin - inicio), hilos, mascaras);
    }

    private static List<Fila> medir(ImageProcessorJava procesador, String conjunto, String prefijo, int copias)
            throws Exception {
        titulo("CONJUNTO: " + conjunto + " — " + copias * procesador.getBloques().size() + " bloques");
        List<Fila> filas = new ArrayList<>();
        byte[][] mascaras = crearMascaras(procesador, copias);

        for (int hilos : HILOS) {
            System.out.println("\n  ▶ " + (hilos == 1 ? "SECUENCIAL (1 hilo)" : "PARALELO (" + hilos + " hilos)"));
            double suma = 0;

            for (int rep = 1; rep <= REPETICIONES; rep++) {
                Corrida corrida = ejecutar(procesador, mascaras, hilos);
                suma += corrida.totalMs();

                if (rep == 1) {
                    imprimirResumen(corrida);
                    procesador.guardar(corrida.mascaras()[0],
                            CARPETA_SALIDA + "/bordes_" + prefijo + "_" + hilos + "h.png");
                }
            }
            double promedio = suma / REPETICIONES;
            System.out.printf("    Promedio: %.2f ms%n", promedio);
            filas.add(new Fila(conjunto, hilos, promedio));
        }
        return filas;
    }

    /** Una línea por hilo y estadísticas de los bloques (sin log por cada bloque). */
    private static void imprimirResumen(Corrida corrida) {
        String linea = "    " + "-".repeat(48);
        boolean secuencial = corrida.hilos().length == 1;
        System.out.println(linea);
        System.out.printf("    %-12s | %-10s | %-18s%n", "Hilo", "Bloques", "Tiempo (ms)");
        System.out.println(linea);

        long min = Long.MAX_VALUE, max = 0, suma = 0;
        int total = 0;
        for (TaskWorker h : corrida.hilos()) {
            System.out.printf("    %-12s | %-10d | %18.2f%n", secuencial ? "main" : h.getName(),
                    h.getTiemposBloque().size(), aMs(h.getTiempoTotalNs()));
            for (long t : h.getTiemposBloque()) {
                min = Math.min(min, t);
                max = Math.max(max, t);
                suma += t;
                total++;
            }
        }
        System.out.println(linea);
        System.out.printf("    Bloques: %d | min %.2f ms | prom %.2f ms | max %.2f ms%n",
                total, aMs(min), aMs(suma) / total, aMs(max));
    }

    /** Speedup = tiempo secuencial / tiempo con N hilos (dentro del mismo conjunto). */
    private static void imprimirTabla(List<Fila> filas) {
        titulo("TABLA COMPARATIVA DE RENDIMIENTO");
        String formato = "| %-32s | %-14s | %-20s | %-21s |%n";
        System.out.printf(formato, "Conjunto de datos / Resolución", "Hilos",
                "Tiempo promedio (ms)", "Aceleración (Speedup)");
        System.out.println("|" + "-".repeat(34) + "|" + "-".repeat(16) + "|"
                + "-".repeat(22) + "|" + "-".repeat(23) + "|");

        double base = 0;
        for (Fila f : filas) {
            if (f.hilos() == 1) base = f.promedioMs();
            System.out.printf(formato, f.conjunto(),
                    f.hilos() == 1 ? "1 (secuencial)" : String.valueOf(f.hilos()),
                    String.format("%.2f", f.promedioMs()),
                    String.format("%.2f", base / f.promedioMs()));
        }
        System.out.println();
    }

    /** Una máscara vacía por cada imagen del lote. */
    private static byte[][] crearMascaras(ImageProcessorJava procesador, int copias) {
        byte[][] mascaras = new byte[copias][];
        for (int c = 0; c < copias; c++) {
            mascaras[c] = procesador.crearMascara();
        }
        return mascaras;
    }

    private static double aMs(long nanos) {
        return nanos / 1_000_000.0;
    }

    private static void titulo(String texto) {
        System.out.println("\n" + "=".repeat(70));
        System.out.println("  " + texto);
        System.out.println("=".repeat(70));
    }
}
