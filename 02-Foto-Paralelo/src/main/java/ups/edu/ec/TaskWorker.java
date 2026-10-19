package ups.edu.ec;

import org.bytedeco.opencv.opencv_core.Mat;

import java.util.ArrayList;
import java.util.List;

/**
 * Hilo de trabajo (extends Thread). Reparto cíclico de las tareas:
 *   el hilo k procesa las tareas k, k + N, k + 2N, ...   (N = número de hilos)
 *
 * Una tarea es un bloque de una copia de la imagen:
 *   copia = tarea / bloquesPorImagen,  bloque = tarea % bloquesPorImagen
 *
 * Cada bloque escribe en una zona distinta de la máscara, por eso no hace falta sincronizar.
 */
public class TaskWorker extends Thread {

    private final int id;
    private final int numHilos;
    private final ImageProcessor procesador;
    private final Mat[] mascaras;

    private final List<Long> tiemposBloque = new ArrayList<>();   // nanos de cada bloque procesado
    private long tiempoTotalNs;

    public TaskWorker(int id, int numHilos, ImageProcessor procesador, Mat[] mascaras) {
        super("Hilo-" + (id + 1));
        this.id = id;
        this.numHilos = numHilos;
        this.procesador = procesador;
        this.mascaras = mascaras;
    }

    @Override
    public void run() {
        List<ImageProcessor.Block> bloques = procesador.getBloques();
        int totalTareas = mascaras.length * bloques.size();

        long inicioHilo = System.nanoTime();
        for (int tarea = id; tarea < totalTareas; tarea += numHilos) {
            int copia = tarea / bloques.size();
            int bloque = tarea % bloques.size();

            long inicio = System.nanoTime();
            procesador.procesarBloque(bloques.get(bloque), mascaras[copia]);
            tiemposBloque.add(System.nanoTime() - inicio);
        }
        tiempoTotalNs = System.nanoTime() - inicioHilo;
    }

    public List<Long> getTiemposBloque() { return tiemposBloque; }
    public long getTiempoTotalNs()       { return tiempoTotalNs; }
}
