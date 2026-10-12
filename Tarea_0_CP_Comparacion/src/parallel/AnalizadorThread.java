package parallel;

public class AnalizadorThread extends Thread {
    private String texto;
    private int inicio;
    private int fin;
    private String palabra;
    private String secuencia;

    public long totalPalabras = 0;
    public long contadorPalabra = 0;
    public long contadorSecuencia = 0;

    public AnalizadorThread(String texto, int inicio, int fin, String palabra, String secuencia) {
        this.texto = texto;
        this.inicio = inicio;
        this.fin = fin;
        this.palabra = palabra;
        this.secuencia = secuencia;
    }

    @Override
    public void run() {
        String secMin = secuencia.toLowerCase();
        int i = inicio;

        while (i < fin) {
            while (i < fin && !Character.isLetterOrDigit(texto.charAt(i))) {
                i++;
            }
            if (i >= fin) break;

            int startPalabra = i;
            while (i < fin && Character.isLetterOrDigit(texto.charAt(i))) {
                i++;
            }

            String p = texto.substring(startPalabra, i);
            totalPalabras++;

            if (p.equalsIgnoreCase(palabra)) {
                contadorPalabra++;
            }

            if (p.toLowerCase().contains(secMin)) {
                contadorSecuencia++;
            }
        }
    }
}
