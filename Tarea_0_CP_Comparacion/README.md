# Actividad Práctica 01: Procesamiento Secuencial y Paralelo de Textos con Java Threads

**Materia:** Computación Paralela 
**Institución:** Universidad Politécnica Salesiana  
**Estudiante:** Paul Sebastian Naspud Vivar  

---

## Descripción

Aplicación en Java que analiza y compara el procesamiento secuencial y paralelo del libro *Don Quijote de la Mancha* (en formato TXT de Project Gutenberg). 

El programa extrae tres métricas sin distinción de mayúsculas ni minúsculas:
1. Número total de palabras.
2. Número de apariciones de la palabra `"Quijote"`.
3. Número de palabras que contienen la secuencia `"ción"`.

Para la versión paralela se utiliza directamente la clase `Thread` de Java (`start()` y `join()`), implementando descomposición de datos con ajuste de límites para evitar cortar palabras entre bloques.

---

## Requisitos

- **Java JDK:** 11 o superior (probado en OpenJDK 21).

---

## Compilación

Desde la raíz del proyecto, compilar todos los archivos `.java` en la carpeta `out`:

```bash
javac -d out $(find src -name "*.java")
```

---

## Ejecución

### 1. Programa Principal (Completo)
Ejecuta la prueba secuencial, la paralela (con 2 y 4 hilos), verifica que los resultados coincidan al 100% y realiza las 5 repeticiones para generar la tabla comparativa de tiempos y aceleración (*Speedup*):

```bash
java -cp out Main
```

### 2. Procesamiento Secuencial (Independiente)
Ejecuta únicamente el análisis con 1 hilo sobre el texto original y el texto ampliado:

```bash
java -cp out secuencial.ProcesadorSecuencial
```

### 3. Procesamiento Paralelo (Independiente)
Ejecuta el análisis concurrente con 2 y 4 hilos sobre el texto original y el texto ampliado:

```bash
java -cp out parallel.ProcesadorParalelo
```

---


