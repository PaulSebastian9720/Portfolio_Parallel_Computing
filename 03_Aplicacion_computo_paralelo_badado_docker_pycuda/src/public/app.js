// La API corre en otro puerto del mismo host; ?api=http://host:puerto la cambia.
const API = new URLSearchParams(location.search).get("api")
  || `${location.protocol}//${location.hostname}:8000`;

const $ = (id) => document.getElementById(id);
const estado = { config: null, dimensiones: null, enCurso: null, urlResultado: null, urlOriginal: null, n: 0 };

const NOTAS_SALIDA = {
  recorte: "Redondea y recorta a [0, 255]. Los valores negativos quedan en 0.",
  absoluto: "Toma |valor| antes de recortar: un borde negativo también se ve.",
  umbral: "255 si el valor es ≥ T, 0 en otro caso.",
};

const fmt = (x, d = 3) => (x === null || x === undefined ? "—" : Number(x).toFixed(d));
const miles = (x) => Number(x).toLocaleString("es");

function mostrarAviso(texto) {
  $("aviso").textContent = texto;
  $("aviso").hidden = !texto;
}

async function pedirJson(ruta) {
  const r = await fetch(API + ruta);
  if (!r.ok) throw new Error(`${ruta}: HTTP ${r.status}`);
  return r.json();
}

// Estado de la GPU en la barra superior.
async function actualizarEstado() {
  try {
    const e = await pedirJson("/api/estado");
    if (!e.disponible) {
      $("e-gpu").textContent = "no disponible";
      $("e-gpu").classList.add("mal");
      mostrarAviso(`GPU no disponible: ${e.error}`);
    } else {
      const g = e.gpu;
      $("e-gpu").textContent = g.nombre;
      $("e-gpu").classList.remove("mal");
      $("e-cc").textContent = `sm ${g.capacidad}`;
      $("e-mem").textContent = `${miles(g.memoria_libre_mb)} / ${miles(g.memoria_total_mb)} MB`;
      $("e-ver").textContent = `CUDA ${g.driver_cuda} / ${g.pycuda} (CUDA ${g.cuda_pycuda})`;
      $("e-ini").textContent = `${fmt(g.ms_contexto, 1)} / ${fmt(g.ms_compilacion, 1)} / ${fmt(g.ms_calentamiento, 1)} ms`;
    }
    $("e-cola").textContent = `${e.pendientes}/${e.max_cola} · ${e.atendidas} ok · ${e.rechazadas} rech.`;
  } catch (err) {
    $("e-gpu").textContent = "sin conexión";
    $("e-gpu").classList.add("mal");
    mostrarAviso(`No se pudo contactar la API en ${API}: ${err.message}`);
  }
}

function llenarSelect(select, valores, elegido) {
  select.replaceChildren(...valores.map((v) => new Option(v, v, false, v == elegido)));
}

// Solo la media elige k (3, 9, 21); nitidez y laplaciano muestran 3 fijo. En la
// personalizada k se escribe a mano.
function actualizarTamanos() {
  const filtro = $("filtro").value;
  const personalizada = filtro === "personalizado";
  $("k").hidden = personalizada;
  $("k-texto").hidden = !personalizada;
  if (personalizada) return;
  const actual = Number($("k").value) || 3;
  const tamanos = estado.config.tamanos[filtro];
  llenarSelect($("k"), tamanos, tamanos.includes(actual) ? actual : tamanos[0]);
  $("k").disabled = filtro !== "media";
}

function kActual() {
  return Number($($("filtro").value === "personalizado" ? "k-texto" : "k").value);
}

function kValido(k) {
  return Number.isInteger(k) && k % 2 === 1 && k >= 1 && k <= estado.config.max_k_personalizada;
}

// Dibuja la máscara k×k. En los filtros fijos es de solo lectura; en la personalizada se
// puede escribir cada coeficiente.
function dibujarMatriz(valores) {
  const k = kActual();
  const editable = $("filtro").value === "personalizado";
  const tabla = $("matriz");
  tabla.classList.toggle("densa", k > 7);
  tabla.replaceChildren();
  for (let i = 0; i < k; i++) {
    const fila = tabla.insertRow();
    for (let j = 0; j < k; j++) {
      const input = document.createElement("input");
      input.type = "text";
      input.inputMode = "decimal";
      input.readOnly = !editable;
      const v = valores ? valores[i * k + j] : 0;
      input.value = Number(v.toPrecision(editable ? 8 : 6)).toString();
      input.title = String(v);
      if (i === (k >> 1) && j === (k >> 1)) input.classList.add("centro");
      input.addEventListener("input", actualizarSuma);
      fila.insertCell().append(input);
    }
  }
  $("editar-mascara").hidden = editable;
  actualizarSuma();
}

function leerMatriz() {
  return [...$("matriz").querySelectorAll("input")].map((i) => Number(i.value.replace(",", ".")));
}

function actualizarSuma() {
  let suma = 0;
  for (const input of $("matriz").querySelectorAll("input")) {
    const v = Number(input.value.replace(",", "."));
    input.classList.toggle("negativo", v < 0);
    suma += Number.isFinite(v) ? v : NaN;
  }
  $("suma-mascara").textContent = Number.isFinite(suma) ? `Σ = ${Number(suma.toFixed(6))}` : "Σ = valor no numérico";
}

function mascaraActual() {
  const filtro = $("filtro").value;
  if (filtro === "personalizado") return null;
  return estado.config.mascaras[`${filtro}_${kActual()}`];
}

function alCambiarFiltro() {
  const filtro = $("filtro").value;
  const anterior = $("filtro").dataset.anterior;
  actualizarTamanos();
  if (filtro === "personalizado") {
    if (anterior !== "personalizado") {
      const k = kActual();
      const identidad = Array(k * k).fill(0);
      identidad[(k * k) >> 1] = 1;
      dibujarMatriz(identidad);
    }
  } else {
    dibujarMatriz(mascaraActual());
  }
  // Modo de salida por defecto del filtro, como en la práctica 02.
  marcarSalida(filtro === "laplaciano" ? "absoluto" : "recorte");
  $("filtro").dataset.anterior = filtro;
  actualizarPrevision();
}

// Al cambiar k en la personalizada se conserva el centro de la matriz anterior.
function alCambiarK() {
  if ($("filtro").value === "personalizado") {
    const k = kActual();
    if (!kValido(k)) {
      mostrarAviso(`k=${$("k-texto").value} no es válido: debe ser impar entre 1 y ${estado.config.max_k_personalizada}`);
      return;
    }
    mostrarAviso("");
    const previo = leerMatriz();
    const kPrevio = Math.round(Math.sqrt(previo.length));
    const nuevos = Array(k * k).fill(0);
    const desfase = (kPrevio - k) / 2;
    for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) {
      const pi = i + desfase, pj = j + desfase;
      if (pi >= 0 && pj >= 0 && pi < kPrevio && pj < kPrevio) nuevos[i * k + j] = previo[pi * kPrevio + pj];
    }
    dibujarMatriz(nuevos);
  } else {
    dibujarMatriz(mascaraActual());
  }
}

// Pasa la máscara del filtro actual a personalizada, para modificarla. La media de 21×21
// pasa del máximo de la personalizada; se copia la media de ese tamaño máximo.
function editarComoPersonalizada() {
  let valores = mascaraActual();
  let k = kActual();
  const maximo = estado.config.max_k_personalizada;
  if (k > maximo) {
    k = maximo;
    valores = Array(k * k).fill(1 / (k * k));
  }
  $("filtro").value = "personalizado";
  $("filtro").dataset.anterior = "personalizado";
  $("k-texto").value = k;
  actualizarTamanos();
  dibujarMatriz(valores);
  actualizarPrevision();
}

function marcarSalida(modo) {
  document.querySelector(`input[name=salida][value=${modo}]`).checked = true;
  alCambiarSalida();
}

// El umbral solo existe en la salida binaria; en recorte y absoluto los valores pasan
// directamente y el control no se muestra ni se envía.
function modoSalida() {
  return document.querySelector("input[name=salida]:checked").value;
}

function alCambiarSalida() {
  const modo = modoSalida();
  $("bloque-umbral").hidden = modo !== "umbral";
  $("nota-salida").textContent = NOTAS_SALIDA[modo];
}

// Lee ancho y alto en el navegador para mostrar bloques e hilos antes de enviar. El PGM no
// lo decodifica el navegador, así que se lee su cabecera.
async function leerDimensiones(archivo) {
  const cabecera = new Uint8Array(await archivo.slice(0, 64).arrayBuffer());
  if (cabecera[0] === 0x50 && cabecera[1] === 0x35) {
    const texto = new TextDecoder("ascii").decode(cabecera).replace(/#[^\n]*\n/g, " ");
    const partes = texto.split(/\s+/);
    return { ancho: Number(partes[1]), alto: Number(partes[2]), pgm: true };
  }
  try {
    const mapa = await createImageBitmap(archivo);
    const d = { ancho: mapa.width, alto: mapa.height, pgm: false };
    mapa.close();
    return d;
  } catch {
    return null;
  }
}

async function alElegirArchivo() {
  const archivo = $("archivo").files[0];
  $("procesar").disabled = !archivo;
  liberarUrl("urlOriginal");
  $("img-original").removeAttribute("src");
  if (!archivo) return;

  estado.dimensiones = await leerDimensiones(archivo);
  const d = estado.dimensiones;
  $("info-archivo").textContent = `${archivo.name} · ${(archivo.size / 2 ** 20).toFixed(2)} MB`
    + (d ? ` · ${d.ancho}×${d.alto}` : " · no se reconoce como imagen");
  $("cap-original").textContent = d ? `${d.ancho}×${d.alto}` : "";
  if (d && !d.pgm) {
    estado.urlOriginal = URL.createObjectURL(archivo);
    $("img-original").src = estado.urlOriginal;
  }
  actualizarPrevision();
}

function actualizarPrevision() {
  const d = estado.dimensiones;
  const hilos = Number($("hilos").value);
  const lim = estado.config ? estado.config.limites : null;
  const nota = $("p-limites");
  nota.classList.remove("mal");
  nota.textContent = lim
    ? `Límites: lado ≤ ${miles(lim.max_lado)}, píxeles ≤ ${miles(lim.max_pixeles)}, archivo ≤ ${lim.max_mb} MB.`
    : "";
  if (!d) {
    for (const id of ["p-dim", "p-px", "p-bloques", "p-lanzados"]) $(id).textContent = "—";
    return;
  }
  const px = d.ancho * d.alto;
  const bloques = Math.ceil(px / hilos);
  $("p-dim").textContent = `${d.ancho} × ${d.alto}`;
  $("p-px").textContent = miles(px);
  $("p-bloques").textContent = miles(bloques);
  $("p-lanzados").textContent = `${miles(bloques * hilos)} (${miles(bloques * hilos - px)} ociosos)`;

  // Solo es un aviso: la solicitud se puede enviar igual y el servicio la rechaza.
  if (lim && (d.ancho > lim.max_lado || d.alto > lim.max_lado || px > lim.max_pixeles)) {
    nota.classList.add("mal");
    nota.textContent = `La imagen pasa los límites (lado ≤ ${miles(lim.max_lado)}, `
      + `píxeles ≤ ${miles(lim.max_pixeles)}); el servicio la rechazará.`;
  }
}

function liberarUrl(clave) {
  if (estado[clave]) URL.revokeObjectURL(estado[clave]);
  estado[clave] = null;
}

function filas(tabla, contenido) {
  tabla.innerHTML = contenido.map(([th, td, extra = ""]) =>
    `<tr${extra}><th>${th}</th>${td}</tr>`).join("");
}

function mostrarLanzamiento(r) {
  const k = r.k;
  filas($("t-lanzamiento"), [
    ["Id de solicitud", `<td>${r.id}</td>`],
    ["Formato / tamaño", `<td>${r.formato} · ${(r.bytes / 2 ** 20).toFixed(2)} MB</td>`],
    ["Ancho W", `<td class="num">${miles(r.ancho)}</td>`],
    ["Alto H", `<td class="num">${miles(r.alto)}</td>`],
    ["Píxeles W·H", `<td class="num">${miles(r.pixeles)}</td>`],
    ["Filtro / máscara", `<td>${r.filtro} ${k}×${k}</td>`],
    ["Hilos por bloque", `<td class="num">${miles(r.hilos)}</td>`],
    ["Bloques (grid X)", `<td class="num">${miles(r.bloques)}</td>`],
    ["Hilos lanzados", `<td class="num">${miles(r.hilos_lanzados)}</td>`],
    ["Hilos ociosos (último bloque)", `<td class="num">${miles(r.hilos_ociosos)}</td>`],
    ["Lecturas de vecinos por píxel", `<td class="num">${miles(k * k)}</td>`],
    ["Salida float min / max", `<td>${fmt(r.rango_float.min, 4)} / ${fmt(r.rango_float.max, 4)}</td>`],
    ["Píxeles sin calcular (NaN)", `<td class="num">${r.rango_float.sin_calcular}</td>`],
    ["Salida a 8 bits", `<td>${r.salida}${r.salida === "umbral" ? ` T=${r.umbral}${r.umbral_abs ? " |v|" : ""}` : ""}</td>`],
  ]);
}

// Tiempos con su alcance. Las etapas GPU se miden con eventos CUDA; el resto con
// perf_counter en el servidor y performance.now() en el navegador.
function mostrarTiempos(t, cliente) {
  const etapas = [
    ["Espera en cola", t.cola, "servidor: hasta que el hilo GPU la toma", false],
    ["Decodificación", t.decodificacion, "formato, gris y float32", false],
    ["CPU → GPU (HtoD)", t.htod, "evento CUDA, memoria paginable", true],
    ["Kernel", t.kernel, "evento CUDA", true],
    ["GPU → CPU (DtoH)", t.dtoh, "evento CUDA", true],
    ["Postproceso", t.postproceso, "float → 8 bits", false],
    ["Codificación PNG", t.codificacion, "incluye escritura a disco", false],
  ];
  const maximo = Math.max(...etapas.map((e) => e[1]));
  const barra = (v, gpu) =>
    `<td class="barra-t"><span class="${gpu ? "gpu" : ""}" style="width:${(100 * v / maximo).toFixed(1)}%"></span></td>`;
  filas($("t-tiempos"), [
    ...etapas.map(([n, v, a, gpu]) => [n, `<td class="num">${fmt(v)}</td>${barra(v, gpu)}<td class="alcance">${a}</td>`]),
    ["GPU total", `<td class="num">${fmt(t.gpu_total)}</td><td></td><td class="alcance">HtoD + kernel + DtoH</td>`, ' class="total"'],
    ["Servidor", `<td class="num">${fmt(t.servidor)}</td><td></td><td class="alcance">desde recibir el formulario hasta responder</td>`, ' class="total"'],
    ["Cliente", `<td class="num">${fmt(cliente)}</td><td></td><td class="alcance">envío → PNG recibido, con red</td>`, ' class="total"'],
  ]);
}

function agregarHistorial(r, cliente, error) {
  estado.n += 1;
  const fila = $("historial").insertRow(0);
  const celdas = r
    ? [estado.n, r.id.slice(0, 8), r.filtro, r.k, r.hilos, miles(r.bloques), `${r.ancho}×${r.alto}`,
      fmt(r.tiempos_ms.htod), fmt(r.tiempos_ms.kernel), fmt(r.tiempos_ms.dtoh),
      fmt(r.tiempos_ms.servidor, 1), fmt(cliente, 1), "200"]
    : [estado.n, (error.id || "").slice(0, 8), $("filtro").value, kActual(), $("hilos").value,
      "—", "—", "—", "—", "—", "—", fmt(cliente, 1), `${error.estado} ${error.error}`];
  for (const c of celdas) fila.insertCell().textContent = c;
  fila.lastChild.className = r ? "ok" : "mal";
}

async function descargarBlob(ruta) {
  const r = await fetch(API + ruta);
  if (!r.ok) throw new Error(`No se pudo descargar ${ruta}: HTTP ${r.status}`);
  return URL.createObjectURL(await r.blob());
}

// Envía la solicitud. Mientras procesa, el botón queda desactivado para no duplicarla.
async function procesar(evento) {
  evento.preventDefault();
  if (estado.enCurso) return;
  const archivo = $("archivo").files[0];
  if (!archivo) return;

  const filtro = $("filtro").value;
  const datos = new FormData();
  datos.append("imagen", archivo);
  datos.append("filtro", filtro);
  datos.append("k", filtro === "personalizado" ? $("k-texto").value : $("k").value);
  datos.append("hilos", $("hilos").value);
  datos.append("salida", modoSalida());
  if (modoSalida() === "umbral") {
    datos.append("umbral", $("umbral").value);
    datos.append("umbral_abs", $("umbral-abs").checked);
  }
  if (filtro === "personalizado") datos.append("mascara", JSON.stringify(leerMatriz()));

  mostrarAviso("");
  estado.enCurso = new AbortController();
  $("procesar").disabled = true;
  $("procesar").textContent = "Procesando…";
  document.body.classList.add("procesando");

  const inicio = performance.now();
  try {
    const r = await fetch(API + "/api/procesar", { method: "POST", body: datos, signal: estado.enCurso.signal });
    const cuerpo = await r.json().catch(() => ({ error: "respuesta_no_json", detalle: `HTTP ${r.status}` }));
    if (!r.ok) {
      const error = { ...cuerpo, estado: r.status };
      agregarHistorial(null, performance.now() - inicio, error);
      mostrarAviso(`${r.status} ${error.error}: ${error.detalle}`);
      return;
    }

    // El tiempo del cliente incluye bajar el PNG, no solo la respuesta JSON.
    const urlResultado = await descargarBlob(cuerpo.url_resultado);
    const cliente = performance.now() - inicio;
    liberarUrl("urlResultado");
    estado.urlResultado = urlResultado;
    $("img-resultado").src = urlResultado;
    $("cap-resultado").textContent = `${cuerpo.filtro} ${cuerpo.k}×${cuerpo.k} · ${cuerpo.hilos} hilos`;
    $("descargar").href = urlResultado;
    $("descargar").download = `${cuerpo.filtro}_${cuerpo.k}_${cuerpo.id.slice(0, 8)}.png`;
    $("descargar").hidden = false;

    if (cuerpo.url_original) {
      liberarUrl("urlOriginal");
      estado.urlOriginal = await descargarBlob(cuerpo.url_original);
      $("img-original").src = estado.urlOriginal;
    }

    mostrarLanzamiento(cuerpo);
    mostrarTiempos(cuerpo.tiempos_ms, cliente);
    agregarHistorial(cuerpo, cliente);
  } catch (err) {
    if (err.name !== "AbortError") mostrarAviso(`Fallo de red o de la API: ${err.message}`);
  } finally {
    estado.enCurso = null;
    $("procesar").disabled = !$("archivo").files[0];
    $("procesar").textContent = "Procesar en GPU";
    document.body.classList.remove("procesando");
    actualizarEstado();
  }
}

async function iniciar() {
  actualizarEstado();
  setInterval(actualizarEstado, 10000);
  try {
    estado.config = await pedirJson("/api/filtros");
  } catch (err) {
    mostrarAviso(`No se pudo leer la configuración de la API en ${API}: ${err.message}`);
    return;
  }
  llenarSelect($("filtro"), estado.config.filtros, "media");
  llenarSelect($("hilos"), estado.config.hilos, 256);
  alCambiarFiltro();

  $("filtro").addEventListener("change", alCambiarFiltro);
  $("k").addEventListener("change", alCambiarK);
  $("k-texto").addEventListener("change", alCambiarK);
  $("hilos").addEventListener("change", actualizarPrevision);
  $("editar-mascara").addEventListener("click", editarComoPersonalizada);
  $("archivo").addEventListener("change", alElegirArchivo);
  $("umbral").addEventListener("input", () => { $("valor-umbral").textContent = $("umbral").value; });
  for (const r of document.querySelectorAll("input[name=salida]")) r.addEventListener("change", alCambiarSalida);
  $("formulario").addEventListener("submit", procesar);
}

iniciar();
