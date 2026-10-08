"""
MineOps 360 — API FastAPI corregida
Fix: "no running event loop" en Railway
Nuevo: soporte para archivos .asc (separados por espacio)
"""

import asyncio
import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional
from io import StringIO

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import sys
sys.path.append(str(Path(__file__).parent))
from engine.config import ProjectConfig, METALES_ECO
from engine.pipeline import MineOpsPipeline, PipelineResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="MineOps 360 API", version="1.0.0")

FRONTEND_URL = os.getenv("FRONTEND_URL", "")
origins = ["*"] if not FRONTEND_URL else [FRONTEND_URL, "http://localhost:5173", "http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

JOBS: dict = {}
WS_CLIENTS: dict = {}
UPLOAD_DIR = Path("/tmp/mineops_uploads")
OUTPUT_DIR = Path("/tmp/mineops_outputs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ThreadPoolExecutor para correr el pipeline sin bloquear
executor = ThreadPoolExecutor(max_workers=2)

# Mapeo de columnas alternativas (.asc y otros formatos mineros)
COLUMN_MAP = {
    "east":      "X",
    "este":      "X",
    "north":     "Y",
    "norte":     "Y",
    "elev":      "Z",
    "elevation": "Z",
    "z_coord":   "Z",
    "tones":     "tonelaje",
    "tonnes":    "tonelaje",
    "ton":       "tonelaje",
    "sg":        "dens",
    "densidad":  "dens",
    "density":   "dens",
    "class":     "fase",
    "roca":      "tipo_roca",
}

# Metales reconocidos con sus símbolos y unidades
METALES_RECONOCIDOS = {
    "cu":    {"simbolo": "Cu", "nombre": "Cobre",     "unidad": "%"},
    "cu_pct":{"simbolo": "Cu", "nombre": "Cobre",     "unidad": "%"},
    "au":    {"simbolo": "Au", "nombre": "Oro",       "unidad": "g/t"},
    "au_gt": {"simbolo": "Au", "nombre": "Oro",       "unidad": "g/t"},
    "ag":    {"simbolo": "Ag", "nombre": "Plata",     "unidad": "g/t"},
    "ag_gt": {"simbolo": "Ag", "nombre": "Plata",     "unidad": "g/t"},
    "ni":    {"simbolo": "Ni", "nombre": "Niquel",    "unidad": "%"},
    "li":    {"simbolo": "Li", "nombre": "Litio",     "unidad": "%"},
    "co":    {"simbolo": "Co", "nombre": "Cobalto",   "unidad": "%"},
    "zn":    {"simbolo": "Zn", "nombre": "Zinc",      "unidad": "%"},
    "fe":    {"simbolo": "Fe", "nombre": "Hierro",    "unidad": "%"},
    "mo":    {"simbolo": "Mo", "nombre": "Molibdeno", "unidad": "%"},
    "grade": {"simbolo": "Cu", "nombre": "Cobre",     "unidad": "%"},
    "ley":   {"simbolo": "Cu", "nombre": "Cobre",     "unidad": "%"},
}

def convertir_asc_a_csv(contenido_bytes: bytes, filename: str, metal_pref: Optional[str] = None):
    """
    Convierte un archivo .asc (separado por espacios) a CSV estándar.
    También maneja archivos CSV con columnas de nombres alternativos.
    Filtra automáticamente info=1 si existe esa columna.
    Subamplea si el archivo tiene más de 50,000 bloques.
    """
    texto = contenido_bytes.decode("utf-8", errors="ignore")
    primera_linea = texto.split("\n")[0].strip()

    # Detectar separador
    if "," in primera_linea:
        sep = ","
    else:
        sep = r"\s+"

    try:
        df = pd.read_csv(StringIO(texto), sep=sep, engine="python")
    except Exception as e:
        logger.error(f"Error leyendo archivo {filename}: {e}")
        raise ValueError(f"No se pudo leer el archivo: {e}")

    logger.info(f"Archivo {filename}: {len(df)} filas, columnas: {list(df.columns)}")

    # Normalizar nombres de columnas
    df.columns = [c.strip() for c in df.columns]
    rename_dict = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in COLUMN_MAP:
            rename_dict[col] = COLUMN_MAP[col_lower]
    if rename_dict:
        df = df.rename(columns=rename_dict)
        logger.info(f"Columnas renombradas: {rename_dict}")

    # Filtrar info=1 si existe columna info
    if "info" in df.columns:
        total_original = len(df)
        df = df[df["info"] == 1].copy()
        logger.info(f"Filtrado info=1: {len(df)} de {total_original} bloques")

    # Verificar columnas mínimas requeridas
    cols_requeridas = ["X", "Y", "Z"]
    faltantes = [c for c in cols_requeridas if c not in df.columns]
    if faltantes:
        raise ValueError(f"Columnas requeridas no encontradas: {faltantes}. Columnas disponibles: {list(df.columns)}")

    # Detectar los minerales que trae el archivo (Cu, Au, Ag...) y elegir cuál se evalúa
    candidatos = []
    for col in df.columns:
        cl = col.lower().strip()
        if cl in METALES_RECONOCIDOS and cl not in ("grade", "ley"):
            candidatos.append((cl, col))
    genericas = [(c.lower().strip(), c) for c in df.columns if c.lower().strip() in ("grade", "ley")]
    metal_detectado = None
    col_metal_original = None
    pref = (metal_pref or "").lower().strip()
    if candidatos:
        elegido = None
        if pref:
            for cl, col in candidatos:
                if METALES_RECONOCIDOS[cl]["simbolo"].lower() == pref:
                    elegido = (cl, col)
                    break
            if elegido is None and not genericas:
                disp = ", ".join(sorted({METALES_RECONOCIDOS[cl]["simbolo"] for cl, _ in candidatos}))
                raise ValueError(
                    f"El archivo no trae una columna del mineral seleccionado ({pref.upper()}). "
                    f"Minerales disponibles en el archivo: {disp}."
                )
        if elegido is None:
            elegido = candidatos[0]
        metal_detectado = dict(METALES_RECONOCIDOS[elegido[0]])
        col_metal_original = elegido[1]
    elif genericas:
        metal_detectado = dict(METALES_RECONOCIDOS[genericas[0][0]])
        col_metal_original = genericas[0][1]
    if metal_detectado is not None:
        metal_detectado["disponibles"] = sorted({METALES_RECONOCIDOS[cl]["simbolo"] for cl, _ in candidatos})
        _ul, _up, _f = METALES_ECO.get(metal_detectado["simbolo"].lower(), METALES_ECO["cu"])
        metal_detectado["unidad"] = _ul
        metal_detectado["unidad_precio"] = _up

    # Renombrar columna del metal a "ley" (nombre genérico para el pipeline)
    if col_metal_original and col_metal_original in df.columns:
        df = df.rename(columns={col_metal_original: "ley"})
        logger.info(f"Metal detectado: {metal_detectado['nombre']} ({metal_detectado['simbolo']}) en columna '{col_metal_original}'")
    elif "ley" not in df.columns:
        df["ley"] = 0.0
        metal_detectado = {"simbolo": "Cu", "nombre": "Cobre", "unidad": "%", "unidad_precio": "USD/lb", "disponibles": []}
        logger.warning("No se encontró columna de ley, usando ley=0")

    # Subsamplear si es muy grande (más de 50,000 bloques)
    MAX_BLOQUES = 50000
    if len(df) > MAX_BLOQUES:
        factor = len(df) // MAX_BLOQUES
        df = df.iloc[::factor].head(MAX_BLOQUES).copy()
        logger.info(f"Subsamplado a {len(df)} bloques (1 de cada {factor})")

    # Seleccionar columnas relevantes para el pipeline
    cols_salida = ["X", "Y", "Z", "ley"]
    if "tonelaje" in df.columns:
        cols_salida.append("tonelaje")
    if "dens" in df.columns:
        cols_salida.append("dens")

    df_out = df[[c for c in cols_salida if c in df.columns]]
    logger.info(f"CSV final: {len(df_out)} bloques, columnas: {list(df_out.columns)}")

    # Retornar CSV + metadato del metal detectado
    csv_bytes = df_out.to_csv(index=False).encode("utf-8")
    return csv_bytes, metal_detectado


# ── Asistente virtual ─────────────────────────────────────────────────────
# Con la variable ANTHROPIC_API_KEY (Railway → Variables) responde con IA, usando los resultados del proyecto.
# Sin clave responde en modo básico (reglas simples) para que el panel siempre funcione.
ASISTENTE_MODELO = os.getenv("ASISTENTE_MODELO", "claude-sonnet-5-5")
ASISTENTE_SISTEMA = """Eres el asistente virtual de Global Mine Planner, una aplicación de planificación minera a cielo abierto.
Hablas en español de Chile, claro y directo, para un usuario de minería que no es programador.
Conoces el DXF descargable (fases, períodos, superficie del pit final) y los módulos: Modelo (modelo de bloques como paralelepípedo, filtro por mineral y ley), ¿Cuánto? (reservas y pit óptimo por \
análisis pit-by-pit con Lerchs-Grossmann y factores de ingresos), ¿Cómo? Fases (pushbacks en forma de cono, talud global, \
recocido simulado) y ¿Cuándo? (plan por años con tasa de mediana minería, tabla de extracción por fase y año, VAN).
Unidades: cobre con ley en % y precio en USD/lb; oro y plata con ley en g/t y precio en USD/oz (1 oz troy = 31,1035 g).
Reglas: usa SOLO los números del contexto entregado y nunca inventes cifras; si falta un dato, dilo y explica dónde sacarlo en la app. \
No des asesoría financiera ni de inversión; explica el método y los supuestos. El campo plan_por_periodo (p=período, min_Mt, est_Mt, mov_Mt, ley, van_MUSD) permite responder qué pasa en cada año. Respuestas breves (máximo ~8 líneas), con pasos \
numerados cuando el usuario pregunte cómo hacer algo."""


def _contexto_texto(ctx: dict) -> str:
    try:
        return json.dumps(ctx, ensure_ascii=False, separators=(',', ':'))[:9000]
    except Exception:
        return "{}"


def _asistente_ia(mensaje: str, historial: list, ctx: dict) -> str:
    import urllib.request
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    msgs = []
    for h in (historial or [])[-8:]:
        rol = "assistant" if h.get("rol") == "asistente" else "user"
        texto = str(h.get("texto", ""))[:2000]
        if texto:
            msgs.append({"role": rol, "content": texto})
    msgs.append({"role": "user", "content": mensaje[:2000]})
    # La API exige que el primer mensaje sea del usuario y que se alternen
    while msgs and msgs[0]["role"] != "user":
        msgs.pop(0)
    limpio = []
    for m in msgs:
        if limpio and limpio[-1]["role"] == m["role"]:
            limpio[-1]["content"] += "\n" + m["content"]
        else:
            limpio.append(m)
    body = json.dumps({
        "model": ASISTENTE_MODELO,
        "max_tokens": 700,
        "system": ASISTENTE_SISTEMA + "\n\nContexto actual del proyecto (JSON): " + _contexto_texto(ctx),
        "messages": limpio,
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=body, method="POST",
        headers={"content-type": "application/json", "x-api-key": key, "anthropic-version": "2023-06-01"},
    )
    with urllib.request.urlopen(req, timeout=40) as r:
        data = json.loads(r.read().decode("utf-8"))
    return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text").strip()


def _faq(ctx: dict) -> list:
    """Preguntas frecuentes del programa: (palabras clave, respuesta). Sólo temas de Global Mine Planner."""
    rm = (ctx or {}).get("resumen_modelo") or {}
    rp = (ctx or {}).get("resumen_pits") or {}
    van = (ctx or {}).get("van_total_MUSD")
    metal = rm.get("metal", "Cu")
    ul, up = rm.get("unidad_ley", "%"), rm.get("unidad_precio", "USD/lb")
    sin = "Aún no hay resultados: sube tu modelo en 'Proyecto' y pulsa Ejecutar pipeline."
    return [
        (["formato", "archivo", "csv", "columna", "east", "north", "elev", "asc", "txt", "subir", "cargar"],
         "El archivo necesita coordenadas (X/EAST, Y/NORTH, Z/ELEV) y la ley del mineral (CU, AU, AG, MO...). La densidad (SG) es opcional: si viene, cada bloque usa su densidad. Acepta CSV, TXT y ASC. El programa detecta las columnas solo y te muestra lo que encontró antes de ejecutar."),
        (["tamano", "tamaño", "soporta", "limite", "límite", "maximo", "máximo", "bloques", "grande"],
         "El motor trabaja con hasta 50.000 bloques: si tu modelo es más grande, toma 1 de cada N filas para mantener la forma del yacimiento. El visor muestra hasta 60.000 bloques."),
        (["ley de corte", "cutoff", "corte"],
         (f"La ley de corte calculada es {rm.get('ley_corte_calculada_pct')} {ul} ({metal}). " if rm.get("ley_corte_calculada_pct") is not None else "")
         + "Se calcula como (costo mina + costo planta) / ((precio − TC/RC) × recuperación × factor de conversión). Sube con los costos y baja con el precio y la recuperación."),
        (["van", "valor actual", "npv"],
         (f"El VAN del plan es {van} MUSD, descontado con la tasa del proyecto. " if van is not None else sin + " ")
         + "Es la suma de (ingresos − costos mina, planta, remanejo y stock) de cada período, descontados."),
        (["pit optimo", "pit óptimo", "optimo", "óptimo", "lerchs", "grossmann", "pit-by-pit", "pit by pit", "cuanto", "cuánto", "factor de ingresos", "rf"],
         (f"El pit óptimo es el nº {rp.get('pit_optimo')} (factor de ingresos {rp.get('rf_optimo')}). " if rp.get("pit_optimo") else "")
         + "El módulo ¿Cuánto? calcula 25 pits anidados con Lerchs-Grossmann variando el precio (factor de ingresos). El óptimo es el que maximiza el VAN especificado; el plan usa los bloques hasta ese pit."),
        (["fase", "cono", "pushback", "talud", "recocido", "como", "cómo"],
         "El módulo ¿Cómo? divide el pit en fases (pushbacks) con recocido simulado. Cada fase acumulada es un pit con paredes al talud global (45° por defecto), por eso tiene forma de cono, y cada fase queda con al menos 12 % de los bloques."),
        (["periodo", "período", "año", "plan", "cuando", "cuándo", "constante", "tasa", "capacidad", "gantt", "extraccion", "extracción", "tabla"],
         "El módulo ¿Cuándo? programa fase por fase y banco por banco. Mantiene una tasa constante: 2,4 Mt/año de mineral y hasta 7,5 Mt/año de movimiento total por defecto (se cambian en Proyecto). Una fase que se abre no se detiene. Ves el Gantt y la tabla de extracción por fase y año (Mt)."),
        (["unidad", "oro", "au", "onza", "oz", "cobre", "cu", "plata", "ag", "precio", "mineral a evaluar", "metal"],
         f"Para {metal} la ley va en {ul} y el precio en {up}. Cobre/zinc/plomo/molibdeno: ley en % y precio en USD/lb. Oro/plata/platino: ley en g/t y precio en USD/oz (1 oz = 31,1035 g). Cambia el mineral en 'Mineral a evaluar' (pestaña Proyecto)."),
        (["dxf", "descargar", "exportar", "autocad", "vulcan", "datamine", "surpac"],
         "Al terminar, el botón 'Descargar DXF' entrega sólidos de cada fase (FASE_n), sólidos de cada período (PERIODO_nn) y la superficie del pit final (SUPERFICIE_PIT_FINAL), cada uno en su capa. También puedes bajar el plan y los bloques en CSV."),
        (["filtro", "filtrar", "modelo", "paralelepipedo", "paralelepípedo", "color", "visor", "3d", "ver"],
         "En la pestaña Modelo ves todo el yacimiento como paralelepípedo con colores por ley. Puedes elegir el mineral y filtrar por rango de ley (mín/máx) para ver sólo los bloques que te interesan."),
        (["densidad", "sg", "tonelaje"],
         "Si tu archivo trae densidad (SG/DENS), cada bloque usa su propia densidad para calcular el tonelaje. Si no, se usa la densidad del proyecto (2,5 t/m³ por defecto)."),
        (["recuperacion", "recuperación", "costo", "costos", "tc", "rc", "parametro", "parámetro", "economic"],
         "Los parámetros económicos (precio, TC/RC, recuperación, costo mina y planta, tasa de descuento) se ingresan en la pestaña Proyecto antes de ejecutar. Cambiarlos modifica la ley de corte, el pit óptimo y el VAN."),
        (["ejecutar", "pipeline", "demora", "tarda", "tiempo", "error", "falla"],
         "Pulsa 'Ejecutar pipeline' en Proyecto. Corre 4 pasos: carga del modelo, optimización del pit, faseamiento y plan. Con 50.000 bloques puede tardar varios minutos. Si hay error, revisa que el archivo tenga coordenadas y una columna de ley del mineral elegido."),
    ]


_FUERA_DE_TEMA = ("Sólo puedo ayudarte con Global Mine Planner. Pregúntame por: el formato del archivo, la ley de corte, el pit óptimo, "
                  "las fases, el plan por años, el VAN, las unidades (Cu/Au), el filtro del modelo o la descarga DXF.")


def _asistente_basico(mensaje: str, ctx: dict) -> str:
    """Preguntas frecuentes (sin IA, sin costo): elige el tema con más coincidencias de palabras clave."""
    m = " " + mensaje.lower() + " "
    mejor, puntos = None, 0
    for claves, resp in _faq(ctx):
        p = sum(len(k) for k in claves if (" " + k + " ") in m or (len(k) > 3 and k in m))
        if p > puntos:
            mejor, puntos = resp, p
    return mejor if mejor else _FUERA_DE_TEMA


@app.post("/api/asistente")
async def asistente(payload: dict):
    mensaje = str(payload.get("mensaje", "")).strip()
    if not mensaje:
        raise HTTPException(status_code=400, detail="Mensaje vacío")
    ctx = payload.get("contexto") or {}
    historial = payload.get("historial") or []
    loop = asyncio.get_event_loop()
    if os.getenv("ANTHROPIC_API_KEY", "").strip() and os.getenv("ASISTENTE_IA", "") == "1":
        try:
            texto = await loop.run_in_executor(None, _asistente_ia, mensaje, historial, ctx)
            if texto:
                return {"respuesta": texto, "modo": "ia"}
        except Exception as e:
            logger.warning(f"Asistente IA falló, uso modo básico: {e}")
    return {"respuesta": _asistente_basico(mensaje, ctx), "modo": "basico"}


@app.get("/")
async def root():
    return {"service": "MineOps 360 API", "status": "ok", "version": "1.0.0"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/run")
async def run_pipeline(
    modelo_csv:  UploadFile = File(...),
    fases_csv:   Optional[UploadFile] = File(None),
    params_json: str = Form("{}"),
):
    job_id = str(uuid.uuid4())

    # Leer contenido del archivo
    contenido = await modelo_csv.read()
    filename = modelo_csv.filename or "modelo.csv"

    # Convertir .asc o normalizar columnas si es necesario
    extension = Path(filename).suffix.lower()
    primera_linea = contenido[:200].decode("utf-8", errors="ignore").split("\n")[0]
    tiene_columnas_alternativas = any(
        col in primera_linea.lower()
        for col in ["east", "north", "elev", "au_gt", "tones", "tonnes", "info"]
    )

    try:
        params_pre = json.loads(params_json)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"params_json inválido: {e}")

    # Siempre se normaliza el archivo: EAST/NORTH/ELEV, AU/CU/AG..., SG (densidad), info...
    logger.info(f"Normalizando archivo {filename}...")
    try:
        contenido, metal_info = convertir_asc_a_csv(contenido, filename, params_pre.get("metal"))
        logger.info(f"Conversión exitosa - Metal: {metal_info}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    csv_path = UPLOAD_DIR / f"{job_id}_modelo.csv"
    csv_path.write_bytes(contenido)

    fases_path = None
    if fases_csv:
        fases_path = UPLOAD_DIR / f"{job_id}_fases.csv"
        fases_path.write_bytes(await fases_csv.read())

    params = params_pre
    if metal_info:
        # El mineral evaluado es el de la columna elegida: define unidad de ley y de precio
        params["metal"] = metal_info["simbolo"].lower()

    JOBS[job_id] = {
        "status":   "queued",
        "progress": {"paso": 0, "total": 4, "mensaje": "En cola..."},
        "result":   None,
        "error":    None,
        "metal_info": metal_info,
        "created_at": time.time(),
    }
    WS_CLIENTS[job_id] = []

    asyncio.create_task(_run_job(job_id, csv_path, fases_path, params))

    return {"job_id": job_id, "status": "queued", "ws_url": f"/ws/{job_id}"}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job no encontrado")
    j = JOBS[job_id]
    return {"job_id": job_id, "status": j["status"], "progress": j["progress"], "error": j["error"]}


@app.get("/api/results/{job_id}")
async def get_results(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job no encontrado")
    if JOBS[job_id]["status"] != "done":
        raise HTTPException(status_code=409, detail=f"Estado: {JOBS[job_id]['status']}")
    return JOBS[job_id]["result"]


@app.get("/api/download/{job_id}/{tipo}")
async def download(job_id: str, tipo: str):
    if tipo not in ("bloques", "plan", "dxf"):
        raise HTTPException(status_code=400, detail="tipo debe ser 'bloques', 'plan' o 'dxf'")
    if tipo == "dxf":
        archivo = OUTPUT_DIR / f"{job_id}_resultados.dxf"
        if not archivo.exists():
            raise HTTPException(status_code=404, detail="DXF no disponible para este job")
        return FileResponse(str(archivo), media_type="application/dxf",
                            filename=f"mineops_fases_periodos_pit_{job_id[:8]}.dxf")
    archivo = OUTPUT_DIR / f"{job_id}_{tipo}.csv"
    if not archivo.exists():
        raise HTTPException(status_code=404, detail="Archivo no encontrado")
    return FileResponse(str(archivo), media_type="text/csv",
                        filename=f"mineops_{tipo}_{job_id[:8]}.csv")


@app.websocket("/ws/{job_id}")
async def ws_progress(websocket: WebSocket, job_id: str):
    await websocket.accept()
    if job_id not in JOBS:
        await websocket.send_json({"error": "Job no encontrado"})
        await websocket.close()
        return

    WS_CLIENTS.setdefault(job_id, []).append(websocket)
    try:
        j = JOBS[job_id]
        await websocket.send_json({**j["progress"], "status": j["status"]})
        while JOBS.get(job_id, {}).get("status") not in ("done", "error"):
            await asyncio.sleep(0.5)
        final = JOBS.get(job_id, {})
        await websocket.send_json({
            "status":  final.get("status"),
            "mensaje": "Completado" if final.get("status") == "done" else final.get("error", ""),
            "paso": 4, "total": 4,
        })
    except WebSocketDisconnect:
        pass
    finally:
        if job_id in WS_CLIENTS and websocket in WS_CLIENTS[job_id]:
            WS_CLIENTS[job_id].remove(websocket)


async def _run_job(job_id, csv_path, fases_path, params):
    """Ejecuta el pipeline en threadpool para no bloquear el event loop."""
    JOBS[job_id]["status"] = "running"
    loop = asyncio.get_event_loop()

    def progress_cb(paso, total, mensaje):
        JOBS[job_id]["progress"] = {"paso": paso, "total": total, "mensaje": mensaje}
        asyncio.run_coroutine_threadsafe(
            _broadcast(job_id, {"paso": paso, "total": total, "mensaje": mensaje, "status": "running"}),
            loop
        )

    def _run_sync():
        try:
            config   = ProjectConfig.desde_dict(params)
            pipeline = MineOpsPipeline(config, progress_cb=progress_cb)
            fases_df = pd.read_csv(fases_path) if fases_path and fases_path.exists() else None
            return pipeline.ejecutar(str(csv_path), fases_df)
        except Exception as e:
            raise e

    try:
        resultado = await loop.run_in_executor(executor, _run_sync)

        resultado.bloques_df.to_csv(OUTPUT_DIR / f"{job_id}_bloques.csv", index=False)
        resultado.plan_df.to_csv(OUTPUT_DIR / f"{job_id}_plan.csv", index=False)

        # DXF: sólidos de fases, sólidos de períodos y superficie del pit final
        dxf_ok = False
        try:
            from engine.dxf_export import generar_dxf
            blq = params.get("bloque", {}) or {}
            await loop.run_in_executor(
                executor,
                lambda: generar_dxf(resultado.bloques_df,
                                    float(blq.get("xsiz", 20)), float(blq.get("ysiz", 20)),
                                    float(blq.get("zsiz", 15)),
                                    str(OUTPUT_DIR / f"{job_id}_resultados.dxf")))
            dxf_ok = True
        except Exception as e:
            logger.warning(f"No se pudo generar el DXF: {e}")

        try:
            from engine.reconciliacion import reconciliar
            recon = reconciliar(ProjectConfig.desde_dict(params), resultado.bloques_df, resultado.plan_df,
                                resultado.van_total_MUSD * 1e6, resultado.resumen_pits, resultado.resumen_modelo)
        except Exception as e:
            logger.warning(f"Reconciliación no disponible: {e}")
            recon = []

        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["result"] = {
            "job_id":           job_id,
            "van_total_MUSD":   resultado.van_total_MUSD,
            "tiempo_total_s":   resultado.tiempo_total_s,
            "resumen_modelo":   resultado.resumen_modelo,
            "resumen_pits":     resultado.resumen_pits,
            "plan_minero":      resultado.plan_minero,
            "reconciliacion":   recon,
            "metal_info":       JOBS[job_id].get("metal_info"),
            "download_bloques": f"/api/download/{job_id}/bloques",
            "download_plan":    f"/api/download/{job_id}/plan",
            "download_dxf":     f"/api/download/{job_id}/dxf" if dxf_ok else None,
        }
        await _broadcast(job_id, {"status": "done", "paso": 4, "total": 4, "mensaje": "Completado"})

    except Exception as e:
        logger.exception(f"Error en job {job_id}: {e}")
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"]  = str(e)
        await _broadcast(job_id, {"status": "error", "error": str(e)})


async def _broadcast(job_id, msg):
    for ws in WS_CLIENTS.get(job_id, []):
        try:
            await ws.send_json(msg)
        except Exception:
            pass
