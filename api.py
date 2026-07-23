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
from engine.config import ProjectConfig
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

def convertir_asc_a_csv(contenido_bytes: bytes, filename: str) -> bytes:
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

    # Detectar metal automáticamente desde los nombres de columna
    metal_detectado = None
    col_metal_original = None
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in METALES_RECONOCIDOS:
            metal_detectado = METALES_RECONOCIDOS[col_lower]
            col_metal_original = col
            break

    # Renombrar columna del metal a "ley" (nombre genérico para el pipeline)
    if col_metal_original and col_metal_original in df.columns:
        df = df.rename(columns={col_metal_original: "ley"})
        logger.info(f"Metal detectado: {metal_detectado['nombre']} ({metal_detectado['simbolo']}) en columna '{col_metal_original}'")
    elif "Cu" in df.columns:
        df = df.rename(columns={"Cu": "ley"})
        metal_detectado = METALES_RECONOCIDOS["cu"]
    elif "ley" not in df.columns:
        df["ley"] = 0.0
        metal_detectado = {"simbolo": "Cu", "nombre": "Cobre", "unidad": "%"}
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

    metal_info = None
    if extension == ".asc" or tiene_columnas_alternativas:
        logger.info(f"Convirtiendo archivo {filename} a CSV estándar...")
        try:
            contenido, metal_info = convertir_asc_a_csv(contenido, filename)
            logger.info(f"Conversión exitosa - Metal: {metal_info}")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    csv_path = UPLOAD_DIR / f"{job_id}_modelo.csv"
    csv_path.write_bytes(contenido)

    fases_path = None
    if fases_csv:
        fases_path = UPLOAD_DIR / f"{job_id}_fases.csv"
        fases_path.write_bytes(await fases_csv.read())

    try:
        params = json.loads(params_json)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"params_json inválido: {e}")

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
    if tipo not in ("bloques", "plan"):
        raise HTTPException(status_code=400, detail="tipo debe ser 'bloques' o 'plan'")
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

        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["result"] = {
            "job_id":           job_id,
            "van_total_MUSD":   resultado.van_total_MUSD,
            "tiempo_total_s":   resultado.tiempo_total_s,
            "resumen_modelo":   resultado.resumen_modelo,
            "resumen_pits":     resultado.resumen_pits,
            "plan_minero":      resultado.plan_minero,
            "metal_info":       JOBS[job_id].get("metal_info"),
            "download_bloques": f"/api/download/{job_id}/bloques",
            "download_plan":    f"/api/download/{job_id}/plan",
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
