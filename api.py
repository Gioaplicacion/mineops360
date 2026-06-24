"""
MineOps 360 — API FastAPI para Railway
========================================
Variables de entorno esperadas en Railway:
  PORT          → Puerto (Railway lo inyecta automáticamente)
  FRONTEND_URL  → URL de tu frontend en Bolt (ej: https://mineops360.bolt.host)
  
Sin FRONTEND_URL, acepta todos los orígenes (útil para desarrollo).
"""

import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

import sys
sys.path.append(str(Path(__file__).parent))
from engine.config import ProjectConfig
from engine.pipeline import MineOpsPipeline, PipelineResult

# ── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MineOps 360 API",
    description="Planificación minera open pit — SaaS + modo local",
    version="1.0.0",
)

# CORS dinámico — acepta el frontend de Bolt + localhost para desarrollo
FRONTEND_URL = os.getenv("FRONTEND_URL", "")
origins = ["*"] if not FRONTEND_URL else [
    FRONTEND_URL,
    "http://localhost:5173",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Storage en memoria (Railway reinicia → jobs temporales) ────────────
JOBS: dict[str, dict] = {}
WS_CLIENTS: dict[str, list] = {}
UPLOAD_DIR = Path("/tmp/mineops_uploads")
OUTPUT_DIR = Path("/tmp/mineops_outputs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Endpoints ──────────────────────────────────────────────────────────
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

    # Guardar CSV
    csv_path = UPLOAD_DIR / f"{job_id}_modelo.csv"
    content = await modelo_csv.read()
    csv_path.write_bytes(content)

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


# ── Background task ────────────────────────────────────────────────────
async def _run_job(job_id, csv_path, fases_path, params):
    JOBS[job_id]["status"] = "running"

    def progress_cb(paso, total, mensaje):
        JOBS[job_id]["progress"] = {"paso": paso, "total": total, "mensaje": mensaje}
        asyncio.create_task(_broadcast(job_id, {
            "paso": paso, "total": total, "mensaje": mensaje, "status": "running"
        }))

    try:
        config   = ProjectConfig.desde_dict(params)
        pipeline = MineOpsPipeline(config, progress_cb=progress_cb)
        fases_df = pd.read_csv(fases_path) if fases_path and fases_path.exists() else None

        loop = asyncio.get_event_loop()
        resultado: PipelineResult = await loop.run_in_executor(
            None, pipeline.ejecutar, str(csv_path), fases_df
        )

        resultado.bloques_df.to_csv(OUTPUT_DIR / f"{job_id}_bloques.csv", index=False)
        resultado.plan_df.to_csv(OUTPUT_DIR / f"{job_id}_plan.csv", index=False)

        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["result"] = {
            "job_id":          job_id,
            "van_total_MUSD":  resultado.van_total_MUSD,
            "tiempo_total_s":  resultado.tiempo_total_s,
            "resumen_modelo":  resultado.resumen_modelo,
            "resumen_pits":    resultado.resumen_pits,
            "plan_minero":     resultado.plan_minero,
            "download_bloques": f"/api/download/{job_id}/bloques",
            "download_plan":    f"/api/download/{job_id}/plan",
        }
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
