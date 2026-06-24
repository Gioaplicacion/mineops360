"""
MineOps 360 — Pipeline Orquestador
====================================
Punto de entrada único que ejecuta el flujo completo:
  CSV → Valorización → Optimización Pit → Scheduling → Plan Minero

Diseñado para ser llamado desde:
  - La API FastAPI (modo cloud)
  - CLI local (modo offline)
  - Tests unitarios
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from .config import ProjectConfig
from .loader import cargar_modelo, ModeloBloques
from .optimizer import optimizar_pits, ResultadoPitOptimizer
from .scheduler import HeuristicaFaseBanco, ResultadoScheduler

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Resultado completo del pipeline para serializar a la API."""
    config:          ProjectConfig
    resumen_modelo:  dict = field(default_factory=dict)
    resumen_pits:    dict = field(default_factory=dict)
    plan_minero:     dict = field(default_factory=dict)
    van_total_MUSD:  float = 0.0
    tiempo_total_s:  float = 0.0
    # DataFrames para descarga / visualización 3D
    bloques_df:      Optional[pd.DataFrame] = None
    plan_df:         Optional[pd.DataFrame] = None


class MineOpsPipeline:
    """
    Orquestador del pipeline completo de planificación minera.

    Uso básico:
        cfg = ProjectConfig.desde_dict(params_usuario)
        pipeline = MineOpsPipeline(cfg, progress_cb=mi_funcion)
        resultado = pipeline.ejecutar("ruta/modelo.csv", fases_df)
    """

    def __init__(
        self,
        config: ProjectConfig,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
    ):
        self.config      = config
        self.progress_cb = progress_cb or (lambda paso, total, msg: None)
        self._pasos_totales = 4

    def ejecutar(
        self,
        csv_path: str | Path,
        fases_df: Optional[pd.DataFrame] = None,
    ) -> PipelineResult:
        """
        Ejecuta el pipeline completo.

        Args:
            csv_path:  Ruta al CSV del modelo de bloques (X, Y, Z, Cu).
            fases_df:  DataFrame con fases asignadas. Si es None, se usa
                       el resultado del optimizer directamente (pit = fase).

        Returns:
            PipelineResult con todos los resultados y DataFrames.
        """
        t0 = time.time()
        result = PipelineResult(config=self.config)

        # ── PASO 1: Cargar y validar modelo ────────────────────────────
        self._emit(1, "Cargando y validando modelo de bloques...")
        modelo = cargar_modelo(csv_path, self.config)
        result.resumen_modelo = modelo.resumen
        logger.info(f"Modelo cargado: {result.resumen_modelo}")

        # ── PASO 2: Optimizar pits (MaxFlow) ───────────────────────────
        self._emit(2, "Optimizando pit final (Lerchs-Grossmann)...")
        opt_result = optimizar_pits(modelo, progress_cb=self.progress_cb)
        result.resumen_pits = opt_result.to_dict()

        # ── PASO 3: Preparar fases ─────────────────────────────────────
        self._emit(3, "Preparando faseamiento...")
        if fases_df is None:
            # Sin faseamiento externo: pit = fase (modo simplificado)
            fases_df = self._pits_como_fases(opt_result)

        # ── PASO 4: Scheduling ─────────────────────────────────────────
        self._emit(4, "Ejecutando scheduler heurístico...")
        scheduler = HeuristicaFaseBanco(opt_result, fases_df, progress_cb=self.progress_cb)
        sch_result = scheduler.ejecutar()

        # ── Ensamblar resultado ────────────────────────────────────────
        result.plan_minero    = sch_result.to_dict()
        result.van_total_MUSD = round(sch_result.van_total / 1e6, 4)
        result.bloques_df     = sch_result.bloques
        result.plan_df        = sch_result.plan
        result.tiempo_total_s = round(time.time() - t0, 2)

        logger.info(
            f"✅ Pipeline completado en {result.tiempo_total_s}s | "
            f"VAN = {result.van_total_MUSD:,.2f} MUSD"
        )
        return result

    def _emit(self, paso: int, mensaje: str) -> None:
        logger.info(f"[{paso}/{self._pasos_totales}] {mensaje}")
        self.progress_cb(paso, self._pasos_totales, mensaje)

    @staticmethod
    def _pits_como_fases(opt_result: ResultadoPitOptimizer) -> pd.DataFrame:
        """Convierte pits anidados en fases (pit N → fase N)."""
        df = opt_result.df.dropna(subset=["pit"]).copy()
        df = df.rename(columns={"x": "X", "y": "Y", "z": "Z", "ley": "Ley"})
        df["fase"] = df["pit"].astype(int)
        df["tonelaje"] = df["ton"]
        return df[["X", "Y", "Z", "fase", "Ley", "tonelaje"]]


# ---------------------------------------------------------------------------
# CLI local (python -m engine.pipeline modelo.csv)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import json

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if len(sys.argv) < 2:
        print("Uso: python -m engine.pipeline <modelo.csv> [params.json]")
        sys.exit(1)

    csv_path = sys.argv[1]
    params   = {}
    if len(sys.argv) >= 3:
        with open(sys.argv[2]) as f:
            params = json.load(f)

    config = ProjectConfig.desde_dict(params)

    def progress(paso, total, msg):
        print(f"  [{paso}/{total}] {msg}")

    pipeline = MineOpsPipeline(config, progress_cb=progress)
    resultado = pipeline.ejecutar(csv_path)

    print(f"\n{'='*50}")
    print(f"VAN Total:  {resultado.van_total_MUSD:,.2f} MUSD")
    print(f"Períodos:   {resultado.plan_minero['periodos']}")
    print(f"Tiempo:     {resultado.tiempo_total_s}s")
    print(f"{'='*50}")

    # Guardar outputs
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    resultado.bloques_df.to_csv(out_dir / "bloques_schedulados.csv", index=False)
    resultado.plan_df.to_csv(out_dir / "plan_minero.csv", index=False)
    print(f"\nResultados guardados en: {out_dir}/")
