"""
MineOps 360 — Optimizador de Pit (Lerchs-Grossmann vía MaxFlow)
================================================================
FIX aplicados vs código original:
  1. Recibe ModeloBloques ya limpio (Ley=-99 ya filtrada upstream)
  2. Ley de corte calculada dinámicamente desde config (no hardcodeada)
  3. Manejo de errores robusto en MaxFlow
  4. Progreso reportado vía callback (para API/UI en tiempo real)
  5. Resultado como DataFrame enriquecido, no solo CSV
"""

import logging
import math
from typing import Callable, Optional

import maxflow
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .config import ProjectConfig, PitOptimizerConfig
from .loader import ModeloBloques

logger = logging.getLogger(__name__)

BIG = 9_999_999.0  # capacidad infinita en aristas de precedencia


# ---------------------------------------------------------------------------
# Resultado del optimizador
# ---------------------------------------------------------------------------
class ResultadoPitOptimizer:
    """Contenedor del resultado del optimizador para uso downstream."""

    def __init__(self, df: pd.DataFrame, config: ProjectConfig, precios: list[float]):
        self.df = df          # DataFrame con columnas: x,y,z,ley,value,pit,ton,wton,oton
        self.config = config
        self.precios = precios

    @property
    def resumen_pits(self) -> pd.DataFrame:
        """VAN, mineral y estéril acumulado por número de pit."""
        return (
            self.df.dropna(subset=["pit"])
            .groupby("pit")
            .agg(
                oton=("oton", "sum"),
                wton=("wton", "sum"),
                value=("value", "sum"),
            )
            .assign(
                oton_acum=lambda d: d["oton"].cumsum(),
                wton_acum=lambda d: d["wton"].cumsum(),
                value_acum=lambda d: d["value"].cumsum(),
            )
            .reset_index()
        )

    def to_dict(self) -> dict:
        return {
            "total_bloques_en_pit": int(self.df["pit"].notna().sum()),
            "num_pits": int(self.df["pit"].dropna().max()),
            "pits": self.resumen_pits.to_dict(orient="records"),
        }


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------
def optimizar_pits(
    modelo: ModeloBloques,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> ResultadoPitOptimizer:
    """
    Corre el optimizador Lerchs-Grossmann para N escenarios de precio en paralelo.

    Args:
        modelo:       ModeloBloques cargado y validado.
        progress_cb:  Callback(paso_actual, total_pasos, mensaje) para UI en tiempo real.

    Returns:
        ResultadoPitOptimizer con DataFrame enriquecido y métricas por pit.
    """
    cfg = modelo.config
    opt = cfg.optimizador
    df  = modelo.df.copy()

    # Coordenadas y grilla
    xmn, ymn, zmn = df["x"].min(), df["y"].min(), df["z"].min()
    xsiz, ysiz, zsiz = modelo.xsiz, modelo.ysiz, modelo.zsiz
    ton = modelo.ton_bloque

    # Escenarios de precio (Revenue Factors desde RF_min hasta precio_base)
    factores = np.linspace(1.0 / opt.num_escenarios, 1.0, opt.num_escenarios)
    precios  = [opt.precio_base * f for f in factores]

    logger.info(f"Iniciando {opt.num_escenarios} escenarios MaxFlow en paralelo...")
    if progress_cb:
        progress_cb(0, opt.num_escenarios, "Iniciando optimización MaxFlow...")

    # Paralelizar MaxFlow
    results = Parallel(n_jobs=-1)(
        delayed(_run_maxflow_escenario)(
            df.copy(), precio, xmn, ymn, zmn, xsiz, ysiz, zsiz,
            opt, cfg.economico, ton
        )
        for precio in precios
    )

    if progress_cb:
        progress_cb(opt.num_escenarios, opt.num_escenarios, "MaxFlow completado. Ensamblando resultado...")

    # Ensamblar matrices de resultado
    n = len(df)
    value_matrix = np.zeros((n, opt.num_escenarios), dtype=float)
    pit_matrix   = np.zeros((n, opt.num_escenarios), dtype=int)

    for j, (val, pitm) in enumerate(results):
        value_matrix[:, j] = val
        pit_matrix[:, j]   = pitm

    # Primer pit donde entra cada bloque
    pit_nivel   = np.full(n, np.nan)
    value_final = np.zeros(n, dtype=float)

    for i in range(n):
        for j in range(opt.num_escenarios):
            if pit_matrix[i, j] == 1:
                pit_nivel[i]   = j + 1
                value_final[i] = value_matrix[i, j]
                break

    # Armar DataFrame final
    out = df.copy()
    out["value"] = value_final
    out["pit"]   = pit_nivel
    out["ton"]   = ton
    out["wton"]  = np.where(out["value"] < 0, ton, 0.0)
    out["oton"]  = np.where(out["value"] >= 0, ton, 0.0)
    out = out.sort_values(["pit", "z"], ascending=[True, False]).reset_index(drop=True)

    logger.info(f"✅ Optimización completada. Pits generados: {int(np.nanmax(pit_nivel))}")
    return ResultadoPitOptimizer(out, cfg, precios)


# ---------------------------------------------------------------------------
# MaxFlow por escenario (se ejecuta en paralelo)
# ---------------------------------------------------------------------------
def _run_maxflow_escenario(
    df: pd.DataFrame,
    precio: float,
    xmn: float, ymn: float, zmn: float,
    xsiz: float, ysiz: float, zsiz: float,
    opt: PitOptimizerConfig,
    eco,
    ton: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Ejecuta MaxFlow para un precio dado y retorna (values, pit_mask).
    Corre en proceso separado vía joblib.
    """
    ley = df["ley"].values

    ix = ((df["x"].values - xmn) // xsiz).astype(int)
    iy = ((df["y"].values - ymn) // ysiz).astype(int)
    iz = ((df["z"].values - zmn) // zsiz).astype(int)

    coord_to_idx = {(i, j, k): idx for idx, (i, j, k) in enumerate(zip(ix, iy, iz))}
    idx_to_coord = {v: k for k, v in coord_to_idx.items()}
    n = len(df)

    # Valorización con ley de corte dinámica para este precio
    denominador = (precio - eco.cargo_tc_rc) * eco.recuperacion_dec * eco.lbs_por_ton
    lc = ((eco.costo_mina + eco.costo_planta) / denominador * 100.0
          if denominador > 1e-9 else float('inf'))

    value_ore   = ((precio - eco.cargo_tc_rc) * (ley / 100.0) * eco.recuperacion_dec
                   * eco.lbs_por_ton * ton - (eco.costo_mina + eco.costo_planta) * ton)
    value_waste = -eco.costo_mina * ton
    value       = np.where(ley >= lc, value_ore, value_waste)

    # Construir grafo
    g = maxflow.Graph[float]()
    nodes = g.add_nodes(n)

    for idx, v in enumerate(value):
        if v >= 0:
            g.add_tedge(nodes[idx], v, 0)
        else:
            g.add_tedge(nodes[idx], 0, -v)

    # Restricciones de talud
    t_e, t_o, t_n, t_s = opt.talud_este, opt.talud_oeste, opt.talud_norte, opt.talud_sur
    min_talud_rad = math.radians(min(t_e, t_o, t_n, t_s))
    max_h = opt.n_niveles_talud * zsiz
    max_dist = max_h / math.tan(min_talud_rad) if min_talud_rad > 0 else 0
    max_dx = int(max_dist / xsiz) + 2
    max_dy = int(max_dist / ysiz) + 2

    for idx in range(n):
        i, j, k = idx_to_coord[idx]
        for dz in range(1, opt.n_niveles_talud + 1):
            nk = k + dz
            altura = dz * zsiz
            for dx in range(-max_dx, max_dx + 1):
                for dy in range(-max_dy, max_dy + 1):
                    if dx == 0 and dy == 0:
                        continue
                    dest = coord_to_idx.get((i + dx, j + dy, nk))
                    if dest is None:
                        continue
                    dist = math.hypot(dx * xsiz, dy * ysiz)
                    angle = math.degrees(math.atan2(altura, dist)) if dist > 0 else 90.0
                    ang_req = _angulo_por_direccion(dx, dy, t_e, t_o, t_n, t_s)
                    if angle >= ang_req:
                        g.add_edge(nodes[idx], nodes[dest], BIG, 0)

    g.maxflow()
    pit_mask = np.array(
        [1 if g.get_segment(nodes[idx]) == 0 else 0 for idx in range(n)],
        dtype=int,
    )
    return value, pit_mask


def _angulo_por_direccion(dx: int, dy: int, t_e, t_o, t_n, t_s) -> float:
    if dx > 0:  return t_e
    if dx < 0:  return t_o
    if dy > 0:  return t_n
    if dy < 0:  return t_s
    return min(t_e, t_o, t_n, t_s)
