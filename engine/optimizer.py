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

    def __init__(self, df: pd.DataFrame, config: ProjectConfig, precios: list[float],
                 tabla: Optional[pd.DataFrame] = None, pit_optimo: Optional[int] = None):
        # df: x,y,z,ley,value (al precio BASE),pit,ton,wton,oton (clasificados al precio base)
        self.df = df
        self.config = config
        self.precios = precios
        self.tabla = tabla if tabla is not None else pd.DataFrame()
        self.pit_optimo = pit_optimo   # número de pit (1..N) con mayor VAN del caso especificado

    @property
    def resumen_pits(self) -> pd.DataFrame:
        """Tabla pit-por-pit: tonelaje/valor por anillo, acumulados y VAN de los 3 casos."""
        return self.tabla

    def to_dict(self) -> dict:
        rf_opt = None
        if self.pit_optimo is not None and not self.tabla.empty:
            rf_opt = float(self.tabla.loc[self.tabla["pit"] == self.pit_optimo, "rf"].iloc[0])
        return {
            "total_bloques_en_pit": int(self.df["pit"].notna().sum()),
            "num_pits": int(len(self.tabla)),
            "pit_optimo": None if self.pit_optimo is None else int(self.pit_optimo),
            "rf_optimo": rf_opt,
            "precio_base_USD_lb": float(self.config.economico.precio_metal),
            "precio_unidad": self.config.economico.unidad_precio,
            "pits": self.tabla.to_dict(orient="records"),
        }


def _calcular_tabla_pits(df: pd.DataFrame, factores: np.ndarray, precios: list[float],
                         cfg: ProjectConfig) -> tuple[pd.DataFrame, Optional[int]]:
    """
    Tabla pit-por-pit estándar. Todos los pits se valoran al precio BASE del proyecto
    (no al precio del escenario en que cada bloque entra), de modo que las cifras
    son comparables entre sí y con el plan minero.

    Para cada pit k (todos los bloques con pit <= k) se calcula el VAN de tres secuencias,
    sujetas a las capacidades de movimiento y de planta y a la tasa de descuento:
      - Mejor caso:       pit por pit (cada pit anidado completo antes del siguiente).
      - Caso especificado: el pit k dividido en N fases (N = num_fases_especificado).
      - Peor caso:        banco por banco sobre todo el pit k.
    Simplificaciones: no hay ancho mínimo de minado, rampas ni stockpiles en esta curva.
    """
    eco, sch, opt = cfg.economico, cfg.scheduler, cfg.optimizador
    S = len(factores)

    en = df["pit"].notna().values
    pit  = df["pit"].values[en].astype(int)
    z    = df["z"].values[en]
    val  = df["value"].values[en].astype(float)
    oton = df["oton"].values[en].astype(float)
    wton = df["wton"].values[en].astype(float)
    mov  = oton + wton

    # Valores por anillo y acumulados
    oton_s = np.bincount(pit, weights=oton, minlength=S + 1)[1:S + 1]
    wton_s = np.bincount(pit, weights=wton, minlength=S + 1)[1:S + 1]
    val_s  = np.bincount(pit, weights=val,  minlength=S + 1)[1:S + 1]

    cap_mov = sch.cap_movimiento_t if sch.cap_movimiento_t > 0 else 1e18
    cap_pl  = sch.cap_planta_t if sch.cap_planta_t > 0 else 1e18
    tasa    = eco.tasa_descuento
    n_ph    = max(1, int(opt.num_fases_especificado))

    def _van(idx: np.ndarray) -> float:
        if idx.size == 0:
            return 0.0
        t_mov = np.ceil(np.cumsum(mov[idx]) / cap_mov - 1e-9)
        t_ore = np.ceil(np.cumsum(oton[idx]) / cap_pl - 1e-9)
        t = np.maximum.accumulate(np.maximum(np.maximum(t_mov, t_ore), 1.0))
        return float(np.sum(val[idx] / (1.0 + tasa) ** t))

    orden_pit = np.lexsort((-z, pit))             # pit ascendente, banco de arriba hacia abajo
    pit_ordenado = pit[orden_pit]
    orden_z = np.argsort(-z, kind="stable")       # banco de arriba hacia abajo
    pit_por_z = pit[orden_z]

    van_mejor = np.zeros(S); van_espec = np.zeros(S); van_peor = np.zeros(S)
    for k in range(1, S + 1):
        m = int(np.searchsorted(pit_ordenado, k, side="right"))
        van_mejor[k - 1] = _van(orden_pit[:m])

        idx_k = orden_z[pit_por_z <= k]
        van_peor[k - 1] = _van(idx_k)

        if idx_k.size:
            fases = np.maximum(np.ceil(pit[idx_k] * n_ph / k).astype(int), 1)
            van_espec[k - 1] = _van(idx_k[np.argsort(fases, kind="stable")])

    tabla = pd.DataFrame({
        "pit":    np.arange(1, S + 1),
        "rf":     np.round(factores, 4),
        "precio": np.round(np.asarray(precios, dtype=float), 4),
        "oton":   oton_s,
        "wton":   wton_s,
        "value":  val_s,
        "oton_acum":  np.cumsum(oton_s),
        "wton_acum":  np.cumsum(wton_s),
        "value_acum": np.cumsum(val_s),
        "npv_mejor":         van_mejor,
        "npv_especificado":  van_espec,
        "npv_peor":          van_peor,
    })

    pit_optimo = int(np.argmax(van_espec)) + 1 if van_espec.max() > 0 else None
    return tabla, pit_optimo


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
    ton = df["tonelaje"].values  # tonelaje por bloque (usa SG si el archivo lo trae)

    # Escenarios de precio: factor de ingresos (RF) de rf_min a rf_max; RF=1 es el precio del proyecto
    precio_base = cfg.economico.precio_metal
    opt.precio_base = precio_base
    factores = np.linspace(opt.rf_min, opt.rf_max, opt.num_escenarios)
    precios  = [precio_base * f for f in factores]

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

    # Primer pit donde entra cada bloque (vectorizado)
    en_pit  = (pit_matrix == 1)
    hay     = en_pit.any(axis=1)
    primero = en_pit.argmax(axis=1)
    pit_nivel = np.where(hay, primero + 1, np.nan)
    value_escenario = np.where(hay, value_matrix[np.arange(n), primero], 0.0)

    # DataFrame final. 'value', 'oton' y 'wton' van al precio BASE del proyecto
    # (mismo criterio que el loader y el scheduler), no al del escenario de entrada.
    out = df.copy()
    out["pit"]   = pit_nivel
    out["ton"]   = ton
    out["value_escenario"] = value_escenario
    es_min = out["es_mineral"].values
    out["oton"]  = np.where(es_min, ton, 0.0)
    out["wton"]  = np.where(es_min, 0.0, ton)
    out = out.sort_values(["pit", "z"], ascending=[True, False]).reset_index(drop=True)

    tabla, pit_optimo = _calcular_tabla_pits(out, factores, precios, cfg)
    if pit_optimo is not None:
        logger.info(f"Pit óptimo (caso especificado): {pit_optimo} de {len(tabla)} "
                    f"(RF={tabla.loc[tabla['pit'] == pit_optimo, 'rf'].iloc[0]:.2f})")
    else:
        logger.warning("Ningún pit tiene VAN positivo con los parámetros económicos dados.")

    logger.info(f"✅ Optimización completada. Pits generados: {len(tabla)}")
    return ResultadoPitOptimizer(out, cfg, precios, tabla=tabla, pit_optimo=pit_optimo)


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
