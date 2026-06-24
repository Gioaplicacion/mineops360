"""
MineOps 360 — Cargador y Validador del Modelo de Bloques
==========================================================
FIX aplicados vs código original:
  1. Filtro de Ley=-99 (bloques nulos / estéril sin valor)
  2. Detección automática de columnas (alias X/Y/Z/Cu)
  3. Validación de geometría y reporte de calidad
  4. Cálculo de tonelaje dinámico desde config (no hardcodeado)
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import ProjectConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Aliases aceptados para cada columna canónica
# ---------------------------------------------------------------------------
ALIAS_MAP = {
    "x":  ["x", "east", "este", "x(m)", "x(ft)", "xcoord"],
    "y":  ["y", "north", "norte", "y(m)", "y(ft)", "ycoord"],
    "z":  ["z", "elev", "elevation", "z(m)", "z(ft)", "zcoord", "cota"],
    "ley": ["cu", "au", "ag", "ley", "grade", "tenor", "au(oz/ton)", "cut"],
}


class ModeloBloques:
    """
    Encapsula el modelo de bloques validado y enriquecido.
    Todos los cálculos económicos se hacen con parámetros de ProjectConfig.
    """

    def __init__(self, df: pd.DataFrame, config: ProjectConfig):
        self.df = df
        self.config = config
        self._xsiz: Optional[float] = None
        self._ysiz: Optional[float] = None
        self._zsiz: Optional[float] = None

    # ------------------------------------------------------------------
    # Propiedades de grilla
    # ------------------------------------------------------------------
    @property
    def xsiz(self) -> float:
        if self._xsiz is None:
            self._xsiz = _inferir_paso(self.df["x"])
        return self._xsiz

    @property
    def ysiz(self) -> float:
        if self._ysiz is None:
            self._ysiz = _inferir_paso(self.df["y"])
        return self._ysiz

    @property
    def zsiz(self) -> float:
        if self._zsiz is None:
            self._zsiz = _inferir_paso(self.df["z"])
        return self._zsiz

    @property
    def ton_bloque(self) -> float:
        return self.xsiz * self.ysiz * self.zsiz * self.config.bloque.densidad

    @property
    def resumen(self) -> dict:
        """Estadísticas básicas del modelo para mostrar en UI."""
        df = self.df
        df_mineral = df[df["es_mineral"]]
        return {
            "total_bloques":    len(df),
            "bloques_mineral":  int(df["es_mineral"].sum()),
            "bloques_esteril":  int((~df["es_mineral"]).sum()),
            "tonelaje_total_Mt": round(df["tonelaje"].sum() / 1e6, 2),
            "tonelaje_mineral_Mt": round(df_mineral["tonelaje"].sum() / 1e6, 2),
            "ley_promedio_pct": round(
                (df_mineral["ley"] * df_mineral["tonelaje"]).sum()
                / df_mineral["tonelaje"].sum() * 100
                if len(df_mineral) > 0 else 0.0, 4
            ),
            "ley_corte_calculada_pct": round(self.config.economico.ley_de_corte, 4),
            "x_range": [float(df["x"].min()), float(df["x"].max())],
            "y_range": [float(df["y"].min()), float(df["y"].max())],
            "z_range": [float(df["z"].min()), float(df["z"].max())],
        }


# ---------------------------------------------------------------------------
# Función principal de carga
# ---------------------------------------------------------------------------
def cargar_modelo(
    path: str | Path,
    config: ProjectConfig,
    sep: str = "auto",
) -> ModeloBloques:
    """
    Carga, limpia y valida el modelo de bloques desde un CSV.

    Args:
        path:   Ruta al archivo CSV.
        config: Configuración del proyecto (parámetros económicos y de bloque).
        sep:    Separador CSV. "auto" lo detecta automáticamente.

    Returns:
        ModeloBloques listo para usar en los módulos downstream.

    Raises:
        ValueError: Si el archivo no tiene las columnas mínimas requeridas.
        FileNotFoundError: Si el archivo no existe.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {path}")

    logger.info(f"Cargando modelo de bloques: {path}")

    # --- 1. Leer CSV con detección automática de separador ---
    try:
        df = (
            pd.read_csv(path, engine="python")
            if sep == "auto"
            else pd.read_csv(path, sep=sep)
        )
    except Exception as e:
        raise ValueError(f"No se pudo leer el CSV: {e}") from e

    logger.info(f"  Filas leídas: {len(df):,} | Columnas: {list(df.columns)}")

    # --- 2. Normalizar nombres de columnas ---
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = _renombrar_columnas(df, config.metal)

    # --- 3. Convertir a numérico y eliminar NaN estructurales ---
    for col in ["x", "y", "z", "ley"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    n_antes = len(df)
    df = df.dropna(subset=["x", "y", "z", "ley"]).reset_index(drop=True)
    if len(df) < n_antes:
        logger.warning(f"  Eliminados {n_antes - len(df):,} bloques con NaN en X/Y/Z/Ley")

    # --- 4. FIX CRÍTICO: filtrar valor centinela Ley=-99 ---
    #    En el código original estos bloques entraban al MaxFlow y
    #    distorsionaban los pits. Ahora se marcan como estéril explícito.
    ley_nula = config.bloque.ley_nula
    mask_nula = df["ley"] <= ley_nula
    n_nulos = mask_nula.sum()
    if n_nulos > 0:
        logger.warning(
            f"  FIX Ley={ley_nula}: {n_nulos:,} bloques ({n_nulos/len(df)*100:.1f}%) "
            f"marcados como estéril (ley→0)"
        )
        df.loc[mask_nula, "ley"] = 0.0

    # Ley negativa residual (no centinela) → log y fijar a 0
    mask_neg = df["ley"] < 0
    if mask_neg.sum() > 0:
        logger.warning(f"  {mask_neg.sum()} bloques con ley < 0 (no centinela) → fijados a 0")
        df.loc[mask_neg, "ley"] = 0.0

    # --- 5. Calcular tonelaje por bloque ---
    #    FIX: antes era xsiz*ysiz*zsiz*2.5 hardcodeado en cada script
    xsiz = _inferir_paso(df["x"])
    ysiz = _inferir_paso(df["y"])
    zsiz = _inferir_paso(df["z"])
    ton_bloque = xsiz * ysiz * zsiz * config.bloque.densidad
    df["tonelaje"] = ton_bloque

    logger.info(f"  Tamaño bloque inferido: {xsiz}×{ysiz}×{zsiz} m → {ton_bloque:,.0f} t/bloque")

    # --- 6. Clasificación mineral / estéril con ley de corte dinámica ---
    lc = config.economico.ley_de_corte
    df["es_mineral"] = df["ley"] >= lc
    logger.info(
        f"  Ley de corte dinámica: {lc:.4f}% | "
        f"Mineral: {df['es_mineral'].sum():,} bloques | "
        f"Estéril: {(~df['es_mineral']).sum():,} bloques"
    )

    # --- 7. Valorización económica por bloque ---
    df["value"] = _calcular_value(df["ley"].values, config, ton_bloque)

    modelo = ModeloBloques(df, config)
    logger.info(f"  ✅ Modelo cargado: {modelo.resumen}")
    return modelo


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------
def _renombrar_columnas(df: pd.DataFrame, ley_var: str) -> pd.DataFrame:
    """Mapea aliases de columnas a nombres canónicos (x, y, z, ley)."""
    rename = {}
    cols = set(df.columns)

    for canon, aliases in ALIAS_MAP.items():
        # Para la ley, el alias principal es el metal del proyecto
        if canon == "ley":
            candidatos = [ley_var.lower()] + aliases
        else:
            candidatos = aliases

        for alias in candidatos:
            if alias in cols and alias not in rename.values():
                rename[alias] = canon
                break

    df = df.rename(columns=rename)

    # Verificar columnas mínimas
    faltantes = [c for c in ["x", "y", "z", "ley"] if c not in df.columns]
    if faltantes:
        raise ValueError(
            f"Columnas requeridas no encontradas: {faltantes}. "
            f"Columnas disponibles: {list(df.columns)}. "
            f"Aliases aceptados: {ALIAS_MAP}"
        )
    return df


def _inferir_paso(serie: pd.Series) -> float:
    """Infiere el tamaño de celda a partir de las coordenadas únicas."""
    vals = np.sort(serie.dropna().unique())
    diffs = np.diff(vals)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        raise ValueError("No se puede inferir el tamaño de celda — coordenadas constantes.")
    return round(float(np.min(diffs)), 4)


def _calcular_value(ley: np.ndarray, config: ProjectConfig, ton: float) -> np.ndarray:
    """
    Valorización económica vectorizada por bloque.
    FIX: ley de corte calculada desde config, no hardcodeada.
    """
    eco = config.economico
    lc = eco.ley_de_corte

    # Valor bloque mineral
    value_mineral = (
        (eco.precio_metal - eco.cargo_tc_rc)
        * (ley / 100.0)
        * eco.recuperacion_dec
        * eco.lbs_por_ton
        * ton
        - (eco.costo_mina + eco.costo_planta) * ton
    )
    # Valor bloque estéril (solo costo de minado)
    value_esteril = -eco.costo_mina * ton

    return np.where(ley >= lc, value_mineral, value_esteril)
