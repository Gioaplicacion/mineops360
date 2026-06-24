"""
MineOps 360 — Configuración Central
====================================
Todos los parámetros del sistema viven aquí.
Nunca hardcodear valores en los módulos de cálculo.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BlockModelConfig:
    """Geometría del modelo de bloques."""
    xsiz: float = 10.0      # tamaño bloque en X (m)
    ysiz: float = 10.0      # tamaño bloque en Y (m)
    zsiz: float = 15.0      # tamaño bloque en Z (m)
    densidad: float = 2.5   # t/m³
    ley_nula: float = -99.0 # valor centinela para bloques sin ley (estéril)

    @property
    def volumen(self) -> float:
        return self.xsiz * self.ysiz * self.zsiz

    @property
    def tonelaje_bloque(self) -> float:
        return self.volumen * self.densidad


@dataclass
class EconomicConfig:
    """Parámetros económicos del proyecto."""
    precio_metal: float = 4.0    # USD/lb (cobre)
    cargo_tc_rc: float = 0.1     # USD/lb  (TC/RC)
    recuperacion: float = 90.0   # % recuperación metalúrgica
    costo_mina: float = 2.0      # USD/t minada
    costo_planta: float = 10.0   # USD/t procesada
    tasa_descuento: float = 0.10 # tasa anual (10%)
    lbs_por_ton: float = 2204.6  # conversión tonelada → libras

    @property
    def recuperacion_dec(self) -> float:
        return self.recuperacion / 100.0

    @property
    def ley_de_corte(self) -> float:
        """
        Ley de corte económica calculada dinámicamente.
        FIX: antes estaba hardcodeada en 0.318 en Valorizacion_modelo_de_bloques.py
        Fórmula: LC = (Cm + Cg) / ((Pr - Cf) * Rec * 2204.6)  [en %]
        """
        denominador = (self.precio_metal - self.cargo_tc_rc) * self.recuperacion_dec * self.lbs_por_ton
        if denominador <= 1e-9:
            return float('inf')
        return ((self.costo_mina + self.costo_planta) / denominador) * 100.0

    @property
    def ley_de_corte_marginal(self) -> float:
        """Ley de corte marginal (solo costo planta)."""
        denominador = (self.precio_metal - self.cargo_tc_rc) * self.recuperacion_dec * self.lbs_por_ton
        if denominador <= 1e-9:
            return float('inf')
        return (self.costo_planta / denominador) * 100.0


@dataclass
class PitOptimizerConfig:
    """Parámetros del optimizador MaxFlow (Lerchs-Grossmann)."""
    num_escenarios: int = 20        # número de Revenue Factors
    precio_base: float = 4.0       # USD/lb precio máximo
    n_niveles_talud: int = 6       # precisión del talud
    talud_este: float = 45.0       # grados
    talud_oeste: float = 45.0
    talud_norte: float = 45.0
    talud_sur: float = 45.0


@dataclass
class SchedulerConfig:
    """Parámetros del scheduler heurístico."""
    horizontes: int = 39            # períodos de simulación
    lag_fase: int = 3               # bancos de diferencia entre fases
    panel_size_x: float = 20.0     # tamaño panel X (m)
    panel_size_y: float = 20.0     # tamaño panel Y (m)
    cap_mineral_t: float = 35_000_000.0   # capacidad mina mineral (t/año)
    cap_movimiento_t: float = 70_000_000.0 # capacidad movimiento total (t/año)
    cap_planta_t: float = 30_000_000.0    # capacidad planta (t/año)
    costo_remanejo: float = 0.60   # USD/t remanejo stockpile
    costo_holding: float = 0.05    # USD/t·período inventario
    usar_lane: bool = False         # activa cutoff dinámico Lane
    lane_cutoffs: dict = field(default_factory=dict)  # {periodo: cutoff%}


@dataclass
class ProjectConfig:
    """Configuración completa del proyecto — punto de entrada único."""
    nombre: str = "Proyecto Minero"
    metal: str = "cu"              # columna de ley en el CSV
    bloque: BlockModelConfig = field(default_factory=BlockModelConfig)
    economico: EconomicConfig = field(default_factory=EconomicConfig)
    optimizador: PitOptimizerConfig = field(default_factory=PitOptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    @classmethod
    def desde_dict(cls, d: dict) -> "ProjectConfig":
        """Construye la config desde un dict (p.ej. JSON de la API)."""
        cfg = cls()
        cfg.nombre = d.get("nombre", cfg.nombre)
        cfg.metal  = d.get("metal", cfg.metal)

        b = d.get("bloque", {})
        cfg.bloque.xsiz      = float(b.get("xsiz", cfg.bloque.xsiz))
        cfg.bloque.ysiz      = float(b.get("ysiz", cfg.bloque.ysiz))
        cfg.bloque.zsiz      = float(b.get("zsiz", cfg.bloque.zsiz))
        cfg.bloque.densidad  = float(b.get("densidad", cfg.bloque.densidad))

        e = d.get("economico", {})
        cfg.economico.precio_metal   = float(e.get("precio_metal", cfg.economico.precio_metal))
        cfg.economico.cargo_tc_rc    = float(e.get("cargo_tc_rc", cfg.economico.cargo_tc_rc))
        cfg.economico.recuperacion   = float(e.get("recuperacion", cfg.economico.recuperacion))
        cfg.economico.costo_mina     = float(e.get("costo_mina", cfg.economico.costo_mina))
        cfg.economico.costo_planta   = float(e.get("costo_planta", cfg.economico.costo_planta))
        cfg.economico.tasa_descuento = float(e.get("tasa_descuento", cfg.economico.tasa_descuento))

        s = d.get("scheduler", {})
        cfg.scheduler.cap_mineral_t    = float(s.get("cap_mineral_t", cfg.scheduler.cap_mineral_t))
        cfg.scheduler.cap_movimiento_t = float(s.get("cap_movimiento_t", cfg.scheduler.cap_movimiento_t))
        cfg.scheduler.cap_planta_t     = float(s.get("cap_planta_t", cfg.scheduler.cap_planta_t))
        cfg.scheduler.usar_lane        = bool(s.get("usar_lane", cfg.scheduler.usar_lane))

        return cfg
