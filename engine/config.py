"""
MineOps 360 — Configuración Central
====================================
Todos los parámetros del sistema viven aquí.
Nunca hardcodear valores en los módulos de cálculo.
"""

from dataclasses import dataclass, field
from typing import Optional


# Unidades por mineral: (unidad de ley, unidad de precio, factor "lbs_por_ton" del motor).
# El motor calcula ingreso = precio * (ley/100) * lbs_por_ton * recuperación * ton.
#  - Cu, Zn, Mo...: ley en %, precio en USD/lb  -> 2204.6 lb por tonelada
#  - Au, Ag, Pt...: ley en g/t, precio en USD/oz -> 100/31.1035 (1 oz troy = 31.1035 g)
GRAMOS_POR_ONZA = 31.1035
METALES_ECO = {
    "cu": ("%",   "USD/lb", 2204.6),
    "zn": ("%",   "USD/lb", 2204.6),
    "pb": ("%",   "USD/lb", 2204.6),
    "mo": ("%",   "USD/lb", 2204.6),
    "ni": ("%",   "USD/lb", 2204.6),
    "co": ("%",   "USD/lb", 2204.6),
    "au": ("g/t", "USD/oz", 100.0 / GRAMOS_POR_ONZA),
    "ag": ("g/t", "USD/oz", 100.0 / GRAMOS_POR_ONZA),
    "pt": ("g/t", "USD/oz", 100.0 / GRAMOS_POR_ONZA),
    "pd": ("g/t", "USD/oz", 100.0 / GRAMOS_POR_ONZA),
}


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
    lbs_por_ton: float = 2204.6  # conversión tonelada → libras (Cu); para Au/Ag = 100/31.1035
    unidad_ley: str = "%"        # unidad de la ley del metal evaluado
    unidad_precio: str = "USD/lb" # unidad del precio del metal

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
    num_escenarios: int = 25        # número de pits anidados (Revenue Factors)
    rf_min: float = 0.30            # menor factor de ingresos (precio = rf × precio base)
    rf_max: float = 1.50            # mayor factor; >1 permite ver dónde cae el VAN
    num_fases_especificado: int = 4 # fases del "caso especificado" de la curva de VAN
    precio_base: float = 4.0       # USD/lb; se sincroniza con economico.precio_metal
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
    # Mediana minería (Instituto de Ingenieros de Minas de Chile: 300–8.000 t de mineral/día).
    # Planta ≈ 6.000 t/día; mina ≈ 6.600 t/día de mineral; movimiento total ≈ 20.500 t/día.
    cap_mineral_t: float = 2_400_000.0     # capacidad mina mineral (t/año)
    cap_movimiento_t: float = 7_500_000.0  # capacidad movimiento total (t/año)
    cap_planta_t: float = 2_200_000.0      # capacidad planta (t/año)
    costo_remanejo: float = 0.60   # USD/t remanejo stockpile
    costo_holding: float = 0.05    # USD/t·período inventario
    dias_efectivos: float = 365.0  # días operacionales efectivos por período
    # Capacidades alternativas en t/día (si > 0 reemplazan a las anuales: t/día × días efectivos)
    cap_mineral_tpd: float = 0.0
    cap_movimiento_tpd: float = 0.0
    cap_planta_tpd: float = 0.0
    precedencia_espacial: bool = True  # exige vecinos 3×3 del banco superior (talud) además de la columna
    cap_stock_t: float = 0.0       # capacidad máx. de inventario en stockpile (0 = sin límite)
    cap_recup_stock_t: float = 0.0 # máx. recuperación desde stockpile por período (0 = sin límite)
    usar_lane: bool = False         # activa cutoff dinámico Lane

    def cap_anual(self, nombre: str) -> float:
        """Capacidad anual efectiva: t/día × días efectivos si se dio en t/día; si no, la anual."""
        tpd = getattr(self, f"cap_{nombre}_tpd", 0.0)
        return tpd * self.dias_efectivos if tpd > 0 else getattr(self, f"cap_{nombre}_t")
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
        cfg.metal  = str(d.get("metal", cfg.metal)).lower()
        ul, up, fac = METALES_ECO.get(cfg.metal, METALES_ECO["cu"])
        cfg.economico.unidad_ley = ul
        cfg.economico.unidad_precio = up
        cfg.economico.lbs_por_ton = fac

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

        # El factor de ingresos 1,0 siempre corresponde al precio del proyecto
        cfg.optimizador.precio_base = cfg.economico.precio_metal

        o = d.get("optimizador", {})
        cfg.optimizador.num_escenarios = int(o.get("num_escenarios", cfg.optimizador.num_escenarios))
        cfg.optimizador.rf_min = float(o.get("rf_min", cfg.optimizador.rf_min))
        cfg.optimizador.rf_max = float(o.get("rf_max", cfg.optimizador.rf_max))

        s = d.get("scheduler", {})
        cfg.scheduler.cap_mineral_t    = float(s.get("cap_mineral_t", cfg.scheduler.cap_mineral_t))
        cfg.scheduler.cap_movimiento_t = float(s.get("cap_movimiento_t", cfg.scheduler.cap_movimiento_t))
        cfg.scheduler.cap_planta_t     = float(s.get("cap_planta_t", cfg.scheduler.cap_planta_t))
        for k in ("dias_efectivos", "cap_mineral_tpd", "cap_movimiento_tpd", "cap_planta_tpd",
                  "cap_stock_t", "cap_recup_stock_t"):
            setattr(cfg.scheduler, k, float(s.get(k, getattr(cfg.scheduler, k))))
        cfg.scheduler.usar_lane        = bool(s.get("usar_lane", cfg.scheduler.usar_lane))

        return cfg
