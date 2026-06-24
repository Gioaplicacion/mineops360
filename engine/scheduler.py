"""
MineOps 360 — Scheduler Heurístico (Fase-Banco)
=================================================
Unifica Heuristica_fase_banco_con_stock.py y _con_Lane.py en una sola clase.

FIX aplicados vs código original:
  1. Clave 17 duplicada en SECONDARY_ADVANCE_DIR → eliminada
  2. Cutoff económico dinámico (Lane) como parámetro, no como archivo externo hardcodeado
  3. Código DRY: una sola clase en vez de dos scripts ~80% idénticos
  4. Progress callback para streaming a la UI
  5. Logging estructurado en vez de print()
  6. Resultado como objeto con métricas, no solo CSV
"""

import logging
import time
from collections import defaultdict
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .config import ProjectConfig, SchedulerConfig
from .optimizer import ResultadoPitOptimizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stockpile (igual en ambas versiones originales, extraído como clase)
# ---------------------------------------------------------------------------
class Stockpile:
    """Gestiona inventario de un tipo de material (marginal o económico)."""

    def __init__(self):
        self._ton: float = 0.0
        self._metal: float = 0.0

    def ingresar(self, ton: float, metal: float) -> None:
        self._ton   += max(0.0, ton)
        self._metal += max(0.0, metal)

    def retirar(self, ton_pedido: float) -> tuple[float, float]:
        """Retira hasta ton_pedido. Devuelve (ton_retirada, metal_retirado)."""
        ton_real = min(ton_pedido, self._ton)
        if self._ton > 1e-9:
            ley_sp = self._metal / self._ton
        else:
            ley_sp = 0.0
        metal_real  = ton_real * ley_sp
        self._ton   -= ton_real
        self._metal -= metal_real
        return ton_real, metal_real

    def get_inventory(self) -> tuple[float, float]:
        return self._ton, self._metal


# ---------------------------------------------------------------------------
# Resultado del scheduler
# ---------------------------------------------------------------------------
class ResultadoScheduler:
    """Contiene el plan minero período a período y el modelo de bloques schedulado."""

    def __init__(self, plan_df: pd.DataFrame, bloques_df: pd.DataFrame,
                 van_total: float, config: ProjectConfig):
        self.plan     = plan_df      # un row por período
        self.bloques  = bloques_df   # modelo de bloques con columna 'periodo'
        self.van_total = van_total
        self.config   = config

    def to_dict(self) -> dict:
        return {
            "van_total_USD":   round(self.van_total, 2),
            "van_total_MUSD":  round(self.van_total / 1e6, 4),
            "periodos":        len(self.plan),
            "plan":            self.plan.to_dict(orient="records"),
        }


# ---------------------------------------------------------------------------
# Scheduler principal
# ---------------------------------------------------------------------------
class HeuristicaFaseBanco:
    """
    Scheduler heurístico que genera el plan minero período a período.

    Soporta dos modos:
      - Cutoff fijo:    config.scheduler.usar_lane = False
      - Cutoff Lane:    config.scheduler.usar_lane = True  (cutoffs en scheduler.lane_cutoffs)
    """

    # FIX: dirección de avance configurable (antes hardcodeada para 20 fases iguales)
    DIR_PRIMARIA_DEFAULT  = "Y-DEC"
    DIR_SECUNDARIA_DEFAULT = "X-INC"

    def __init__(
        self,
        resultado_optimizer: ResultadoPitOptimizer,
        fases_df: pd.DataFrame,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
    ):
        """
        Args:
            resultado_optimizer: Resultado del paso anterior (pits anidados).
            fases_df:  DataFrame con columnas X,Y,Z,fase,Ley,tonelaje
                       (resultado del faseamiento — puede venir del optimizer o de un CSV externo).
            progress_cb: Callback para streaming de progreso a la UI.
        """
        self.opt_result  = resultado_optimizer
        self.config      = resultado_optimizer.config
        self.sch         = self.config.scheduler
        self.eco         = self.config.economico
        self.fases_df    = fases_df.copy()
        self.progress_cb = progress_cb or (lambda *a: None)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def ejecutar(self) -> ResultadoScheduler:
        t0 = time.time()
        cfg = self.sch
        df  = self.fases_df

        # Validar columnas
        for col in ["X", "Y", "Z", "fase", "Ley", "tonelaje"]:
            if col not in df.columns:
                raise ValueError(f"Columna requerida faltante en fases_df: '{col}'")

        df["fase"] = df["fase"].astype(int)
        phases = sorted(df["fase"].unique())
        logger.info(f"Fases detectadas: {phases} | Bloques: {len(df):,}")

        # Parámetros económicos
        lbs      = self.eco.lbs_por_ton
        precio   = self.eco.precio_metal
        descuento = self.eco.cargo_tc_rc
        recup    = self.eco.recuperacion_dec
        valor_por_ton_proc = (precio - descuento) * recup * lbs  # USD/t·%Cu → se multiplica por ley

        # Cutoffs
        lc_marginal = self.eco.ley_de_corte_marginal
        lc_vector   = self._construir_vector_cutoff(lc_marginal)

        disc_factor = np.array([1 / (1 + self.eco.tasa_descuento) ** t for t in range(cfg.horizontes + 1)])

        # Panelización
        df["panel_X"] = (df["X"] // cfg.panel_size_x) * cfg.panel_size_x
        df["panel_Y"] = (df["Y"] // cfg.panel_size_y) * cfg.panel_size_y
        df["panel_Z"] = df["Z"]

        paneles_df = self._panelizar(df)
        nU = len(paneles_df)
        logger.info(f"Paneles generados: {nU:,}")

        # Estructuras de secuencia
        panel_key_to_u = {
            tuple(row[:4]): idx
            for idx, row in paneles_df[["fase", "panel_X", "panel_Y", "panel_Z"]].iterrows()
        }
        unique_benches = sorted(paneles_df["panel_Z"].unique(), reverse=True)
        bench_idx_map  = {pz: i for i, pz in enumerate(unique_benches)}

        precedencias   = self._construir_precedencias(paneles_df, panel_key_to_u, bench_idx_map, unique_benches)
        panel_sequence, panel_cursors, fase_bancos = self._construir_secuencias(paneles_df, phases)
        top_bench      = self._calcular_top_bench(paneles_df, phases, bench_idx_map)

        # Estado de la simulación
        mined_mask   = np.zeros(nU, dtype=bool)
        mined_period = np.zeros(nU, dtype=int)
        sp_marginal  = Stockpile()
        sp_economic  = Stockpile()
        res_rows     = []
        total_van    = 0.0

        # Período de inicio por fase (lag)
        fase_start_period = {f: (i * cfg.lag_fase + 1) for i, f in enumerate(phases)}

        # ─── BUCLE PRINCIPAL ───────────────────────────────────────────
        for t in range(1, cfg.horizontes + 1):
            self.progress_cb(t, cfg.horizontes, f"Período {t}/{cfg.horizontes}")

            lc_econ_t = lc_vector[t]
            discount_t = disc_factor[t]

            tons_ore_t   = 0.0
            tons_waste_t = 0.0
            mined_details = []  # [(u, ton_econ, metal_econ, ton_marg, metal_marg)]

            remaining_ore   = float(cfg.cap_mineral_t)
            remaining_mina  = float(cfg.cap_movimiento_t)

            # Minar paneles por fase y banco
            for fase in phases:
                if t < fase_start_period.get(fase, 1):
                    continue
                for pz in fase_bancos[fase]:
                    key = (fase, pz)
                    if key not in panel_sequence:
                        continue
                    seq  = panel_sequence[key]
                    cur  = panel_cursors.get(key, 0)

                    while cur < len(seq):
                        u = seq[cur]
                        if mined_mask[u]:
                            cur += 1
                            continue

                        # Verificar precedencia vertical
                        if any(not mined_mask[p] for p in precedencias[u]):
                            break

                        row = paneles_df.iloc[u]
                        ton_total = row["ton_total"]

                        if ton_total <= 0:
                            mined_mask[u] = True
                            mined_period[u] = t
                            cur += 1
                            continue

                        # Clasificar con cutoff dinámico del período
                        ley_bloque = row["metal_total"] / ton_total * 100.0 if ton_total > 0 else 0.0
                        if ley_bloque >= lc_econ_t:
                            ton_econ  = ton_total
                            ton_marg  = 0.0
                            ton_waste = 0.0
                        elif ley_bloque >= lc_marginal:
                            ton_econ  = 0.0
                            ton_marg  = ton_total
                            ton_waste = 0.0
                        else:
                            ton_econ  = 0.0
                            ton_marg  = 0.0
                            ton_waste = ton_total

                        ton_ore_panel = ton_econ + ton_marg

                        # Verificar capacidades
                        if remaining_ore < ton_ore_panel - 1.0:
                            break
                        if remaining_mina < ton_total - 1.0:
                            break

                        # Minar
                        metal_econ = (row["metal_total"] * ton_econ / ton_total) if ton_total > 0 else 0.0
                        metal_marg = (row["metal_total"] * ton_marg / ton_total) if ton_total > 0 else 0.0

                        mined_mask[u]   = True
                        mined_period[u] = t
                        tons_ore_t   += ton_ore_panel
                        tons_waste_t += ton_waste
                        remaining_ore  -= ton_ore_panel
                        remaining_mina -= ton_total
                        mined_details.append((u, ton_econ, metal_econ, ton_marg, metal_marg))
                        cur += 1

                    panel_cursors[key] = cur

            # ─── PLANTA ────────────────────────────────────────────────
            metal_econ_t  = sum(d[2] for d in mined_details)
            metal_marg_t  = sum(d[4] for d in mined_details)
            ton_econ_t    = sum(d[1] for d in mined_details)
            ton_marg_t    = sum(d[3] for d in mined_details)

            # Enviar mineral económico a planta, marginal a stockpile
            sp_economic.ingresar(ton_econ_t, metal_econ_t)
            sp_marginal.ingresar(ton_marg_t, metal_marg_t)

            # Pull desde stockpile para completar planta
            feed_total = 0.0
            metal_feed = 0.0
            ton_fresh  = 0.0

            cap_planta = cfg.cap_planta_t
            # Primero feed fresco (económico ya en SP)
            pull_econ, metal_pull_econ = sp_economic.retirar(min(cap_planta, ton_econ_t))
            feed_total += pull_econ
            metal_feed += metal_pull_econ
            ton_fresh  += pull_econ

            deficit = cap_planta - feed_total
            pull_marg_econ, metal_marg_econ = sp_economic.retirar(min(deficit, 0.0))  # econ sp vacío
            pull_sp_marg, metal_sp_marg = sp_marginal.retirar(min(deficit, sp_marginal._ton))
            feed_total += pull_sp_marg
            metal_feed += metal_sp_marg

            # Economía período
            ley_head_t = (metal_feed / feed_total * 100.0) if feed_total > 1e-9 else 0.0
            ingresos   = valor_por_ton_proc * metal_feed
            c_mina     = self.eco.costo_mina * (tons_ore_t + tons_waste_t)
            c_planta   = self.eco.costo_planta * feed_total
            c_rehandle = self.sch.costo_remanejo * pull_sp_marg
            inv_m, _   = sp_marginal.get_inventory()
            inv_e, _   = sp_economic.get_inventory()
            c_hold     = self.sch.costo_holding * (inv_m + inv_e)

            van_t = (ingresos - c_mina - c_planta - c_rehandle - c_hold) * discount_t
            total_van += van_t

            res_rows.append({
                "periodo":              t,
                "mineral_mined_Mt":     tons_ore_t / 1e6,
                "esteril_Mt":           tons_waste_t / 1e6,
                "mov_total_Mt":         (tons_ore_t + tons_waste_t) / 1e6,
                "mina_ley_mined_pct":   (sum(d[2]+d[4] for d in mined_details) / tons_ore_t * 100.0)
                                        if tons_ore_t > 1e-9 else 0.0,
                "planta_feed_Mt":       feed_total / 1e6,
                "planta_ley_head_pct":  ley_head_t,
                "sp_inventario_Mt":     (inv_m + inv_e) / 1e6,
                "VAN_net_MUSD":         van_t / 1e6,
                "cutoff_econ_pct":      lc_econ_t,
            })

            if tons_ore_t < 1.0 and tons_waste_t < 1.0:
                if not np.any(~mined_mask):
                    logger.info(f"Modelo completamente minado en período {t}.")
                    break

        # ─── ARMAR RESULTADO ───────────────────────────────────────────
        plan_df = pd.DataFrame(res_rows)
        plan_df["VAN_acum_MUSD"] = plan_df["VAN_net_MUSD"].cumsum()

        # Mapear período al modelo de bloques original
        panel_key_period_map = {}
        for u in range(nU):
            if mined_period[u] > 0:
                r = paneles_df.iloc[u]
                panel_key_period_map[(int(r["fase"]), r["panel_X"], r["panel_Y"], r["panel_Z"])] = mined_period[u]

        df["periodo"] = df.apply(
            lambda row: panel_key_period_map.get(
                (row["fase"], row["panel_X"], row["panel_Y"], row["panel_Z"]), 0
            ), axis=1
        )

        bloques_out = df[["X", "Y", "Z", "tonelaje", "Ley", "fase", "periodo"]].copy()

        logger.info(f"✅ Scheduler completado en {time.time()-t0:.1f}s | VAN={total_van/1e6:,.2f} MUSD")
        return ResultadoScheduler(plan_df, bloques_out, total_van, self.config)

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------
    def _construir_vector_cutoff(self, lc_marginal: float) -> np.ndarray:
        """Construye vector de cutoffs por período (fijo o Lane dinámico)."""
        H = self.sch.horizontes
        vector = np.full(H + 1, self.eco.ley_de_corte)

        if self.sch.usar_lane and self.sch.lane_cutoffs:
            lane = self.sch.lane_cutoffs
            ultimo = self.eco.ley_de_corte
            for t in range(1, H + 1):
                if t in lane:
                    ultimo = lane[t]
                vector[t] = max(ultimo, lc_marginal)
            logger.info(f"Cutoffs Lane activos: P1={vector[1]:.3f}% | P{H}={vector[H]:.3f}%")
        else:
            logger.info(f"Cutoff fijo: {vector[1]:.4f}%")

        return vector

    def _panelizar(self, df: pd.DataFrame) -> pd.DataFrame:
        """Agrega bloques en paneles por (fase, panel_X, panel_Y, panel_Z)."""
        paneles = (
            df.groupby(["fase", "panel_X", "panel_Y", "panel_Z"])
            .agg(ton_total=("tonelaje", "sum"), metal_total=("Ley", lambda x: (x * df.loc[x.index, "tonelaje"] / 100.0).sum()))
            .reset_index()
        )
        return paneles[paneles["ton_total"] > 0.1].reset_index(drop=True)

    def _construir_precedencias(self, paneles_df, panel_key_to_u, bench_idx_map, unique_benches) -> dict:
        prec = defaultdict(list)
        for u, row in paneles_df.iterrows():
            fase, px, py, pz = row["fase"], row["panel_X"], row["panel_Y"], row["panel_Z"]
            b_idx = bench_idx_map.get(pz)
            if b_idx is not None and b_idx > 0:
                pz_arriba = unique_benches[b_idx - 1]
                u_arriba  = panel_key_to_u.get((fase, px, py, pz_arriba))
                if u_arriba is not None:
                    prec[u].append(u_arriba)
        return prec

    def _construir_secuencias(self, paneles_df, phases) -> tuple:
        """
        FIX: usa DIR_PRIMARIA_DEFAULT y DIR_SECUNDARIA_DEFAULT como fallback
        en vez de diccionarios hardcodeados de 20 entradas idénticas.
        """
        panel_sequence: dict = {}
        panel_cursors:  dict = {}
        fase_bancos:    dict = defaultdict(list)

        for fase in phases:
            prim = self.DIR_PRIMARIA_DEFAULT
            secu = self.DIR_SECUNDARIA_DEFAULT

            sort_cols, sort_asc = [], []
            for d, cols, asc in [(prim, sort_cols, sort_asc), (secu, sort_cols, sort_asc)]:
                if "Y-DEC" in d: cols.append("panel_Y"); asc.append(False)
                elif "Y-INC" in d: cols.append("panel_Y"); asc.append(True)
                elif "X-INC" in d: cols.append("panel_X"); asc.append(True)
                elif "X-DEC" in d: cols.append("panel_X"); asc.append(False)

            if "panel_X" not in sort_cols: sort_cols.append("panel_X"); sort_asc.append(True)
            if "panel_Y" not in sort_cols: sort_cols.append("panel_Y"); sort_asc.append(True)

            df_fase = paneles_df[paneles_df["fase"] == fase]
            bancos  = sorted(df_fase["panel_Z"].unique(), reverse=True)
            fase_bancos[fase] = bancos

            for pz in bancos:
                df_b = df_fase[df_fase["panel_Z"] == pz].sort_values(sort_cols, ascending=sort_asc)
                key  = (fase, pz)
                panel_sequence[key] = list(df_b.index)
                panel_cursors[key]  = 0

        return panel_sequence, panel_cursors, fase_bancos

    def _calcular_top_bench(self, paneles_df, phases, bench_idx_map) -> dict:
        top = {}
        for fase in phases:
            df_f = paneles_df[paneles_df["fase"] == fase]
            if not df_f.empty:
                top[fase] = bench_idx_map[df_f["panel_Z"].max()]
        return top
