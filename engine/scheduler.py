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
                 van_total: float, config: ProjectConfig,
                 fase_periodo: Optional[pd.DataFrame] = None, interrupciones: Optional[list] = None):
        self.fase_periodo = fase_periodo if fase_periodo is not None else pd.DataFrame()
        self.interrupciones = interrupciones or []
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
            "unidad_metal":    "t" if self.config.economico.unidad_ley == "%" else "oz",
            "fase_periodo":    self.fase_periodo.round(4).to_dict(orient="records"),
            "interrupciones":  self.interrupciones[:300],
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
        cap_ore = float(cfg.cap_anual('mineral'))
        cap_mov = float(cfg.cap_anual('movimiento'))
        cap_pl  = float(cfg.cap_anual('planta'))
        cap_stock = float(cfg.cap_stock_t) if cfg.cap_stock_t > 0 else float('inf')
        cap_recup = float(cfg.cap_recup_stock_t) if cfg.cap_recup_stock_t > 0 else float('inf')

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
        fase_per_rows = []
        interrupciones = []
        sp_marginal  = Stockpile()
        sp_economic  = Stockpile()
        res_rows     = []
        total_van    = 0.0

        # ── Plan CONSTANTE (observación del ingeniero) ─────────────────
        # * Una fase que se abre NO se detiene: queda abierta hasta agotarse.
        # * Tasa constante de mineral (cap_mineral_t) y de movimiento total
        #   (meta = movimiento total / períodos estimados, tope cap_movimiento_t).
        # * Si la(s) fase(s) abiertas no alcanzan para llenar la tasa, se abre la
        #   siguiente (aunque no se cumpla el lag) en vez de dejar capacidad ociosa.
        ton_arr   = paneles_df["ton_total"].values
        fase_arr  = paneles_df["fase"].values
        ley_arr   = np.where(ton_arr > 0, paneles_df["metal_total"].values / np.maximum(ton_arr, 1e-9) * 100.0, 0.0)
        ore_total = float(ton_arr[ley_arr >= lc_marginal].sum())
        mov_total = float(ton_arr.sum())
        t_est     = int(max(1.0, np.ceil(ore_total / max(cap_ore, 1.0))))   # vida estimada del pit
        meta_mov  = float(min(cap_mov, 1.02 * mov_total / t_est))
        meta_mov  = max(meta_mov, float(cap_ore))
        logger.info(f"Plan constante: mineral {cap_ore/1e6:.2f} Mt/período | movimiento meta {meta_mov/1e6:.2f} Mt/período | {t_est} períodos estimados")

        fases_abiertas = {phases[0]}
        fase_prev = {f: (phases[i - 1] if i > 0 else None) for i, f in enumerate(phases)}
        pendientes = {k: list(v) for k, v in panel_sequence.items()}

        def bancos_completados(fase) -> int:
            n = 0
            for pz in fase_bancos[fase]:
                if not pendientes.get((fase, pz)):
                    n += 1
                else:
                    break
            return n

        def fase_agotada(fase) -> bool:
            return all(not pendientes.get((fase, pz)) for pz in fase_bancos[fase])

        def puede_abrir(fase, relajado) -> bool:
            prev = fase_prev[fase]
            if prev is None:
                return True
            if prev not in fases_abiertas:
                return False
            if fase_agotada(prev):
                return True
            req = 1 if relajado else min(cfg.lag_fase, len(fase_bancos[prev]))
            return bancos_completados(prev) >= req

        # ─── BUCLE PRINCIPAL ───────────────────────────────────────────
        for t in range(1, cfg.horizontes + 1):
            self.progress_cb(t, cfg.horizontes, f"Período {t}/{cfg.horizontes}")

            lc_econ_t = lc_vector[t]
            discount_t = disc_factor[t]

            tons_ore_t   = 0.0
            tons_waste_t = 0.0
            mined_details = []  # [(u, ton_econ, metal_econ, ton_marg, metal_marg)]

            remaining_ore   = float(cap_ore)
            remaining_mina  = meta_mov
            inv_ini = sp_marginal.get_inventory()[0] + sp_economic.get_inventory()[0]
            ore_room = cap_pl + max(0.0, cap_stock - inv_ini)   # alimentación directa + espacio libre en stock

            def minar_fase(fase, forzar=False) -> int:
                """Mina paneles de la fase mientras quepan en las tasas. Devuelve cuántos minó."""
                nonlocal remaining_ore, remaining_mina, tons_ore_t, tons_waste_t
                n_min = 0
                for pz in fase_bancos[fase]:
                    lista = pendientes.get((fase, pz))
                    if not lista:
                        continue
                    quedan = []
                    for u in lista:
                        if mined_mask[u]:
                            continue
                        # precedencia vertical (banco superior de la misma fase)
                        if any(not mined_mask[p] for p in precedencias[u]):
                            quedan.append(u)
                            continue
                        ton_total = float(ton_arr[u])
                        if ton_total <= 0:
                            mined_mask[u] = True
                            mined_period[u] = t
                            continue
                        lb = float(ley_arr[u])
                        if lb >= lc_econ_t:
                            ton_econ, ton_marg, ton_waste = ton_total, 0.0, 0.0
                        elif lb >= lc_marginal:
                            ton_econ, ton_marg, ton_waste = 0.0, ton_total, 0.0
                        else:
                            ton_econ, ton_marg, ton_waste = 0.0, 0.0, ton_total
                        ton_ore_panel = ton_econ + ton_marg
                        cabe = (remaining_ore >= ton_ore_panel - 1.0) and (remaining_mina >= ton_total - 1.0) \
                               and (tons_ore_t + ton_ore_panel <= ore_room + 1.0)
                        if not cabe and not (forzar and n_min == 0 and not mined_details):
                            quedan.append(u)
                            continue
                        mt = float(paneles_df["metal_total"].iat[u])
                        mined_mask[u]   = True
                        mined_period[u] = t
                        tons_ore_t   += ton_ore_panel
                        tons_waste_t += ton_waste
                        remaining_ore  -= ton_ore_panel
                        remaining_mina -= ton_total
                        mined_details.append((u, ton_econ, mt * ton_econ / ton_total,
                                              ton_marg, mt * ton_marg / ton_total))
                        n_min += 1
                    pendientes[(fase, pz)] = quedan
                return n_min

            # fases que cumplen el lag se abren; luego se minan en orden de prioridad
            for f in phases:
                if f not in fases_abiertas and puede_abrir(f, relajado=False):
                    fases_abiertas.add(f)
            extra_usado = False
            for _ in range(2 * len(phases) + 2):
                for f in phases:
                    if f in fases_abiertas and not fase_agotada(f):
                        minar_fase(f)
                # falta mineral → permitir pre-stripping hasta el tope físico de movimiento
                if remaining_ore > 0.03 * cap_ore and not extra_usado and cap_mov > meta_mov:
                    remaining_mina += cap_mov - meta_mov
                    extra_usado = True
                    continue
                # ¿queda capacidad ociosa? → abrir la siguiente fase en vez de parar
                falta = (remaining_ore > 0.03 * cap_ore) or (remaining_mina > 0.03 * meta_mov)
                if not falta:
                    break
                sig = next((f for f in phases if f not in fases_abiertas and puede_abrir(f, relajado=True)), None)
                if sig is None:
                    break
                fases_abiertas.add(sig)
            if not mined_details and not mined_mask.all():
                # evita períodos vacíos: fuerza el primer panel elegible
                for f in phases:
                    if f in fases_abiertas and minar_fase(f, forzar=True):
                        break

            # ─── PLANTA / STOCKPILE (balance de masa) ──────────────────
            # alimentación = mineral directo desde mina + recuperado desde stockpile
            # inventario final = inventario inicial + enviado a stock − recuperado
            metal_econ_t  = sum(d[2] for d in mined_details)
            metal_marg_t  = sum(d[4] for d in mined_details)
            ton_econ_t    = sum(d[1] for d in mined_details)
            ton_marg_t    = sum(d[3] for d in mined_details)

            directo = min(ton_econ_t, cap_pl)                       # mineral económico directo a planta
            metal_directo = metal_econ_t * (directo / ton_econ_t) if ton_econ_t > 1e-9 else 0.0
            a_stock_e = ton_econ_t - directo
            sp_economic.ingresar(a_stock_e, metal_econ_t - metal_directo)
            sp_marginal.ingresar(ton_marg_t, metal_marg_t)          # marginal siempre a stock
            a_stock = a_stock_e + ton_marg_t

            feed_total = directo
            metal_feed = metal_directo
            deficit    = max(0.0, cap_pl - feed_total)
            lim        = cap_recup
            pull_e, m_pull_e = sp_economic.retirar(min(deficit, lim))
            lim -= pull_e
            pull_m, m_pull_m = sp_marginal.retirar(min(deficit - pull_e, lim))
            feed_total += pull_e + pull_m
            metal_feed += m_pull_e + m_pull_m
            recuperado = pull_e + pull_m

            # Economía período
            ley_head_t = (metal_feed / feed_total * 100.0) if feed_total > 1e-9 else 0.0
            ingresos   = valor_por_ton_proc * metal_feed
            c_mina     = self.eco.costo_mina * (tons_ore_t + tons_waste_t)
            c_planta   = self.eco.costo_planta * feed_total
            c_rehandle = self.sch.costo_remanejo * recuperado
            inv_m, _   = sp_marginal.get_inventory()
            inv_e, _   = sp_economic.get_inventory()
            c_hold     = self.sch.costo_holding * (inv_m + inv_e)

            van_t = (ingresos - c_mina - c_planta - c_rehandle - c_hold) * discount_t
            total_van += van_t

            # Metal contenido / recuperable (t para leyes en %, oz para g/t)
            rec = self.eco.recuperacion_dec
            if self.eco.unidad_ley == "%":
                m_cont = metal_feed                      # (ton × ley%/100) = toneladas de metal
            else:
                m_cont = metal_feed * 100.0 / 31.1035    # g/t → gramos → onzas troy
            ley_mina_t = (sum(d[2] + d[4] for d in mined_details) / tons_ore_t * 100.0) if tons_ore_t > 1e-9 else 0.0

            # Extracción por fase y período + causa de cada detención de fase abierta
            por_fase = defaultdict(lambda: [0.0, 0.0, 0.0])
            for d in mined_details:
                f_ = int(fase_arr[d[0]]); tt_ = float(ton_arr[d[0]])
                por_fase[f_][0] += d[1] + d[3]; por_fase[f_][1] += tt_ - d[1] - d[3]; por_fase[f_][2] += d[2] + d[4]
            for f_, (o_, w_, m_) in por_fase.items():
                fase_per_rows.append({"fase": f_, "periodo": t, "mineral_Mt": o_ / 1e6, "esteril_Mt": w_ / 1e6,
                                      "ley": (m_ / o_ * 100.0) if o_ > 1e-9 else 0.0})
            for f in phases:
                if f in fases_abiertas and not fase_agotada(f) and f not in por_fase:
                    acces = False
                    for pz in fase_bancos[f]:
                        for u in pendientes.get((f, pz), []):
                            if not any(not mined_mask[p] for p in precedencias[u]):
                                acces = True; break
                        if acces: break
                    motivo = ("capacidad de mina/planta/stock agotada en el período" if acces
                              else "sin acceso: espera extracción del banco superior")
                    interrupciones.append({"fase": int(f), "periodo": t, "motivo": motivo})

            res_rows.append({
                "periodo":              t,
                "mineral_mined_Mt":     tons_ore_t / 1e6,
                "esteril_Mt":           tons_waste_t / 1e6,
                "mov_total_Mt":         (tons_ore_t + tons_waste_t) / 1e6,
                "mina_ley_mined_pct":   ley_mina_t,
                "planta_directo_Mt":    directo / 1e6,
                "a_stock_Mt":           a_stock / 1e6,
                "desde_stock_Mt":       recuperado / 1e6,
                "planta_feed_Mt":       feed_total / 1e6,
                "planta_ley_head_pct":  ley_head_t,
                "metal_contenido":      m_cont,
                "metal_recuperable":    m_cont * rec,
                "sp_inventario_Mt":     (inv_m + inv_e) / 1e6,
                "VAN_net_MUSD":         van_t / 1e6,
                "cutoff_econ_pct":      lc_econ_t,
            })

            if mined_mask.all():
                inv_m_fin, _ = sp_marginal.get_inventory()
                inv_e_fin, _ = sp_economic.get_inventory()
                if inv_m_fin + inv_e_fin < 1.0:
                    logger.info(f"Modelo completamente minado y procesado en período {t}.")
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

        cols_out = ["X", "Y", "Z", "tonelaje", "Ley", "fase", "periodo"]
        if "pit" in df.columns:          # pit anidado de cada bloque (para ver los pits en el visor)
            cols_out.append("pit")
        bloques_out = df[cols_out].copy()

        logger.info(f"✅ Scheduler completado en {time.time()-t0:.1f}s | VAN={total_van/1e6:,.2f} MUSD")
        return ResultadoScheduler(plan_df, bloques_out, total_van, self.config,
                                  pd.DataFrame(fase_per_rows), interrupciones)

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
        """
        Precedencia de cada panel (dentro de su fase):
          1. vertical: el panel existente más cercano por encima en la misma columna (aunque falten bancos
             intermedios, p. ej. modelos decimados), lo que encadena toda la columna;
          2. espacial: los paneles vecinos (3×3) del banco inmediato superior, para respetar el talud.
        """
        prec = defaultdict(list)
        px_s, py_s = float(self.sch.panel_size_x), float(self.sch.panel_size_y)
        # banco inmediato superior existente en la fase, por columna
        for (fase, px, py), g in paneles_df.groupby(["fase", "panel_X", "panel_Y"]):
            g = g.sort_values("panel_Z", ascending=False)
            idx = list(g.index)
            for k in range(1, len(idx)):
                prec[idx[k]].append(idx[k - 1])
        if getattr(self.sch, "precedencia_espacial", True):
            for u, row in paneles_df.iterrows():
                fase, px, py, pz = row["fase"], row["panel_X"], row["panel_Y"], row["panel_Z"]
                b_idx = bench_idx_map.get(pz)
                if b_idx is None or b_idx == 0:
                    continue
                pz_arriba = unique_benches[b_idx - 1]
                for dx in (-px_s, 0.0, px_s):
                    for dy in (-py_s, 0.0, py_s):
                        if dx == 0.0 and dy == 0.0:
                            continue
                        v = panel_key_to_u.get((fase, px + dx, py + dy, pz_arriba))
                        if v is not None and v not in prec[u]:
                            prec[u].append(v)
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
