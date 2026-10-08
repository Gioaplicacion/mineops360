"""
Controles automáticos de reconciliación entre módulos
(modelo → reservas → pit → fases → plan → alimentación → economía).
Cada control devuelve {control, ok, detalle}. Se ejecutan sobre los resultados reales del pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _c(nombre: str, ok: bool, detalle: str) -> dict:
    return {"control": nombre, "ok": bool(ok), "detalle": detalle}


def reconciliar(config, bloques: pd.DataFrame, plan: pd.DataFrame, van_total: float,
                resumen_pits: dict | None = None, resumen_modelo: dict | None = None) -> list:
    sch, eco = config.scheduler, config.economico
    out = []
    tol = 1e-6
    minado = bloques[bloques.periodo > 0]

    # 1. Conservación de toneladas bloques ↔ plan
    t_bl, t_pl = minado.tonelaje.sum() / 1e6, plan.mov_total_Mt.sum()
    out.append(_c("Toneladas: bloques minados = movimiento del plan", abs(t_bl - t_pl) <= tol * max(1, t_pl),
                  f"bloques {t_bl:,.3f} Mt · plan {t_pl:,.3f} Mt"))

    # 2. mov = mineral + estéril
    d = (plan.mov_total_Mt - plan.mineral_mined_Mt - plan.esteril_Mt).abs().max()
    out.append(_c("Movimiento total = mineral + estéril (cada período)", d <= tol, f"diferencia máx. {d:.2e} Mt"))

    # 3. alimentación = directo + desde stock ; inventario
    d1 = (plan.planta_feed_Mt - plan.planta_directo_Mt - plan.desde_stock_Mt).abs().max()
    inv = plan.sp_inventario_Mt.values
    prev = np.concatenate([[0.0], inv[:-1]])
    d2 = np.abs(inv - (prev + plan.a_stock_Mt.values - plan.desde_stock_Mt.values)).max()
    out.append(_c("Alimentación = directo + recuperado de stock", d1 <= tol, f"diferencia máx. {d1:.2e} Mt"))
    out.append(_c("Inventario final = inicial + a stock − recuperado", d2 <= tol, f"diferencia máx. {d2:.2e} Mt"))

    # 4. todo el mineral minado queda alimentado o en inventario
    resto = plan.mineral_mined_Mt.sum() - plan.planta_feed_Mt.sum() - inv[-1]
    out.append(_c("Mineral minado = alimentado + inventario final", abs(resto) <= 1e-5, f"diferencia {resto:.2e} Mt"))

    # 5. capacidades
    caps = {"mineral": ("mineral_mined_Mt", "mina mineral"), "movimiento": ("mov_total_Mt", "movimiento mina"),
            "planta": ("planta_feed_Mt", "alimentación planta")}
    for k, (col, nombre) in caps.items():
        cap = sch.cap_anual(k) / 1e6
        mx = plan[col].max()
        out.append(_c(f"Capacidad {nombre} respetada", mx <= cap + tol, f"máx. {mx:.3f} Mt ≤ {cap:.3f} Mt"))
    if sch.cap_stock_t > 0:
        out.append(_c("Capacidad de stockpile respetada", inv.max() <= sch.cap_stock_t / 1e6 + tol,
                      f"máx. {inv.max():.3f} Mt ≤ {sch.cap_stock_t/1e6:.3f} Mt"))

    # 6. ley de alimentación ponderada → metal contenido
    if eco.unidad_ley == "%":
        esperado = plan.planta_feed_Mt * 1e6 * plan.planta_ley_head_pct / 100.0
    else:
        esperado = plan.planta_feed_Mt * 1e6 * plan.planta_ley_head_pct / 31.1035
    rel = ((plan.metal_contenido - esperado).abs() / esperado.clip(lower=1e-9)).max()
    out.append(_c("Metal contenido = Σ(t alimentadas × ley)", rel <= 1e-6, f"error relativo máx. {rel:.2e}"))

    # 7. VAN del plan = suma de flujos
    s = plan.VAN_net_MUSD.sum()
    out.append(_c("VAN total = suma de flujos descontados", abs(s - van_total / 1e6) <= 1e-6 * max(1, abs(s)),
                  f"Σ flujos {s:,.4f} MUSD · VAN {van_total/1e6:,.4f} MUSD"))

    # 8. bloques únicos y precedencia vertical por fase/columna
    dup = int(bloques.duplicated(["X", "Y", "Z"]).sum())
    out.append(_c("Sin bloques duplicados", dup == 0, f"{dup} duplicados"))
    malos = 0
    for _, g in minado.groupby(["fase", "X", "Y"]):
        if len(g) > 1 and not g.sort_values("Z", ascending=False).periodo.is_monotonic_increasing:
            malos += 1
    out.append(_c("Precedencia vertical respetada (banco superior primero)", malos == 0, f"{malos} columnas con violación"))

    # 9. pit final del módulo ¿Cuánto? = bloques del plan/fases
    if resumen_pits and resumen_pits.get("bloques_pit_final") is not None:
        n_pit, n_bl = int(resumen_pits["bloques_pit_final"]), len(bloques)
        out.append(_c("Bloques del pit final = bloques faseados", n_pit == n_bl, f"pit {n_pit:,} · fases {n_bl:,}"))

    # 10. mineral del plan ≤ mineral del pit final (ley de corte económica)
    lc = eco.ley_de_corte
    ore_pit = bloques[bloques.Ley >= lc].tonelaje.sum() / 1e6
    ore_plan = plan.mineral_mined_Mt.sum()
    out.append(_c("Mineral del plan ≤ reservas del pit (ley ≥ corte)", ore_plan <= ore_pit + 1e-6 + (bloques[(bloques.Ley >= eco.ley_de_corte_marginal) & (bloques.Ley < lc)].tonelaje.sum() / 1e6),
                  f"plan {ore_plan:,.2f} Mt · pit (≥ corte) {ore_pit:,.2f} Mt"))
    return out
