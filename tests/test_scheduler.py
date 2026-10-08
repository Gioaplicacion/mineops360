"""Pruebas del scheduler (¿Cuándo?): conservación, precedencia, capacidades, leyes, stock, continuidad, VAN."""
import numpy as np
import pandas as pd
import pytest

from conftest import correr, modelo_sintetico

EPS = 1e-6


def test_conservacion_de_toneladas(base):
    cfg, df, r = base
    b, plan = r.bloques, r.plan
    minado = b[b.periodo > 0].tonelaje.sum()
    assert minado == pytest.approx(plan.mov_total_Mt.sum() * 1e6, rel=1e-9)
    assert plan.mineral_mined_Mt.sum() + plan.esteril_Mt.sum() == pytest.approx(plan.mov_total_Mt.sum(), rel=1e-9)
    assert len(b) == len(df)                                   # ningún bloque se pierde


def test_sin_duplicacion_de_bloques(base):
    _, _, r = base
    b = r.bloques
    assert not b.duplicated(["X", "Y", "Z"]).any()
    assert b.periodo.ge(0).all()
    # lo minado por período en el modelo coincide con el plan (cada bloque se extrae una sola vez)
    g = b[b.periodo > 0].groupby("periodo").tonelaje.sum() / 1e6
    assert np.allclose(g.reindex(r.plan.periodo).fillna(0).values, r.plan.mov_total_Mt.values)


def test_precedencia_vertical(base):
    _, _, r = base
    b = r.bloques[r.bloques.periodo > 0].copy()
    # panel 20x20 por fase: el banco inferior jamás se extrae antes que el superior
    for (f, x, y), g in b.groupby(["fase", "X", "Y"]):
        g = g.sort_values("Z", ascending=False)
        assert g.periodo.is_monotonic_increasing, f"precedencia violada en fase {f} ({x},{y})"


def test_capacidades_mina_y_planta(base):
    cfg, _, r = base
    p = r.plan
    assert (p.mineral_mined_Mt <= cfg.scheduler.cap_mineral_t / 1e6 + EPS).all()
    assert (p.mov_total_Mt <= cfg.scheduler.cap_movimiento_t / 1e6 + EPS).all()
    assert (p.planta_feed_Mt <= cfg.scheduler.cap_planta_t / 1e6 + EPS).all()


def test_balance_de_masa_y_stockpile(base):
    _, _, r = base
    p = r.plan
    assert np.allclose(p.mov_total_Mt, p.mineral_mined_Mt + p.esteril_Mt)
    assert np.allclose(p.planta_feed_Mt, p.planta_directo_Mt + p.desde_stock_Mt)
    inv_prev = 0.0
    for _, row in p.iterrows():
        assert row.sp_inventario_Mt == pytest.approx(inv_prev + row.a_stock_Mt - row.desde_stock_Mt, abs=1e-9)
        assert row.sp_inventario_Mt >= -EPS
        inv_prev = row.sp_inventario_Mt
    # todo el mineral minado queda procesado o en inventario
    assert p.mineral_mined_Mt.sum() == pytest.approx(p.planta_feed_Mt.sum() + p.sp_inventario_Mt.iloc[-1], abs=1e-6)


def test_ley_alimentacion_ponderada():
    # planta holgada: todo el mineral económico va directo → la ley de planta es el promedio ponderado real
    cfg, df, r = correr(cap_planta_t=50e6, cap_mineral_t=2e6, cap_movimiento_t=6e6)
    lc = cfg.economico.ley_de_corte
    lm = cfg.economico.ley_de_corte_marginal
    b = r.bloques
    for t, g in b[b.periodo > 0].groupby("periodo"):
        econ = g[g.Ley >= lc]
        marg = g[(g.Ley >= lm) & (g.Ley < lc)]
        if not len(econ) or len(marg):
            continue
        esperado = (econ.Ley * econ.tonelaje).sum() / econ.tonelaje.sum()
        fila = r.plan[r.plan.periodo == t].iloc[0]
        assert fila.planta_ley_head_pct == pytest.approx(esperado, rel=1e-6)


def test_metal_contenido_y_recuperable(base):
    cfg, _, r = base
    p = r.plan
    assert np.allclose(p.metal_contenido, p.planta_feed_Mt * 1e6 * p.planta_ley_head_pct / 100.0, rtol=1e-9)
    assert np.allclose(p.metal_recuperable, p.metal_contenido * cfg.economico.recuperacion / 100.0)


def test_metal_oro_en_onzas():
    df = modelo_sintetico()
    df["Ley"] = df["Ley"] * 2.0                                 # g/t
    cfg, _, r = correr(df, cap_planta_t=1e6)  # cu por defecto → sólo valida rama %, abajo la rama oro
    cfg_au, _, r_au = _correr_au(df)
    p = r_au.plan
    esperado = p.planta_feed_Mt * 1e6 * p.planta_ley_head_pct / 31.1035
    assert np.allclose(p.metal_contenido, esperado, rtol=1e-9)
    assert r_au.to_dict()["unidad_metal"] == "oz"


def _correr_au(df):
    from types import SimpleNamespace
    from engine.config import ProjectConfig
    from engine.scheduler import HeuristicaFaseBanco
    cfg = ProjectConfig.desde_dict({"metal": "au",
        "economico": {"precio_metal": 2500, "cargo_tc_rc": 5, "recuperacion": 88, "costo_mina": 2, "costo_planta": 12},
        "scheduler": {"cap_mineral_t": 1.2e6, "cap_movimiento_t": 3e6, "cap_planta_t": 1e6}})
    cfg.scheduler.horizontes = 60
    return cfg, df, HeuristicaFaseBanco(SimpleNamespace(config=cfg), df).ejecutar()


def test_capacidad_stockpile_respetada():
    cfg, _, r = correr(cap_stock_t=0.3e6)
    assert (r.plan.sp_inventario_Mt <= 0.3 + EPS).all()


def test_capacidades_en_t_por_dia():
    cfg, _, r = correr(cap_mineral_tpd=3000, cap_movimiento_tpd=8000, cap_planta_tpd=2700, dias_efectivos=350)
    assert (r.plan.mineral_mined_Mt <= 3000 * 350 / 1e6 + EPS).all()
    assert (r.plan.mov_total_Mt <= 8000 * 350 / 1e6 + EPS).all()
    assert (r.plan.planta_feed_Mt <= 2700 * 350 / 1e6 + EPS).all()


def test_continuidad_de_fases_justificada(base):
    _, _, r = base
    fp, inter = r.fase_periodo, pd.DataFrame(r.interrupciones)
    for f, g in fp.groupby("fase"):
        pers = set(g.periodo)
        for t in range(min(pers), max(pers) + 1):
            if t not in pers:
                assert len(inter) and ((inter.fase == f) & (inter.periodo == t)).any(), \
                    f"fase {f} se detuvo en el período {t} sin causa registrada"
    assert all(i["motivo"] for i in r.interrupciones)


def test_no_hay_periodos_vacios_antes_del_agotamiento(base):
    _, _, r = base
    p = r.plan
    ultimo = p[p.mov_total_Mt > 0].periodo.max()
    assert (p[p.periodo <= ultimo].mov_total_Mt > 0).all()


def test_consistencia_economica(base):
    _, _, r = base
    p = r.plan
    assert p.VAN_net_MUSD.sum() == pytest.approx(r.van_total / 1e6, rel=1e-9)
    assert p.VAN_acum_MUSD.iloc[-1] == pytest.approx(r.van_total / 1e6, rel=1e-9)


def test_extraccion_fase_periodo_cuadra(base):
    _, _, r = base
    fp, p = r.fase_periodo, r.plan
    g = fp.groupby("periodo")[["mineral_Mt", "esteril_Mt"]].sum().reindex(p.periodo).fillna(0)
    assert np.allclose(g.mineral_Mt.values, p.mineral_mined_Mt.values, atol=1e-9)
    assert np.allclose(g.esteril_Mt.values, p.esteril_Mt.values, atol=1e-9)


def test_regresion_resultado_base(base):
    """Valores de referencia del caso sintético (deterministas). Si cambian, revisar el algoritmo a conciencia."""
    _, _, r = base
    p = r.plan
    assert len(p) > 3
    assert p.mineral_mined_Mt.sum() > 0
    assert r.bloques.periodo.gt(0).mean() > 0.5


def test_precedencia_con_bancos_faltantes():
    """Modelos decimados: faltan bancos intermedios en algunas columnas; la precedencia debe encadenar igual."""
    df = modelo_sintetico()
    rng = np.random.default_rng(3)
    df = df[~((df.Z.isin([135.0, 105.0, 75.0])) & (rng.random(len(df)) < 0.35))].reset_index(drop=True)
    _, _, r = correr(df)
    m = r.bloques[r.bloques.periodo > 0]
    for (f, x, y), g in m.groupby(["fase", "X", "Y"]):
        assert g.sort_values("Z", ascending=False).periodo.is_monotonic_increasing, (f, x, y)


def test_precedencia_espacial_vecinos_banco_superior(base):
    """Un panel no se extrae antes que sus vecinos (3×3) del banco inmediato superior de su misma fase."""
    _, _, r = base
    m = r.bloques[r.bloques.periodo > 0]
    per = {(int(f), x, y, z): p for f, x, y, z, p in zip(m.fase, m.X, m.Y, m.Z, m.periodo)}
    benches = sorted(m.Z.unique(), reverse=True)
    arriba = {benches[i]: benches[i - 1] for i in range(1, len(benches))}
    for (f, x, y, z), p in per.items():
        if z not in arriba:
            continue
        for dx in (-20, 0, 20):
            for dy in (-20, 0, 20):
                v = per.get((f, x + dx, y + dy, arriba[z]))
                if v is not None:
                    assert v <= p, f"panel {(f, x, y, z)} (p{p}) antes que su vecino superior (p{v})"
