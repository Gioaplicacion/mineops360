from conftest import base  # noqa: F401
from engine.reconciliacion import reconciliar


def test_reconciliacion_sin_fallas(base):
    cfg, df, r = base
    res = reconciliar(cfg, r.bloques, r.plan, r.van_total, {"bloques_pit_final": len(r.bloques)})
    malos = [c for c in res if not c["ok"]]
    assert len(res) >= 14 and not malos, malos


def test_reconciliacion_detecta_inconsistencias(base):
    cfg, df, r = base
    plan = r.plan.copy()
    plan.loc[0, "planta_feed_Mt"] += 0.5                        # alimentación inconsistente
    res = reconciliar(cfg, r.bloques, plan, r.van_total)
    assert any(not c["ok"] for c in res)
    b = r.bloques.copy()
    b.loc[b.periodo > 0, "tonelaje"] *= 2                       # toneladas que no cuadran con el plan
    assert any(not c["ok"] for c in reconciliar(cfg, b, r.plan, r.van_total))
