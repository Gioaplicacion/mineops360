import numpy as np
import pytest

from conftest import base  # noqa: F401  (fixture)
from engine.dxf_export import generar_dxf

ezdxf = pytest.importorskip("ezdxf")


def test_dxf_capas_y_geometria(base, tmp_path):
    _, df, r = base
    ruta = tmp_path / "t.dxf"
    res = generar_dxf(r.bloques, 20, 20, 15, str(ruta))
    assert res["SUPERFICIE_PIT_FINAL"] > 0
    fases = sorted(int(f) for f in r.bloques.fase.unique())
    assert all(res.get(f"FASE_{f}", 0) > 0 for f in fases)
    pers = sorted(int(p) for p in r.bloques.periodo.unique() if p > 0)
    assert all(res.get(f"PERIODO_{p:02d}", 0) > 0 for p in pers)
    d = ezdxf.readfile(str(ruta))
    msp = d.modelspace()
    assert len(msp) == sum(res.values())
    # todas las caras dentro de la caja del modelo (± medio bloque)
    pts = np.array([v for e in msp for v in (e.dxf.vtx0, e.dxf.vtx1, e.dxf.vtx2)])
    assert pts[:, 0].min() >= df.X.min() - 10 - 1e-6 and pts[:, 0].max() <= df.X.max() + 10 + 1e-6
    assert pts[:, 2].min() >= df.Z.min() - 7.5 - 1e-6 and pts[:, 2].max() <= df.Z.max() + 7.5 + 1e-6
