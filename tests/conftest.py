import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.config import ProjectConfig
from engine.scheduler import HeuristicaFaseBanco


def modelo_sintetico(n=14, nz=8, seed=7):
    """Modelo determinista: cono invertido, 3 fases anidadas, ley gaussiana centrada."""
    rng = np.random.default_rng(seed)
    xs, ys, zs = np.arange(n) * 20.0, np.arange(n) * 20.0, 150.0 - np.arange(nz) * 15.0
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    df = pd.DataFrame({"X": X.ravel(), "Y": Y.ravel(), "Z": Z.ravel()})
    c = (n - 1) * 10.0
    hd = np.hypot(df.X - c, df.Y - c)
    # sólo bloques dentro de un cono (talud 45°): radio decrece con la profundidad
    radio = 150.0 - (150.0 - df.Z) * 0.6
    df = df[hd <= radio].copy()
    hd = hd[df.index]
    df["Ley"] = np.clip(1.2 * np.exp(-(hd / 80.0) ** 2) * (1 + 0.2 * rng.standard_normal(len(df))), 0.01, None)
    df["tonelaje"] = 20 * 20 * 15 * 2.5
    # fases por distancia al eje (la 1 es la interior)
    q = pd.qcut(hd.rank(method="first"), 3, labels=[1, 2, 3]).astype(int)
    df["fase"] = q.values
    return df.reset_index(drop=True)


def correr(df=None, **sch):
    df = modelo_sintetico() if df is None else df
    p = {"metal": "cu",
         "economico": {"precio_metal": 4, "cargo_tc_rc": 0.1, "recuperacion": 90, "costo_mina": 2, "costo_planta": 10},
         "scheduler": {"cap_mineral_t": 1.2e6, "cap_movimiento_t": 3.0e6, "cap_planta_t": 1.0e6}}
    p["scheduler"].update(sch)
    cfg = ProjectConfig.desde_dict(p)
    cfg.scheduler.horizontes = 60
    res = HeuristicaFaseBanco(SimpleNamespace(config=cfg), df).ejecutar()
    return cfg, df, res


@pytest.fixture(scope="session")
def base():
    return correr()
