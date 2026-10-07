"""
MineOps 360 — Exportación DXF de resultados
===========================================
Genera un DXF (ASCII, formato R12, abre en AutoCAD, Vulcan, Datamine, Surpac, Leapfrog…)
con mallas 3DFACE agrupadas por capa:

  FASE_n               sólido de cada fase (¿Cómo?)
  PERIODO_nn           sólido de cada período del plan (¿Cuándo?)
  SUPERFICIE_PIT_FINAL superficie del pit final (¿Cuánto?): piso y paredes del pit

Los sólidos se construyen uniendo los bloques de cada grupo y emitiendo sólo las caras
exteriores (se fusionan caras contiguas para mantener el archivo liviano).
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy import ndimage

# Colores ACI (AutoCAD) para fases y períodos
_ACI_FASES = [1, 3, 5, 2, 6, 4, 30, 40, 150, 190]
_MAX_CELDAS = 40_000_000


def _paso(valores: np.ndarray, defecto: float) -> float:
    u = np.unique(np.round(valores.astype(float), 3))
    if len(u) < 2:
        return float(defecto)
    d = np.diff(u)
    d = d[d > 1e-6]
    if len(d) == 0:
        return float(defecto)
    # paso más frecuente (robusto a modelos con huecos)
    vals, cnt = np.unique(np.round(d, 3), return_counts=True)
    return float(vals[np.argmax(cnt)])


class _Grilla:
    def __init__(self, df: pd.DataFrame, dx: float, dy: float, dz: float):
        self.sx = _paso(df["X"].values, dx)
        self.sy = _paso(df["Y"].values, dy)
        self.sz = _paso(df["Z"].values, dz)
        self.x0 = float(df["X"].min())
        self.y0 = float(df["Y"].min())
        self.z0 = float(df["Z"].min())
        self.ix = np.rint((df["X"].values - self.x0) / self.sx).astype(int)
        self.iy = np.rint((df["Y"].values - self.y0) / self.sy).astype(int)
        self.iz = np.rint((df["Z"].values - self.z0) / self.sz).astype(int)
        self.shape = (self.ix.max() + 3, self.iy.max() + 3, self.iz.max() + 3)  # borde de 1 celda
        if np.prod(self.shape, dtype=np.int64) > _MAX_CELDAS:
            raise ValueError("Modelo demasiado grande para exportar a DXF")

    def mascara(self, sel: np.ndarray, cerrar: bool = True) -> np.ndarray:
        m = np.zeros(self.shape, dtype=bool)
        m[self.ix[sel] + 1, self.iy[sel] + 1, self.iz[sel] + 1] = True
        if cerrar and m.any():
            # rellena huecos de modelos decimados / con franjas (cierre morfológico suave)
            m = ndimage.binary_closing(m, structure=np.ones((3, 3, 3), bool), iterations=1) | m
            m = ndimage.binary_fill_holes(m) | m
        return m


def _caras(m: np.ndarray, direccion: str) -> np.ndarray:
    """Máscara de celdas con cara exterior en la dirección dada."""
    f = m.copy()
    if direccion == "+x": f[:-1] &= ~m[1:];  f[-1] = m[-1]
    elif direccion == "-x": f[1:] &= ~m[:-1]; f[0] = m[0]
    elif direccion == "+y": f[:, :-1] &= ~m[:, 1:]; f[:, -1] = m[:, -1]
    elif direccion == "-y": f[:, 1:] &= ~m[:, :-1]; f[:, 0] = m[:, 0]
    elif direccion == "+z": f[:, :, :-1] &= ~m[:, :, 1:]; f[:, :, -1] = m[:, :, -1]
    elif direccion == "-z": f[:, :, 1:] &= ~m[:, :, :-1]; f[:, :, 0] = m[:, :, 0]
    return f


def _runs(f: np.ndarray, eje: int):
    """Corridas contiguas de True a lo largo de `eje`. Devuelve (idx_otros..., inicio, fin_inclusive)."""
    g = np.moveaxis(f, eje, -1)
    pad = np.zeros(g.shape[:-1] + (g.shape[-1] + 2,), dtype=np.int8)
    pad[..., 1:-1] = g
    d = np.diff(pad, axis=-1)
    ini = np.argwhere(d == 1)      # (a, b, k_inicio)
    fin = np.argwhere(d == -1)     # (a, b, k_fin_exclusivo)
    return ini, fin[:, -1] - 1


def _quads(m: np.ndarray, G: _Grilla, direcciones: Iterable[str]) -> list:
    """Lista de cuadriláteros (4 vértices) de las caras exteriores, fusionando corridas."""
    out = []
    hx, hy, hz = G.sx / 2, G.sy / 2, G.sz / 2
    for d in direcciones:
        f = _caras(m, d)
        if not f.any():
            continue
        eje_run = 2 if d[1] in "xy" else 0          # ±x,±y corren en Z ; ±z corre en X
        ini, kfin = _runs(f, eje_run)
        if len(ini) == 0:
            continue
        a, b, k0 = ini[:, 0], ini[:, 1], ini[:, 2]
        k1 = kfin
        if eje_run == 2:                            # a=i, b=j, run en k
            cx = G.x0 + (a - 1) * G.sx
            cy = G.y0 + (b - 1) * G.sy
            za = G.z0 + (k0 - 1) * G.sz - hz
            zb = G.z0 + (k1 - 1) * G.sz + hz
            if d[1] == "x":
                x = cx + (hx if d[0] == "+" else -hx)
                v = [(x, cy - hy, za), (x, cy + hy, za), (x, cy + hy, zb), (x, cy - hy, zb)]
            else:
                y = cy + (hy if d[0] == "+" else -hy)
                v = [(cx - hx, y, za), (cx + hx, y, za), (cx + hx, y, zb), (cx - hx, y, zb)]
        else:                                       # ±z : g tiene ejes (j, k, i) tras moveaxis → a=j, b=k, run en i
            cy = G.y0 + (a - 1) * G.sy
            cz = G.z0 + (b - 1) * G.sz
            xa = G.x0 + (k0 - 1) * G.sx - hx
            xb = G.x0 + (k1 - 1) * G.sx + hx
            z = cz + (hz if d[0] == "+" else -hz)
            v = [(xa, cy - hy, z), (xb, cy - hy, z), (xb, cy + hy, z), (xa, cy + hy, z)]
        out.append(np.stack([np.stack(p, axis=1) for p in v], axis=1))   # (n, 4, 3)
    return out


def _escribir_caras(buf: list, capa: str, bloques: list) -> int:
    n = 0
    for arr in bloques:
        for q in arr:
            buf.append("0\n3DFACE\n8\n%s\n" % capa)
            for i, (x, y, z) in enumerate(q):
                buf.append("%d\n%.2f\n%d\n%.2f\n%d\n%.2f\n" % (10 + i, x, 20 + i, y, 30 + i, z))
            n += 1
    return n


def generar_dxf(
    bloques: pd.DataFrame,
    dx: float = 20.0,
    dy: float = 20.0,
    dz: float = 15.0,
    ruta: str = "salida.dxf",
    incluir_periodos: bool = True,
) -> dict:
    """
    bloques: columnas X, Y, Z (+ fase, periodo opcionales). Contiene el pit final.
    Devuelve un resumen {capa: n_caras}.
    """
    df = bloques.reset_index(drop=True)
    G = _Grilla(df, dx, dy, dz)
    capas: list[tuple[str, int]] = []
    cuerpo: list[str] = []
    resumen: dict = {}

    # ── Superficie del pit final (¿Cuánto?) : piso y paredes del volumen extraído ──
    m_pit = G.mascara(np.ones(len(df), bool))
    q = _quads(m_pit, G, ["+x", "-x", "+y", "-y", "-z"])
    capas.append(("SUPERFICIE_PIT_FINAL", 7))
    resumen["SUPERFICIE_PIT_FINAL"] = _escribir_caras(cuerpo, "SUPERFICIE_PIT_FINAL", q)

    # ── Sólidos de fases (¿Cómo?) ──
    if "fase" in df.columns:
        fases = sorted(int(f) for f in df["fase"].dropna().unique() if int(f) > 0)
        for i, f in enumerate(fases):
            capa = f"FASE_{f}"
            m = G.mascara((df["fase"] == f).values)
            q = _quads(m, G, ["+x", "-x", "+y", "-y", "+z", "-z"])
            capas.append((capa, _ACI_FASES[i % len(_ACI_FASES)]))
            resumen[capa] = _escribir_caras(cuerpo, capa, q)

    # ── Sólidos del plan por período (¿Cuándo?) ──
    if incluir_periodos and "periodo" in df.columns:
        pers = sorted(int(p) for p in df["periodo"].dropna().unique() if int(p) > 0)
        for p in pers:
            capa = f"PERIODO_{p:02d}"
            m = G.mascara((df["periodo"] == p).values)
            q = _quads(m, G, ["+x", "-x", "+y", "-y", "+z", "-z"])
            capas.append((capa, 11 + (p * 7) % 240))
            resumen[capa] = _escribir_caras(cuerpo, capa, q)

    # ── Ensamblar DXF R12 ──
    cab = ["0\nSECTION\n2\nHEADER\n9\n$ACADVER\n1\nAC1009\n0\nENDSEC\n",
           "0\nSECTION\n2\nTABLES\n0\nTABLE\n2\nLAYER\n70\n%d\n" % (len(capas) + 1),
           "0\nLAYER\n2\n0\n70\n0\n62\n7\n6\nCONTINUOUS\n"]
    for nombre, color in capas:
        cab.append("0\nLAYER\n2\n%s\n70\n0\n62\n%d\n6\nCONTINUOUS\n" % (nombre, color))
    cab.append("0\nENDTAB\n0\nENDSEC\n0\nSECTION\n2\nENTITIES\n")
    with open(ruta, "w", encoding="ascii", newline="\n") as fh:
        fh.write("".join(cab))
        fh.write("".join(cuerpo))
        fh.write("0\nENDSEC\n0\nEOF\n")
    return resumen
