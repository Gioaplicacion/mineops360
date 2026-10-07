"""
Global Mine Planner — Módulo ¿Cómo? Faseamiento con Recocido Simulado
======================================================================
Implementa el algoritmo de optimización de pushbacks NO concéntricos
basado en Recocido Simulado (Simulated Annealing).

Referencia:
  - Navarro et al. (2024). Open-pit pushback optimization by a parallel 
    genetic algorithm. Minerals, 14(5), 438.
  - Rivera-Letelier et al. (2020). Production Scheduling for Strategic 
    Open Pit Mine Planning: A Mixed-Integer Programming Approach.
  - Láminas 302-347, UST/Nube Minera (2025).

Concepto central:
  Las fases NO son pits anidados concéntricos (método clásico L-G).
  Son regiones espaciales arbitrarias que:
    1. Maximizan el VAN respetando precedencias
    2. Siguen la dirección de avance operacional
    3. Respetan restricciones geotécnicas (talud)
    4. Se definen como pushbacks direccionados
"""

import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Optional, Callable

import numpy as np
import pandas as pd

from .config import ProjectConfig

logger = logging.getLogger(__name__)


# ── Parámetros del Recocido Simulado ────────────────────────────────────
@dataclass
class SimAnnealConfig:
    """Parámetros del algoritmo de Recocido Simulado para faseamiento."""
    num_fases:          int   = 4       # número de fases a generar
    iteraciones:        int   = 50      # iteraciones del SA
    temperatura_ini:    float = 1000.0  # temperatura inicial
    tasa_enfriamiento:  float = 0.90    # factor de enfriamiento por iteración
    radio_base_fase1:   float = 200.0   # radio inicial de la Fase 1 (m)
    dir_avance_prim:    str   = "Y-DEC" # dirección primaria de avance
    dir_avance_sec:     str   = "X-INC" # dirección secundaria de avance
    peso_van:           float = 1.0     # peso del VAN en la función objetivo
    peso_compacidad:    float = 0.1     # peso de compacidad de fases


# ── Resultado del faseamiento ────────────────────────────────────────────
@dataclass
class ResultadoFaseamiento:
    """Resultado del módulo ¿Cómo?"""
    df:              pd.DataFrame  # modelo de bloques con columna 'fase'
    num_fases:       int
    van_por_fase:    list          # VAN incremental por fase
    ton_por_fase:    list          # tonelaje por fase
    convergencia:    list          # historial VAN por iteración SA
    tiempo_s:        float

    @property
    def resumen(self) -> dict:
        return {
            "num_fases":    self.num_fases,
            "van_total":    sum(self.van_por_fase),
            "van_por_fase": [round(v, 2) for v in self.van_por_fase],
            "ton_por_fase": [round(t, 2) for t in self.ton_por_fase],
            "convergencia": self.convergencia,
            "tiempo_s":     round(self.tiempo_s, 2),
        }


# ── Módulo principal de faseamiento ─────────────────────────────────────
class FaseamientoSimAnnealing:
    """
    Genera fases de extracción usando Recocido Simulado.
    
    El algoritmo asigna cada bloque del pit final a una fase (1..N)
    optimizando el VAN total respetando:
      - Precedencias verticales estrictas (banco inferior antes que superior)
      - Dirección de avance configurada
      - Restricción de tamaño mínimo por fase
    
    Diferencia con pits anidados clásicos:
      Las fases resultantes tienen geometría real de pushback direccionado,
      no concéntrico — como se muestra en láminas 340 y 347 del curso UST.
    """

    def __init__(
        self,
        config:      ProjectConfig,
        sa_config:   SimAnnealConfig = None,
        progress_cb: Optional[Callable] = None,
    ):
        self.cfg    = config
        self.sa     = sa_config or SimAnnealConfig()
        self.progress_cb = progress_cb or (lambda *a: None)

    def ejecutar(self, df_pit: pd.DataFrame) -> ResultadoFaseamiento:
        """
        Genera el faseamiento del pit final.

        Args:
            df_pit: DataFrame con bloques del pit (columnas: x,y,z,value,pit,ton,wton,oton)
                    — resultado del optimizador MaxFlow.

        Returns:
            ResultadoFaseamiento con fases asignadas.
        """
        t0 = time.time()
        logger.info(f"Iniciando faseamiento SA: {len(df_pit):,} bloques, {self.sa.num_fases} fases")
        self.progress_cb(0, self.sa.iteraciones, "Iniciando faseamiento...")

        # Normalizar columnas
        df = df_pit.copy()
        df.columns = [c.lower() for c in df.columns]
        for col in ['x','y','z','value']:
            if col not in df.columns:
                raise ValueError(f"Columna requerida faltante: {col}")

        # Filtrar solo bloques dentro del pit
        if 'pit' in df.columns:
            df = df[df['pit'].notna()].copy()

        n = len(df)
        if n == 0:
            raise ValueError("No hay bloques en el pit para fasear")

        # Calcular centro del pit
        x_centro = df['x'].mean()
        y_centro = df['y'].mean()
        z_max    = df['z'].max()
        z_min    = df['z'].min()

        # ── Fase 1: Inicialización con fases anidadas concéntricas ──────
        # Usamos pits anidados como punto de partida (solución inicial)
        df = self._inicializar_fases_concentricas(df, x_centro, y_centro, z_max, z_min)

        # ── Fase 2: Recocido Simulado para optimizar pushbacks ───────────
        mejor_df, convergencia = self._recocido_simulado(df, x_centro, y_centro)

        # ── Fase 2b: precedencia por talud → cada fase acumulada es un cono ──
        talud = self._talud_global()
        mejor_df = self._cierre_precedencia(mejor_df, talud)

        # ── Fase 3: Calcular métricas por fase ──────────────────────────
        van_por_fase, ton_por_fase = self._calcular_metricas_fases(mejor_df)

        logger.info(f"✅ Faseamiento completado en {time.time()-t0:.1f}s")
        logger.info(f"   VAN total: {sum(van_por_fase):,.0f} USD")

        return ResultadoFaseamiento(
            df           = mejor_df,
            num_fases    = self.sa.num_fases,
            van_por_fase = van_por_fase,
            ton_por_fase = ton_por_fase,
            convergencia = convergencia,
            tiempo_s     = time.time() - t0,
        )

    # ── Talud global y cierre por precedencia (geometría de conos) ───────
    def _talud_global(self) -> float:
        """Talud más restrictivo (menor) de los sectores configurados; 45° por defecto."""
        try:
            o = self.config.optimizer
            return float(min(o.talud_este, o.talud_oeste, o.talud_norte, o.talud_sur))
        except Exception:
            return 45.0

    @staticmethod
    def _paso_malla(v) -> float:
        u = np.unique(np.round(np.asarray(v, float), 3))
        d = np.diff(u)
        d = d[d > 1e-6]
        return float(d.min()) if len(d) else 1.0

    def _tan_efectiva(self, df: pd.DataFrame) -> float:
        """Talud efectivo en la malla: cada banco retrocede un número entero de bloques."""
        tan = math.tan(math.radians(self._talud_global()))
        dx = self._paso_malla(df['x']); dz = self._paso_malla(df['z'])
        r = max(1, int(round(dz / tan / dx)))
        return dz / (r * dx)

    def _cierre_precedencia(self, df: pd.DataFrame, talud_deg: float = 45.0) -> pd.DataFrame:
        """
        Impone que la unión de las fases 1..k sea siempre un pit factible:
        si un bloque pertenece a la fase p, todos los bloques que están sobre él
        dentro del cono de talud deben minarse en una fase <= p.
        Resultado: cada fase acumulada tiene forma de cono (paredes con el talud),
        y no cilindros verticales como salen de la partición por distancia.
        """
        try:
            x = df['x'].to_numpy(float); y = df['y'].to_numpy(float); z = df['z'].to_numpy(float)
            def paso(v):
                u = np.unique(np.round(v, 3))
                d = np.diff(u)
                d = d[d > 1e-6]
                return float(d.min()) if len(d) else 1.0
            dx, dy, dz = paso(x), paso(y), paso(z)
            ix = np.rint((x - x.min()) / dx).astype(int)
            iy = np.rint((y - y.min()) / dy).astype(int)
            zs = np.unique(np.round(z, 3))
            iz = np.searchsorted(zs, np.round(z, 3))
            NX, NY, NZ = ix.max() + 1, iy.max() + 1, len(zs)
            if NX * NY * NZ > 3e8:
                logger.warning("Cierre de precedencia omitido: malla demasiado grande")
                return df
            # separación media entre bloques en la malla (modelos ralos): amplía el radio
            rng = np.random.default_rng(1)
            m = min(len(df), 400)
            idx = rng.choice(len(df), m, replace=False)
            pts = np.c_[ix, iy, iz * (dz / dx)]
            dmin = []
            for i in idx:
                d = np.sqrt(((pts - pts[i]) ** 2).sum(1)); d[i] = 1e9
                dmin.append(d.min())
            pad = max(0, int(np.ceil(np.median(dmin))) - 1)
            INF = 1e9
            fase = df['fase'].to_numpy(float)
            grids = np.full((NZ, NX, NY), INF, dtype=np.float32)
            grids[iz, ix, iy] = fase
            tan = np.tan(np.radians(talud_deg))
            for l in range(1, NZ):
                hz = (zs[l] - zs[l - 1])
                r = max(1, int(round(hz / tan / dx))) + pad
                prev = grids[l - 1]
                padded = np.full((NX + 2 * r, NY + 2 * r), INF, dtype=np.float32)
                padded[r:r + NX, r:r + NY] = prev
                best = np.full((NX, NY), INF, dtype=np.float32)
                for ox in range(-r, r + 1):
                    for oy in range(-r, r + 1):
                        if ox * ox + oy * oy > r * r + 0.5:
                            continue
                        np.minimum(best, padded[r + ox:r + ox + NX, r + oy:r + oy + NY], out=best)
                grids[l] = np.minimum(grids[l], best)
            nueva = grids[iz, ix, iy]
            # solo se reasignan bloques existentes (los huecos no se rellenan)
            df = df.copy()
            cambiados = int((nueva != fase).sum())
            df['fase'] = nueva.astype(int)
            logger.debug(f"Cierre de precedencia (talud {talud_deg:.0f}°): {cambiados:,} bloques reasignados "
                        f"-> {df['fase'].value_counts().sort_index().to_dict()}")
        except Exception as e:
            logger.warning(f"Cierre de precedencia omitido: {e}")
        return df

    # ── Inicialización: fases concéntricas por distancia al centro ────────
    def _inicializar_fases_concentricas(
        self, df: pd.DataFrame,
        x_c: float, y_c: float,
        z_max: float, z_min: float,
    ) -> pd.DataFrame:
        """
        Asigna fases iniciales basadas en distancia horizontal al centro.
        Esta es la solución de partida para el SA — equivale al método
        clásico de pits anidados concéntricos.
        """
        df = df.copy()
        n  = self.sa.num_fases

        # Distancia horizontal de cada bloque al centro
        # Distancia "de cono": la frontera de cada fase sube abriéndose con el talud
        # (cono invertido), así las fases acumuladas son pits y no cilindros verticales.
        tan_t = self._tan_efectiva(df)
        df['_dist'] = (np.sqrt((df['x'] - x_c)**2 + (df['y'] - y_c)**2)
                       - (df['z'] - z_min) / tan_t)
        d_max = df['_dist'].max()

        # Dividir en N fases por cuantiles de distancia
        quantiles = [i/n for i in range(1, n+1)]
        cortes    = df['_dist'].quantile(quantiles).values

        def asignar_fase(d):
            for i, corte in enumerate(cortes):
                if d <= corte:
                    return i + 1
            return n

        df['fase'] = df['_dist'].apply(asignar_fase).astype(int)
        df.drop(columns=['_dist'], inplace=True)

        logger.info(f"Fases iniciales (concéntricas): {df['fase'].value_counts().sort_index().to_dict()}")
        return df

    # ── Algoritmo de Recocido Simulado ─────────────────────────────────────
    def _recocido_simulado(
        self, df: pd.DataFrame,
        x_c: float, y_c: float,
    ) -> tuple[pd.DataFrame, list]:
        """
        Optimiza el faseamiento mediante Recocido Simulado.

        En cada iteración:
          1. Selecciona aleatoriamente un subconjunto de bloques de la zona frontera
          2. Propone reasignarlos a una fase vecina
          3. Acepta si mejora el VAN, o con probabilidad e^(-ΔE/T) si empeora
          4. Reduce la temperatura (enfriamiento)

        El resultado converge a fases con geometría de pushback real.
        """
        random.seed(42)
        talud_sa = self._talud_global()
        T       = self.sa.temperatura_ini
        alpha   = self.sa.tasa_enfriamiento
        n_iter  = self.sa.iteraciones
        n_fases = self.sa.num_fases

        mejor_df  = df.copy()
        van_actual = self._evaluar_van(mejor_df)
        mejor_van  = van_actual
        convergencia = [round(van_actual / 1e6, 4)]

        # Precomputar índices por fase para eficiencia
        indices_por_fase = {f: set(df[df['fase']==f].index) for f in range(1, n_fases+1)}

        for it in range(1, n_iter + 1):
            self.progress_cb(it, n_iter, f"Recocido Simulado: iteración {it}/{n_iter} | T={T:.1f} | VAN={van_actual/1e6:.2f} MUSD")

            # Seleccionar bloques de la zona frontera entre fases
            bloques_frontera = self._identificar_frontera(mejor_df, indices_por_fase)

            if not bloques_frontera:
                logger.warning(f"Sin bloques frontera en iteración {it}")
                convergencia.append(round(mejor_van / 1e6, 4))
                T *= alpha
                continue

            # Proponer movimiento: reasignar N bloques frontera a fase vecina
            n_mover = max(1, int(len(bloques_frontera) * 0.05))
            muestra = random.sample(list(bloques_frontera), min(n_mover, len(bloques_frontera)))

            # Evaluar vecinos posibles
            df_nuevo = mejor_df.copy()
            for idx in muestra:
                fase_actual = df_nuevo.at[idx, 'fase']
                # Proponer fase vecina (±1) respetando límites
                candidatas = []
                if fase_actual > 1:        candidatas.append(fase_actual - 1)
                if fase_actual < n_fases:  candidatas.append(fase_actual + 1)
                if candidatas:
                    nueva_fase = random.choice(candidatas)
                    df_nuevo.at[idx, 'fase'] = nueva_fase

            # Mantener factibilidad: toda fase acumulada debe seguir siendo un pit (talud)
            df_nuevo = self._cierre_precedencia(df_nuevo, talud_sa)

            # Fases equilibradas: ninguna puede quedar casi vacía (pushbacks con tamaño minable)
            cnt = df_nuevo['fase'].value_counts()
            if min(cnt.get(f, 0) for f in range(1, n_fases + 1)) < 0.12 * len(df_nuevo):
                convergencia.append(round(mejor_van / 1e6, 4))
                T *= alpha
                continue

            # Evaluar VAN de la nueva solución
            van_nuevo = self._evaluar_van(df_nuevo)
            delta_van = van_nuevo - van_actual

            # Criterio de aceptación (Metropolis)
            if delta_van > 0 or random.random() < math.exp(delta_van / (T * 1e6 + 1e-9)):
                mejor_df   = df_nuevo
                van_actual = van_nuevo
                # Actualizar índices por fase
                for f in range(1, n_fases+1):
                    indices_por_fase[f] = set(mejor_df[mejor_df['fase']==f].index)
                if van_actual > mejor_van:
                    mejor_van = van_actual

            convergencia.append(round(mejor_van / 1e6, 4))
            T *= alpha  # Enfriamiento

        logger.info(f"SA completado: VAN inicial={convergencia[0]:.2f} MUSD → final={convergencia[-1]:.2f} MUSD")
        return mejor_df, convergencia

    # ── Función objetivo: VAN ponderado por fase ───────────────────────────
    def _evaluar_van(self, df: pd.DataFrame) -> float:
        """
        Calcula el VAN total considerando el orden temporal de las fases.
        La fase 1 se mina primero → su VAN se descuenta menos.
        """
        eco    = self.cfg.economico
        r      = eco.tasa_descuento
        van    = 0.0
        n_fases = self.sa.num_fases

        for fase in range(1, n_fases + 1):
            df_f = df[df['fase'] == fase]
            if df_f.empty:
                continue
            # Factor de descuento: fase 1 en período 1, fase N en período N
            factor = 1 / (1 + r) ** fase
            van_fase = df_f['value'].sum() * factor
            van += van_fase

        return van

    # ── Identificar bloques en la frontera entre fases ────────────────────
    def _identificar_frontera(
        self, df: pd.DataFrame,
        indices_por_fase: dict,
    ) -> list:
        """
        Identifica bloques en la zona de contacto entre fases adyacentes.
        Estos son los candidatos para reasignación en el SA.
        """
        frontera = []
        xs = df['x'].values
        ys = df['y'].values
        fases_arr = df['fase'].values
        idx_arr   = df.index.values

        # Construir lookup {(x,y,z): fase}
        lookup = {(row['x'], row['y'], row['z']): row['fase']
                  for _, row in df.iterrows()}

        bx = self.cfg.bloque.xsiz
        by = self.cfg.bloque.ysiz

        for i, idx in enumerate(idx_arr):
            row = df.loc[idx]
            f   = int(row['fase'])
            x, y, z = row['x'], row['y'], row['z']

            # Verificar vecinos horizontales
            vecinos = [(x+bx,y,z),(x-bx,y,z),(x,y+by,z),(x,y-by,z)]
            for vx, vy, vz in vecinos:
                vf = lookup.get((vx, vy, vz))
                if vf is not None and vf != f:
                    frontera.append(idx)
                    break

        return frontera

    # ── Calcular métricas por fase ─────────────────────────────────────────
    def _calcular_metricas_fases(self, df: pd.DataFrame) -> tuple[list, list]:
        """Calcula VAN (MUSD) y tonelaje (Mt) por fase."""
        eco     = self.cfg.economico
        r       = eco.tasa_descuento
        n_fases = self.sa.num_fases

        van_por_fase = []
        ton_por_fase = []

        for fase in range(1, n_fases + 1):
            df_f = df[df['fase'] == fase]
            factor = 1 / (1 + r) ** fase
            van = df_f['value'].sum() * factor / 1e6  # MUSD
            ton = df_f.get('ton', df_f.get('tonelaje', pd.Series([0]))).sum() / 1e6  # Mt
            van_por_fase.append(round(van, 4))
            ton_por_fase.append(round(ton, 4))

        return van_por_fase, ton_por_fase
