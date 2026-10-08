# Auditoría Global Mine Planner — estado verificado (v69)

Todo lo marcado ✅ tiene evidencia de ejecución (pruebas automáticas `tests/`, corrida con el modelo real de 127.405 bloques y corrida punta a punta por la API). Lo marcado ⛔ **no está implementado** o **no está verificado**.

## 1. Diagnóstico de errores encontrados

| # | Error | Causa | Corrección |
|---|-------|-------|-----------|
| 1 | Plan "no constante": movimiento 7,5 → 3,2 → 7,5 Mt, capacidad ociosa | El scheduler cortaba el banco completo (`break`) al primer panel que no cabía y no abría fases siguientes | Reescritura del bucle: llena capacidad saltando paneles que no caben; fase abierta nunca se cierra; abre la siguiente si sobra capacidad; meta de movimiento = movimiento total / vida del pit |
| 2 | **Precedencia vertical violada** con el modelo real (14 columnas: banco profundo antes que el superior) | Sólo se exigía el banco *inmediatamente* superior; si faltaba ese panel (modelos decimados) no había restricción | Precedencia = panel existente más cercano arriba en la columna (encadena toda la columna) + vecinos 3×3 del banco superior (talud) |
| 3 | Curvas ficticias en ¿Cuánto? cuando faltaban datos del backend (`rf*62`, `*1.15`, `*0.72`, VAN=30) | Respaldos "sintéticos" en el visor | Eliminados: sin datos reales el gráfico se oculta |
| 4 | Stockpile sin tope; no se distinguía mineral directo vs. recuperado | Todo el mineral pasaba por stock; costo de remanejo sólo al marginal | Balance explícito directo / a stock / desde stock; tope de inventario y de recuperación configurables |
| 5 | Etiqueta "% Cu" fija en ley de corte | Texto fijo en el visor | Usa unidad y metal del modelo (g/t Au, etc.) |

## 2. Archivos modificados
`engine/scheduler.py` (secuenciamiento, precedencias, balance de masa, metal, causas de detención), `engine/config.py` (t/día, días efectivos, tope de stock, precedencia espacial), `engine/reconciliacion.py` (nuevo), `engine/dxf_export.py` (nuevo), `api.py` (DXF, reconciliación), `gmp_v69.html` (panel de alimentación, controles, sin curvas ficticias), `tests/` (nuevo).

## 3. Matriz de trazabilidad (algoritmo → pestaña)

| Pestaña | Módulo / función | Algoritmo | Invocado desde | Entradas | Salida | Consumido por (UI) | Evidencia |
|---|---|---|---|---|---|---|---|
| Modelo / ¿Cuánto? | `engine/loader.py` | Lectura, alias de columnas, densidad por bloque, tonelaje, ley media, ley de corte | `pipeline.ejecutar` paso 1 | CSV/TXT/ASC | modelo normalizado + `resumen_modelo` | KPIs, visor Modelo | corrida real ✅ |
| ¿Cuánto? | `engine/optimizer.py` | Lerchs-Grossmann (PyMaxflow), 25 escenarios de factor de ingresos | paso 2 | bloques, economía | `resumen_pits` (curvas pit-by-pit), `pit` por bloque | gráfico "Pozo por pozo", pit shell | corrida real ✅; reconciliación pit↔fases ✅ |
| ¿Cómo? | `engine/faseamiento.py` | Recocido simulado + inicialización por distancia de cono + cierre de precedencia (talud 45°) | paso 3 | bloques del pit final | `fase` por bloque | pestaña Fases | ✅ (cierre validado en pruebas previas); ⛔ sin prueba automática propia |
| ¿Cuándo? | `engine/scheduler.py` `HeuristicaFaseBanco` | Heurística fase-banco, capacidades, stock, ley, VAN | paso 4 | fases, capacidades | `plan`, `bloques` con período, `fase_periodo`, `interrupciones` | Gantt, tablas, panel de alimentación | 20 pruebas ✅ |
| ¿Cuándo? | `engine/reconciliacion.py` | 14 controles cruzados | `api._run_job` | resultados | `reconciliacion` | Resumen → "Controles de reconciliación" | ✅ corrida real y por API |
| Descargas | `engine/dxf_export.py` | Caras exteriores de sólidos (fases, períodos, superficie del pit) | `api._run_job` | bloques con fase/período | `.dxf` | botón "Descargar DXF" | ✅ lectura con ezdxf |

## 4. Resultados de pruebas
- `pytest tests/` → **20 aprobadas, 0 fallidas**: conservación de toneladas, sin duplicados, precedencia vertical (también con bancos faltantes) y espacial, capacidades mina/planta (t/año y t/día), tope de stockpile, balance de stock, ley ponderada (cobre) y metal en onzas (oro), continuidad de fases justificada, sin períodos vacíos, VAN = Σ flujos, DXF (capas y geometría), reconciliación (aprueba y detecta inconsistencias).
- **Pruebas de mutación**: se introdujeron a propósito 5 errores (sin precedencia, capacidad de planta +50 %, balance de stock −10 %, ley −3 %, sin tope de stock) → **las 5 fueron detectadas**.
- Modelo real (20.172 bloques en el pit): 14/14 controles de reconciliación aprobados tras la corrección #2 (antes: 13/14).
- Corrida punta a punta por API: 14/14 aprobados, DXF generado.

## 5. Balances (modelo real, VAN 265,2 MUSD)
Mineral 2,40 Mt/año constante desde el año 5; alimentación 2,20 Mt/año; stock crece 0,2 Mt/año (mina > planta); Σ mineral minado = Σ alimentado + inventario final (diferencia 1e-14 Mt). Metal contenido año 5: 12.374 t = 2,2 Mt × 0,562 %.

## 6. Pendiente / no verificado
- ⛔ **Lectura de DXF de entrada** (topografía, diseños, contornos): hoy sólo se **exporta** DXF; no se lee ninguno. Los archivos aceptados (CSV/TXT/ASC) no contienen geometrías. Requiere función nueva (lectura con `ezdxf`, validación de cierre y coordenadas, y visualización).
- ⛔ Recálculo/invalidación automática al cambiar parámetros: hoy cada cambio exige "Ejecutar pipeline" de nuevo; no hay caché parcial por etapa.
- ⛔ Optimización del secuenciamiento: es una heurística, no un optimizador económico (MIP/Whittle). Las interrupciones se registran con causa, pero no se prueban contra un óptimo.
- ⛔ Restricciones de acceso (rampas), destinos múltiples (lixiviación/botadero) y escenario explícito de sobrecapacidad.
- ⛔ Precedencia entre fases (fase N+1 nunca sobre el banco aún no minado de la N) sólo se controla por apertura (lag/avance); no por panel.
- ⛔ El VAN del plan y las curvas pit-by-pit no están reconciliados (usan supuestos distintos: el pit-by-pit no considera capacidades).
- ⛔ El motor decima a 50.000 bloques; las leyes/tonelajes reportados son los del modelo decimado.
- ⛔ Faseamiento (recocido simulado) sin pruebas automáticas propias.
