# pmarb — Polymarket vs casas de apuestas (investigación + paper trading)

Busca partidos de fútbol en los que el precio de Polymarket no coincide con la probabilidad justa de una casa sharp (Pinnacle, una vez quitado el margen), decide cuál de las dos fuentes acierta más **con datos históricos** y solo genera señal cuando el EV supera los costes.

> **Alcance actual:** el repo solo lee datos públicos, hace backtests y registra señales en papel. No envía órdenes.
>
> **Pregunta de esta ronda:** ¿cuántas señales netas por semana quedan después de comisión, spread y liquidez?

### Estado regulatorio (fuentes)
- **Fecha del bloqueo, 26/05/2026, y expediente sancionador:** [nota de prensa de Consumo (dsca.gob.es)](https://www.dsca.gob.es/en/comunicacion/notas-prensa/consumo-abre-expediente-sancionador-plataformas-polymarket-kalshi-ordena) y [comunicado de la DGOJ (ordenacionjuego.es)](https://www.ordenacionjuego.es/en/node/3309).
- **Tipo de bloqueo:** es una orden de la DGOJ a los proveedores de internet, no un geobloqueo de Polymarket. España no aparece en la [lista de restricciones geográficas de Polymarket](https://help.polymarket.com/en/articles/13364163-geographic-restrictions).
- **Comisiones:** [Trading fees (Polymarket)](https://help.polymarket.com/en/articles/13364478-trading-fees).

### Pistas de datos (una pista = una casa; nunca se cambia de casa)
| Pista | Temporada | Referencia | Validez |
|---|---|---|---|
| **A** | 25/26 hasta el 08/01/2026 | Pinnacle (`PSH/PSD/PSA`, cierre `PSC*`) | **Único veredicto válido.** Pinnacle desaparece de football-data el 08/01/2026 (210 de 380 partidos en 25/26) |
| **B** | 26/27 | Betfair Exchange (`BFEH/BFED/BFEA`, cierre `BFEC*`) | Solo referencia |
| **C** | 25/26 + 26/27 | `Avg` (o `Max` con `--book Max`) | **Inválida**, solo diagnóstico. `Max` es un outlier por construcción e infla el edge |

- **Valor vacío:** se descarta la fila y se cuenta. El informe muestra por fichero las fechas descartadas y si forman un **bloque** (corte de la fuente, p. ej. "bloque final desde 2026-01-08") o un **goteo** (huecos sueltos).
- **Columna ausente en un fichero** (p. ej. `PSH` en 26/27): aviso con el nombre del fichero, ese fichero se omite, la temporada se trunca en el último partido con datos y se continúa. Solo se aborta si la columna falta en todos los ficheros, con el mensaje "columna ausente".
- El backtest se aborta si un dataset mezcla casas.
- **Betfair Exchange:** es un exchange, no una casa. No tiene margen en la cuota; cobra una comisión del 5 % sobre la ganancia neta, rebajable al 2 %. En 25/26 hay 22 partidos sin cierre de BFE, que se descartan.

### Horizontes (`backtest --decision-h`)
| `--decision-h` | Cuotas | Precio y volumen de PM | Papel |
|---|---|---|---|
| `6` | apertura | T-6h | **Operativo.** Es el horizonte por defecto del escáner (T-6h ±1h). En vivo, el escáner usa la cuota de la casa del momento; en el backtest, la apertura es más antigua, así que es un proxy prudente |
| `24` (por defecto) | apertura (recogida viernes/martes por la tarde = T-24h..T-72h) | T-24h | **Validación conservadora** |
| `close` | cierre | T-10min | **Techo de investigación.** El cierre no se conoce al decidir, así que el veredicto nunca pasa de "techo de investigación" |

`build` guarda los tres puntos (`ts_/p_pm_/vol_` + `24`, `6`, `close`) con una sola llamada a `prices-history` por mercado, de modo que cambiar de horizonte no obliga a reconstruir el dataset.

## Pipeline
| Paso | Módulo | Qué hace |
|---|---|---|
| 1 | `odds.py` | Cuotas históricas gratis de football-data.co.uk (cierre de Pinnacle `PSCH/PSCD/PSCA`) y en vivo de The Odds API (plan gratis) |
| 2 | `devig.py` | Quita el margen con tres métodos: `shin`, `prop` (proporcional) y `log` (logarítmico/power). El backtest los compara |
| 3 | `polymarket.py` | Gamma API (eventos/mercados) + CLOB público (`prices-history`, `book`). Precio ≤ t, sin lookahead |
| 4 | `matching.py` | Empareja equipos (normalización, alias, fuzzy) y mapea cada mercado binario a H/D/A |
| 5 | `model.py` | Calibración: Brier, log-loss, ECE, curva de fiabilidad. `Blend` = logística sobre `logit(p_pm)` y `logit(p_book)` con L2. **Los pesos te dicen quién tiene razón** |
| 6a | `spreads.py` | Muestrea `/spread` y `/book` en mercados de fútbol vivos y calcula la distribución del medio spread por tramo de precio (`spread_table.json`) |
| 6 | `backtest.py` | Walk-forward sobre la rejilla {cierre, apertura} × {shin, prop, log} × medio spread {0,01, 0,02, 0,03, 0,05 [, calibrado]}. La comisión de taker (`shares·0,05·p·(1−p)`) se cobra al entrar, gane o pierda. El EV se calcula sobre el ask. La liquidez es el volumen negociado en los 7 días previos a cada punto de decisión, sacado de data-api `/trades`. Ese endpoint devuelve lo más reciente primero, así que se pagina con `offset` hasta pasar el inicio de la ventana; cada mercado se consulta una vez y se guarda en caché. Si se agota el offset antes, el volumen queda como **desconocido** y esos mercados se reportan en una línea aparte, sin descartarlos. Reporta señales por semana, nivel de spread al que muere el edge, grupos de señales y guardarraíl de Brier |
| 7 | `scan.py` | Escáner en vivo: compara Pinnacle con el order book real y escribe las señales en `signals.csv` |

## Uso
```bash
pip install -r requirements.txt
python -m pmarb build --track A     # 25/26 Pinnacle hasta 08/01/2026 -> dataset_A.csv (aborta si la cobertura es baja)
python -m pmarb build --track B     # 26/27 Betfair Exchange           -> dataset_B.csv
python -m pmarb build --track C --book Avg   # diagnóstico, inválida
python -m pmarb backtest dataset_A.csv --decision-h 6       # horizonte operativo
python -m pmarb backtest dataset_A.csv --decision-h close   # techo de investigación
python -m pmarb spreads sample      # cada hora, 1-2 semanas; solo mercados a <48h del saque
python -m pmarb spreads summary     # -> spread_table.json (mediana del medio spread por precio x horas al saque)
python -m pmarb backtest dataset_A.csv --spread-table spread_table.json   # veredicto
python -m pmarb backtest dataset_B.csv --spread-table spread_table.json   # referencia
python -m unittest discover -s tests -t .
```
La cobertura de fútbol de Polymarket empieza en 2025, por eso solo se usan 25/26 y 26/27.

## Cómo leer el resultado
- **Embudo por liga (diagnóstico principal):** cuatro escalones, en el horizonte de `--decision-h`:
  1. partidos emparejados,
  2. partidos con mercado vivo en el horizonte elegido,
  3. partidos que superan el filtro de volumen (los de volumen desconocido van en columna aparte),
  4. partidos con señal neta (solo en el periodo fuera de muestra).

  Si el tercer escalón deja **menos de 20 partidos en una liga**, esa liga sale marcada: el horizonte elegido no es viable ahí (con T-24h, hay que probar `--decision-h 6`).
- **Suelo de muestra:** con menos de **150 apuestas colocadas**, ningún veredicto pasa de "no evaluable", aunque el IC salga limpio.
- **`regla_pin`:** la regla sin modelo, es decir, Pinnacle directa frente al ask con el mismo umbral y los mismos costes. Pasa por **la misma máscara de filas** que el modelo (`eligible`: mercado vivo, cobertura, periodo fuera de muestra y volumen). Hay un test que lo comprueba.
  - **El veredicto pasa automáticamente a "candidata: REGLA SIN MODELO"** si la regla está viva (IC > 0, al menos 150 apuestas y Pinnacle mejor que PM en log-loss) y además el modelo no puede ser candidato (no vivo, `solo_modelo` pierde o no es evaluable) o rinde menos.
  - El ROI del grupo `con_pinnacle` **no** es el de esta regla: esas señales las elige y las dimensiona el modelo.
- **`sig/wk`, `sig/wk_med` y `zero_wks`:** la cifra que decide el proyecto. Señales netas después de comisión, medio spread y volumen previo mínimo.
- **`half_spr`:** es **medio** spread en $ por acción. Comprar como taker cuesta `mid + half_spr` y se paga al entrar. `edge_dies_at_half_spread` es el primer nivel en el que el IC 95 % del ROI deja de quedar por encima de 0. El nivel `cal` usa la tabla medida con `spreads`.
- **Cierre frente a apertura:** el cierre no se conoce al decidir, así que solo vale como investigación. El veredicto es "estrategia candidata" únicamente si hay edge en apertura.
- **`method_sensitive`:** diferencia de ROI entre métodos de de-vig para cada línea y spread. Si supera 1 punto, se está midiendo el método.
- **Grupos:**
  - `con_pinnacle`: la casa sola ya dispara la misma operación.
  - `solo_modelo`: la señal existe solo porque el blend se separa también de la casa.
  
  Se reportan ROI y Brier de cada grupo. Si `solo_modelo` pierde (ROI < 0 con al menos 30 señales) en la línea de apertura, el veredicto **baja a "no candidata"**. Con menos de 30 señales, el veredicto es "no evaluable".
- **Guardarraíl:** diferencia de Brier entre modelo y casa, con IC 95 % por bootstrap de partidos. Si el modelo es peor que la casa más allá del IC, sale un aviso (`guardrail_warning`).
- **Volumen desconocido:** sale en una línea aparte (filas, señales por semana, ROI), fuera de las cifras principales. Suelen ser los mercados más líquidos.
- **Aborta sin dar cifras** si hay pocos partidos emparejados (`--min-coverage`, `--min-matches`).
