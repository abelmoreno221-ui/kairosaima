# pmarb — Polymarket vs casas de apuestas (investigación + paper trading)

Busca partidos de fútbol en los que el precio de Polymarket no coincide con la probabilidad justa de una casa sharp (Pinnacle, una vez quitado el margen), decide cuál de las dos fuentes acierta más **con datos históricos** y solo genera señal cuando el EV supera los costes.

> **Alcance actual:** el repo solo lee datos públicos, hace backtests y registra señales en papel. No envía órdenes.
>
> **Pregunta de esta ronda:** ¿cuántas señales netas por semana quedan después de comisión, spread y liquidez?

### Estado regulatorio (fuentes)
- **Fecha del bloqueo, 26/05/2026, y expediente sancionador:** [nota de prensa de Consumo (dsca.gob.es)](https://www.dsca.gob.es/en/comunicacion/notas-prensa/consumo-abre-expediente-sancionador-plataformas-polymarket-kalshi-ordena) y [comunicado de la DGOJ (ordenacionjuego.es)](https://www.ordenacionjuego.es/en/node/3309).
- **Tipo de bloqueo:** es una orden de la DGOJ a los proveedores de internet, no un geobloqueo de Polymarket. España no aparece en la [lista de restricciones geográficas de Polymarket](https://help.polymarket.com/en/articles/13364163-geographic-restrictions).
- **Comisiones:** [Trading fees (Polymarket)](https://help.polymarket.com/en/articles/13364478-trading-fees).

### Pistas de datos (nunca se mezclan fuentes)
| Pista | Temporada | Referencia | Validez |
|---|---|---|---|
| **A** | 25/26 hasta el 08/01/2026 | Pinnacle (`PSH/PSD/PSA`, cierre `PSC*`) | **Único veredicto válido.** Pinnacle desaparece de football-data el 08/01/2026 (210 de 380 partidos en 25/26) |
| **B** | 26/27 | Betfair Exchange (`BFEH/BFED/BFEA`, cierre `BFEC*`) | Solo referencia: no hay Pinnacle |

- Si la casa de la pista no está en **todos** los ficheros, se pasa a otra para la pista entera, primero `Avg` y luego `Max`. `Max` es el último recurso porque es un outlier por construcción e infla el edge; si se usa, el resultado sale marcado como no válido.
- El backtest se aborta si el dataset mezcla casas.

### Horizontes
- **Apertura de football-data:** se recoge los viernes y martes por la tarde, así que equivale a **T-24h a T-72h** del saque. El backtest con esta línea toma el precio y el volumen de Polymarket a **T-24h**, y el escáner decide también a **T-24h** (±1h).
- **Cierre:** se fija al saque. El precio de Polymarket se toma a T-10min. Solo sirve como investigación.

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
python -m pmarb spreads sample      # cada hora, 1-2 semanas; solo mercados a <48h del saque
python -m pmarb spreads summary     # -> spread_table.json (mediana del medio spread por precio x horas al saque)
python -m pmarb backtest dataset_A.csv --spread-table spread_table.json   # veredicto
python -m pmarb backtest dataset_B.csv --spread-table spread_table.json   # referencia
python -m unittest discover -s tests -t .
```
La cobertura de fútbol de Polymarket empieza en 2025, por eso solo se usan 25/26 y 26/27.

## Cómo leer el resultado
- **`sig/wk`, `sig/wk_med` y `zero_wks`:** la cifra que decide el proyecto. Señales netas después de comisión, medio spread y volumen previo mínimo.
- **`half_spr`:** es **medio** spread en $ por acción. Comprar como taker cuesta `mid + half_spr` y se paga al entrar. `edge_dies_at_half_spread` es el primer nivel en el que el IC 95 % del ROI deja de quedar por encima de 0. El nivel `cal` usa la tabla medida con `spreads`.
- **Cierre frente a apertura:** el cierre no se conoce al decidir, así que solo vale como investigación. El veredicto es "estrategia candidata" únicamente si hay edge en apertura.
- **`method_sensitive`:** diferencia de ROI entre métodos de de-vig para cada línea y spread. Si supera 1 punto, se está midiendo el método.
- **Grupos:**
  - `con_pinnacle`: la casa sola ya dispara la misma operación.
  - `solo_modelo`: la señal existe solo porque el blend se separa también de la casa.
  
  Se reportan ROI y Brier de cada grupo. Si `solo_modelo` pierde (ROI < 0 con al menos 30 señales) en la línea de apertura, el veredicto **baja a "no candidata"**. Con menos de 30 señales se marca como no evaluable.
- **Guardarraíl:** diferencia de Brier entre modelo y casa, con IC 95 % por bootstrap de partidos. Si el modelo es peor que la casa más allá del IC, sale un aviso (`guardrail_warning`).
- **Volumen desconocido:** sale en una línea aparte (filas, señales por semana, ROI), fuera de las cifras principales. Suelen ser los mercados más líquidos.
- **Aborta sin dar cifras** si hay pocos partidos emparejados (`--min-coverage`, `--min-matches`).
