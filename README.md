# pmarb — Polymarket vs casas de apuestas (investigación + paper trading)

Busca partidos de fútbol en los que el precio de Polymarket no coincide con la probabilidad justa de una casa sharp (Pinnacle, una vez quitado el margen), decide cuál de las dos fuentes acierta más **con datos históricos** y solo genera señal cuando el EV supera los costes.

> **Alcance actual:** el repo solo lee datos públicos, hace backtests y registra señales en papel. No envía órdenes.
>
> **Pregunta de esta ronda:** ¿cuántas señales netas por semana quedan después de comisión, spread y liquidez?

### Estado regulatorio (fuentes)
- **Fecha del bloqueo, 26/05/2026:** nota de prensa del Ministerio de Derechos Sociales, Consumo y Agenda 2030 en [dsca.gob.es](https://www.dsca.gob.es) y comunicado de la DGOJ en [ordenacionjuego.es](https://www.ordenacionjuego.es).
- **Tipo de bloqueo:** es una orden de la DGOJ a los proveedores de internet, no un geobloqueo de Polymarket. España no aparece en la lista de países restringidos del centro de ayuda de Polymarket ([help.polymarket.com](https://help.polymarket.com)).
- *Pendiente:* sustituir los enlaces a la portada por las URL exactas de cada documento.

## Pipeline
| Paso | Módulo | Qué hace |
|---|---|---|
| 1 | `odds.py` | Cuotas históricas gratis de football-data.co.uk (cierre de Pinnacle `PSCH/PSCD/PSCA`) y en vivo de The Odds API (plan gratis) |
| 2 | `devig.py` | Quita el margen con tres métodos: `shin`, `prop` (proporcional) y `log` (logarítmico/power). El backtest los compara |
| 3 | `polymarket.py` | Gamma API (eventos/mercados) + CLOB público (`prices-history`, `book`). Precio ≤ t, sin lookahead |
| 4 | `matching.py` | Empareja equipos (normalización, alias, fuzzy) y mapea cada mercado binario a H/D/A |
| 5 | `model.py` | Calibración: Brier, log-loss, ECE, curva de fiabilidad. `Blend` = logística sobre `logit(p_pm)` y `logit(p_book)` con L2. **Los pesos te dicen quién tiene razón** |
| 6a | `spreads.py` | Muestrea `/spread` y `/book` en mercados de fútbol vivos y calcula la distribución del medio spread por tramo de precio (`spread_table.json`) |
| 6 | `backtest.py` | Walk-forward sobre la rejilla {cierre, apertura} × {shin, prop, log} × medio spread {0,01, 0,02, 0,03, 0,05 [, calibrado]}. La comisión de taker (`shares·0,05·p·(1−p)`) se cobra al entrar, gane o pierda. El EV se calcula sobre el ask. La liquidez es el volumen negociado **antes** de T (data-api `/trades`, una consulta por mercado y en caché). Reporta señales por semana, nivel de spread al que muere el edge, grupos de señales y guardarraíl de Brier |
| 7 | `scan.py` | Escáner en vivo: compara Pinnacle con el order book real y escribe las señales en `signals.csv` |

## Uso
```bash
pip install -r requirements.txt
python -m pmarb build --seasons 2526 2627 --leagues E0 SP1 I1 D1 F1 --book PS   # aborta si la cobertura es baja
python -m pmarb spreads sample      # repetir cada hora 1-2 semanas (cron)
python -m pmarb spreads summary     # -> spread_table.json (mediana del medio spread por tramo)
python -m pmarb backtest dataset.csv --spread-table spread_table.json
python -m unittest discover -s tests -t .
```
Temporadas 25/26 y 26/27, porque la cobertura de fútbol de Polymarket empieza en 2025. Si un fichero no trae columnas de Pinnacle (`PS*`), `build` se detiene y lista las casas disponibles (`--book Avg|Max|B365`).

## Cómo leer el resultado
- **`sig/wk`, `sig/wk_med` y `zero_wks`:** la cifra que decide el proyecto. Señales netas después de comisión, medio spread y volumen previo mínimo.
- **`half_spr`:** es **medio** spread en $ por acción. Comprar como taker cuesta `mid + half_spr` y se paga al entrar. `edge_dies_at_half_spread` es el primer nivel en el que el IC 95 % del ROI deja de quedar por encima de 0. El nivel `cal` usa la tabla medida con `spreads`.
- **Cierre frente a apertura:** el cierre no se conoce al decidir, así que solo vale como investigación. El veredicto es "estrategia candidata" únicamente si hay edge en apertura.
- **`method_sensitive`:** diferencia de ROI entre métodos de de-vig para cada línea y spread. Si supera 1 punto, se está midiendo el método.
- **Grupos:**
  - `con_pinnacle`: la casa sola ya dispara la misma operación.
  - `solo_modelo`: la señal existe solo porque el blend se separa también de la casa.
  
  Se reportan ROI y Brier de cada grupo. Si `solo_modelo` no aguanta (`solo_modelo_fails_open`), es el ruido del blend el que fabrica las señales.
- **Guardarraíl:** diferencia de Brier entre modelo y casa, con IC 95 % por bootstrap de partidos. Si el modelo es peor que la casa más allá del IC, sale un aviso (`guardrail_warning`).
- **Aborta sin dar cifras** si hay pocos partidos emparejados (`--min-coverage`, `--min-matches`).
