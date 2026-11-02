# pmarb — Polymarket vs casas de apuestas (investigación + paper trading)

Busca partidos de fútbol en los que el precio de Polymarket no coincide con la probabilidad justa de una casa sharp (Pinnacle, una vez quitado el margen), decide cuál de las dos fuentes acierta más **con datos históricos** y solo genera señal cuando el EV supera los costes.

> **Alcance actual:** el repo solo lee datos públicos, hace backtests y registra señales en papel. No envía órdenes. En España, Polymarket está bloqueado desde el 26/05/2026 por una orden de la DGOJ a los proveedores de internet; no es un geobloqueo de la propia Polymarket.
>
> **Pregunta de esta ronda:** ¿cuántas señales netas por semana quedan después de comisión, spread y filtro de liquidez?

## Pipeline
| Paso | Módulo | Qué hace |
|---|---|---|
| 1 | `odds.py` | Cuotas históricas gratis de football-data.co.uk (cierre de Pinnacle `PSCH/PSCD/PSCA`) y en vivo de The Odds API (plan gratis) |
| 2 | `devig.py` | Quita el margen con tres métodos: `shin`, `prop` (proporcional) y `log` (logarítmico/power). El backtest los compara |
| 3 | `polymarket.py` | Gamma API (eventos/mercados) + CLOB público (`prices-history`, `book`). Precio ≤ t, sin lookahead |
| 4 | `matching.py` | Empareja equipos (normalización, alias, fuzzy) y mapea cada mercado binario a H/D/A |
| 5 | `model.py` | Calibración: Brier, log-loss, ECE, curva de fiabilidad. `Blend` = logística sobre `logit(p_pm)` y `logit(p_book)` con L2. **Los pesos te dicen quién tiene razón** |
| 6 | `backtest.py` | Walk-forward con una rejilla de {cierre, apertura} × {shin, prop, log}. Comisión de taker real: `shares·0,05·p·(1−p)`, que equivale al 2,5 % del stake a 0,50. EV neto mínimo del 4 %, filtro de liquidez, tamaño de entrenamiento por bloque, **señales/semana** e IC 95 % |
| 7 | `scan.py` | Escáner en vivo: compara Pinnacle con el order book real y escribe las señales en `signals.csv` |

## Uso
```bash
pip install -r requirements.txt
python -m pmarb synthetic --out ds.csv            # probar el pipeline con datos simulados
python -m pmarb build --fd E0_2324.csv E0_2425.csv SP1_2425.csv --out ds.csv
python -m pmarb backtest ds.csv --min-edge 0.04 --min-liquidity 1000   # rejilla + veredicto -> backtest.json
python -m pmarb fit ds.csv                        # -> weights.json
ODDS_API_KEY=xxx python -m pmarb scan --sport soccer_epl
python -m unittest discover -s tests -t .
```

## Cómo leer el resultado
- **`sig/wk_mean`, `sig/wk_med` y `zero_wks`:** son la cifra que decide el proyecto. Cuentan las señales que quedan después de comisión, spread y liquidez.
- **Cierre frente a apertura:** la línea de cierre no se conoce en el momento de decidir, así que solo vale como referencia de investigación. La de apertura (`PSH/PSD/PSA`) sirve de proxy de lo disponible a T-6h. El veredicto es "estrategia candidata" únicamente si hay edge en apertura.
- **`*_method_sensitive`:** si el ROI se mueve más de 1 punto entre los métodos de de-vig, se está midiendo el método y no un edge.
- **`beats_pm`:** es imprescindible. **`beats_book`:** dice si mezclar con Polymarket aporta algo frente a usar solo la casa.
- **Bloques:** el tamaño de entrenamiento de cada bloque sale por pantalla y en `backtest.json`.
- **Limitaciones conocidas:** la liquidez histórica es el valor final que da Gamma (un proxy, no una serie) y el spread histórico es un supuesto (`--half-spread`).
