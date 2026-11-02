# pmarb — Polymarket vs casas de apuestas (investigación + paper trading)

Busca partidos de fútbol en los que el precio de Polymarket no coincide con la probabilidad justa de una casa sharp (Pinnacle, una vez quitado el margen), decide cuál de las dos fuentes acierta más **con datos históricos** y solo genera señal cuando el EV supera los costes.

> ⚠️ **Polymarket está bloqueado en España** (DGOJ, 2025) y lo bloquea por geolocalización. Operar desde España con VPN incumple sus términos y la Ley 13/2011 del juego. Este repo **no ejecuta órdenes**: solo lee datos públicos, hace backtests y registra señales en papel.

## Pipeline
| Paso | Módulo | Qué hace |
|---|---|---|
| 1 | `odds.py` | Cuotas históricas gratis de football-data.co.uk (cierre de Pinnacle `PSCH/PSCD/PSCA`) y en vivo de The Odds API (plan gratis) |
| 2 | `devig.py` | Quita el margen: multiplicativo, power o **Shin** (por defecto; corrige el sesgo favorito-longshot) |
| 3 | `polymarket.py` | Gamma API (eventos/mercados) + CLOB público (`prices-history`, `book`). Precio ≤ t, sin lookahead |
| 4 | `matching.py` | Empareja equipos (normalización, alias, fuzzy) y mapea cada mercado binario a H/D/A |
| 5 | `model.py` | Calibración: Brier, log-loss, ECE, curva de fiabilidad. `Blend` = logística sobre `logit(p_pm)` y `logit(p_book)` con L2. **Los pesos te dicen quién tiene razón** |
| 6 | `backtest.py` | Walk-forward: compra YES o NO solo si `p_modelo − ask − fee > min_edge`. Kelly fraccional, tope en USDC, drawdown, IC 95 % por bootstrap |
| 7 | `scan.py` | Escáner en vivo: compara Pinnacle con el order book real y escribe las señales en `signals.csv` |

## Uso
```bash
pip install -r requirements.txt
python -m pmarb synthetic --out ds.csv            # probar el pipeline con datos simulados
python -m pmarb build --fd E0_2324.csv E0_2425.csv SP1_2425.csv --out ds.csv
python -m pmarb backtest ds.csv --fee-rate 0.0 --min-edge 0.02
python -m pmarb fit ds.csv                        # -> weights.json
ODDS_API_KEY=xxx python -m pmarb scan --sport soccer_epl
python -m unittest discover -s tests -t .
```

## Qué validar antes de creerte un resultado
- **El modelo tiene que batir a las dos fuentes por separado** en log-loss fuera de muestra (`oos_model` frente a `oos_pm` y `oos_book`). Si no las bate, no hay edge.
- **El límite inferior de `roi_per_trade_ci95` tiene que ser mayor que 0.** El test `test_no_edge_when_sources_equal` muestra que un modelo sobreajustado se inventa "edges" que pierden dinero; por eso la regularización está en L2 = 30.
- **Lookahead:** el cierre de Pinnacle se fija en el saque inicial. Usa `--lead-min` pequeño (precio de Polymarket justo antes del partido) o cuotas de apertura.
- **Costes reales:** consulta el fee vigente de cada mercado (`feeRate` en Gamma, fórmula `rate·min(p,1−p)` por share), el spread real y la profundidad del libro. Operar no paga gas (los relayers son de Polymarket); depositar y retirar sí.
- **Arbitraje con las dos patas:** las casas con licencia en España limitan rápido a quien gana. La versión sostenible usa la casa sharp solo como señal.
