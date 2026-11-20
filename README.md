# GTA VI Trophy Rarity Forecast — make.fun entry

Forecasts every GTA VI PlayStation 5 trophy's earn rate on Exophase at
2026-11-26T00:00:00Z (one week after launch), as five quantiles per trophy,
from the house's recorded reads up to the deadline.

Competition: `gta6-trophy-rarity-forecast-01` (pot $250).

## Program

`main.py` — Python 3.13, standard library only, deterministic, offline.
Reads every `*.in` case in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes
one `<id>.out` per case to `$MAKEFUN_OUTPUT_DIR` (else `/output`).

## Model

An earn rate is earners / tracked players, and it moves under two forces:
**progression** (players earn the trophy, rate rises) and **dilution**
(Exophase refreshes its tracked-player count, so the rate falls).
Launch night was frozen at 638 tracked players, then refreshed to 3,868
with every rate lower — naive persistence is systematically wrong.

Per trophy, on the logit scale:

- **Trend**: least-squares slope of logit(rate) over the last 48 reads,
  clamped to ±3 logit/day.
- **Saturating extrapolation**: slope · τ · (1 − e^(−h/τ)), τ = 1.25 days —
  a multi-day horizon only ever buys about one day's worth of trend.
- **Week-one ceilings** per trophy class (from name/description):
  early story 40%, activities 25%, mid-game 5%, late/grind 1%,
  never below the trophy's current rate. The trophy at 100%
  (Gone Fishing) stays near 100%.
- **Frozen history** (no read ever changed): no trend exists, so use
  description-based priors — the very earliest beats rise
  (+1.3 logit/day: Chapter 1, Gone Fishing, Breakfast Run, fishing
  challenge), the next tier rises a little (+0.4: Chapter 2, arcade,
  bowling, 100 vehicles), everything else falls hard (−3.0 logit/day)
  because the next Exophase refresh dilutes it in one step.
- **Spread**: random-walk term growing with √horizon plus residual
  volatility (or, when frozen, prior volatility — an unseen refresh can
  move a rate ~1.8 logit at once), capped at 2.5 logit units.

## Verification (practice cases p00/p01, run locally)

- Format: 55 trophies × 5 quantiles, ids 1–55 once, finite,
  non-decreasing, 2,851 bytes per output (limit 65,536).
- Determinism: two runs byte-identical.
- Runtime: ~0.06 s for both cases (limit 8 s per case).
- Practice cut, pinball loss on the logit scale vs the RULES.txt-described
  answer approximation (refresh to 3,868 players, earners frozen):
  **0.8620** (typical miss 0.62 pp, 90% coverage 0.98) vs house baseline
  **1.6877** (miss 0.73 pp, coverage 0.00) — a 49% margin. Under a
  second, 4,276-earns-consistent answer approximation: 0.6902 vs 1.4014.
