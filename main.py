#!/usr/bin/env python3
"""Forecast GTA VI PS5 trophy earn rates on Exophase at a target date.

Reads every ``*.in`` case in the input directory and writes one
``<id>.out`` forecast per case to the output directory.

Model (per trophy, on the logit scale)
--------------------------------------
An earn rate is earners / tracked players. It moves under two forces:
progression (players earn the trophy, rate rises) and dilution (Exophase
refreshes its tracked-player count, new players who have not earned the
trophy yet enter the denominator, rate falls). Launch night was frozen
at 638 tracked players, then refreshed to 3,868 with every rate lower,
so naive persistence is systematically wrong.

* Trend: least-squares slope of logit(rate) over the last 48 reads,
  clamped to +/-3 logit/day.
* Extrapolation saturates: slope * tau * (1 - exp(-h / tau)) with
  tau = 1.25 days, so a multi-day horizon only ever buys about one
  day's worth of trend.
* Week-one ceiling per trophy class (early story 40%, activities 25%,
  mid-game 5%, late/grind 1%), never below the trophy's current rate.
* Frozen history (no read ever changed): no trend exists, so use
  description-based priors instead -- the very earliest beats rise
  (+1.3 logit/day), the next tier rises a little (+0.4), everything
  else falls hard (-3.0 logit/day) because the next Exophase refresh
  dilutes it in one step.
* Spread: a random-walk term growing with sqrt(horizon) plus the
  fit's residual volatility (or, when frozen, the prior volatility,
  which is large because an unseen refresh can move a rate by
  ~1.8 logit at once), capped at 2.5 logit units.

Standard library only, deterministic, no network.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

# Standard-normal quantiles at the five forecast levels.
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)

TAU_DAYS = 1.25          # saturation timescale of the trend integral
SLOPE_CLAMP = 3.0        # logit/day, either direction
WINDOW = 48              # reads used for the trend fit
SPREAD_CAP = 2.5         # logit units, max 5%-95% half-spread scale
RW_COEF = 0.5            # random-walk spread coefficient
FROZEN_VOL = 1.6         # logit spread when history is frozen (a refresh
                         # we have not seen can move a rate ~1.8 logit)
RESID_FLOOR = 0.15       # minimum residual volatility for a fitted trend

# Week-one ceilings (percent) per trophy class.
CEILING_EARLY_STORY = 40.0
CEILING_ACTIVITY = 25.0
CEILING_MIDGAME = 5.0
CEILING_LATE = 1.0

# Frozen-history priors (logit/day) by description.
PRIOR_EARLIEST = 1.3
PRIOR_EARLY = 0.4
PRIOR_FALL = -3.0


def parse_time(text):
    """Parse an ISO-8601 UTC timestamp to epoch seconds."""
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def clamp_rate(rate):
    """Clamp a percent rate into the range the logit can represent."""
    return min(99.99, max(0.01, float(rate)))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(x):
    if x >= 0.0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def classify(text, tier):
    """Week-one ceiling class from a trophy's name + description."""
    t = text.lower()
    if tier == "platinum":
        return CEILING_LATE
    if ("chapter 1" in t or "chapter 2" in t or "chapter 3" in t
            or "chapter 4" in t or "gone fishing" in t
            or "breakfast run" in t):
        return CEILING_EARLY_STORY
    # Stranger / side-strand finales come before the generic "final
    # mission" rule: they are mid-game, the story finale is late.
    if ("complete the final mission for" in t
            or "complete 8 mission strands" in t
            or "explore all of leonida" in t
            or "complete the cartel heist" in t
            or "complete child's play" in t
            or "complete headliners" in t
            or "complete vcia takedown" in t
            or "complete operation savage glory" in t
            or "hunt a unique animal" in t
            or "earn $250,000 laundering" in t
            or "purchase all garage" in t):
        return CEILING_MIDGAME
    if ("final mission" in t or "100% completion" in t
            or "250 stunts" in t or "gold medal in all" in t
            or "100 locations" in t or "50 challenges" in t
            or "5 bank vaults" in t or "collectible card sets" in t
            or "meditate at every spot" in t):
        return CEILING_LATE
    return CEILING_ACTIVITY


def frozen_prior(text):
    """Description-based prior slope (logit/day) for a frozen trophy."""
    t = text.lower()
    if ("chapter 1" in t or "gone fishing" in t or "breakfast run" in t
            or "fishing challenge" in t):
        return PRIOR_EARLIEST
    if ("chapter 2" in t or "arcade" in t or "bowling" in t
            or "100 vehicles" in t):
        return PRIOR_EARLY
    return PRIOR_FALL


def least_squares(xs, ys):
    """Return (intercept, slope) of the least-squares line through xs, ys."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:
        return my, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return my - slope * mx, slope


def forecast_trophy(trophy, history, horizon_days):
    """Five quantiles (percent) for one trophy at the target."""
    text = trophy["name"] + " " + trophy.get("description", "")
    ceiling = classify(text, trophy.get("tier", ""))

    # Collect (time in days, logit rate) pairs, oldest first.
    points = []
    for read in history:
        rate = read["rates"][trophy["index"]]
        if rate is None:
            continue
        points.append((parse_time(read["at"]) / 86400.0, logit(rate)))
    if not points:
        # No data at all: a rare trophy near the floor, honestly wide.
        median = logit(0.5)
        spread = SPREAD_CAP
        return quantiles(median, spread)

    last_rate = inv_logit(points[-1][1])
    ceiling = max(ceiling, last_rate)

    frozen = len({round(y, 9) for _, y in points}) <= 1
    if frozen:
        slope = frozen_prior(text)
        base = points[-1][1]
        spread = min(SPREAD_CAP, RW_COEF * math.sqrt(max(horizon_days, 0.0))
                     + FROZEN_VOL)
    else:
        window = points[-WINDOW:]
        xs = [x - window[0][0] for x, _ in window]
        ys = [y for _, y in window]
        intercept, slope = least_squares(xs, ys)
        slope = max(-SLOPE_CLAMP, min(SLOPE_CLAMP, slope))
        # Fitted value at the last read anchors the extrapolation.
        base = intercept + slope * xs[-1]
        resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
        vol = math.sqrt(sum(r * r for r in resid) / len(resid))
        spread = min(SPREAD_CAP,
                     RW_COEF * math.sqrt(max(horizon_days, 0.0))
                     + max(vol, RESID_FLOOR))

    # Saturating trend integral: the trend runs for ~tau days, not forever.
    delta = slope * TAU_DAYS * (1.0 - math.exp(-horizon_days / TAU_DAYS))
    median_logit = base + delta

    # The week-one ceiling caps the central estimate, never the current rate.
    median_rate = min(inv_logit(median_logit), ceiling)
    median_logit = logit(median_rate)
    return quantiles(median_logit, spread)


def quantiles(median_logit, spread):
    """Five quantiles in percent, non-decreasing, rounded to 4 decimals."""
    out = []
    prev = 0.0
    for z in Z:
        q = round(min(100.0, max(0.0, inv_logit(median_logit + z * spread))), 4)
        q = max(q, prev)
        out.append(q)
        prev = q
    return out


def forecast_case(case):
    horizon_days = case["horizon_minutes"] / 1440.0
    trophies = case["trophies"]
    history = case.get("history", [])
    rows = []
    for i, trophy in enumerate(trophies):
        trophy = dict(trophy)
        trophy["index"] = i
        rows.append({"id": trophy["id"],
                     "q": forecast_trophy(trophy, history, horizon_days)})
    rows.sort(key=lambda r: r["id"])
    return {"trophies": rows}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(output_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(input_dir) if n.endswith(".in"))
    for name in names:
        with open(os.path.join(input_dir, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        result = forecast_case(case)
        out_name = os.path.splitext(name)[0] + ".out"
        with open(os.path.join(output_dir, out_name), "w",
                  encoding="utf-8") as f:
            json.dump(result, f, separators=(",", ":"))
            f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
