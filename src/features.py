"""
Features and the train/val/test split.

This is the only place either of those is defined. Every model imports from
here, so no model can train on a different feature set or a different split
than the one it's compared against.

Two rules everything hangs on.

1. The load rule. We forecast day D at midnight, so the newest demand figure we
   have is 23:00 on day D-1. No feature touches the load series at a lag under
   24 hours. selfcheck.py pokes the series and checks nothing reacts too soon.

2. Weather is different, and this is the bit worth being careful about. You
   genuinely do have a weather forecast for tomorrow when you make a load
   forecast - that is how it works in practice. So temperature at the target
   hour is not automatically cheating. What IS cheating is using ERA5's
   after-the-fact reanalysis and pretending it was a forecast, because a real
   forecast has error in it. weather_mode below makes the choice explicit
   instead of burying it:

     "lagged"   only temperature from 24h+ ago. Nothing to argue about.
     "noisy"    target-hour temperature with synthetic error added. A
                SENSITIVITY TEST, not a forecast - see below.
     "perfect"  target-hour temperature exactly. Perfect prognosis - an upper
                bound on what weather can buy you, not a deployable result.
                The gap between "perfect" and "noisy" is the cost of the
                imposed error model, NOT the measured cost of real forecast
                error. Only an archived forecast answers that.
     "none"     no weather at all, i.e. the old model.

Everything calendar-related is computed in Europe/Berlin, not UTC. The index
stays UTC because that has no daylight-saving ambiguity, but German demand
follows German clocks: 23:00 UTC on 31 December is already New Year's Day in
Germany. Getting this wrong mislabels the hour on every single row and the
weekday on about 7% of them.
"""

import numpy as np
import pandas as pd

TZ = "Europe/Berlin"

# Size of the synthetic error injected by weather_mode="noisy", degrees C.
#
# Be clear about what this is NOT. It is not a weather forecast. It starts from
# ERA5 truth and adds independent Gaussian noise, so it assumes the error is
# unbiased, uncorrelated hour to hour, the same at every lead time, the same in
# every season, and the same everywhere in the country. Real forecast error is
# none of those things. Using an archived operational forecast is the only way
# to answer this properly, and I haven't got one.
#
# So this measures the cost of temperature error UNDER THIS IMPOSED ERROR MODEL,
# and nothing more. 1.0 is a plausible order of magnitude for day-ahead 2m
# temperature, not a measured value for any particular model or region.
def _cyclical(values, period):
    r = 2 * np.pi * values / period
    return np.sin(r), np.cos(r)

