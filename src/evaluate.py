"""
Baselines, metrics, the scoring table, and the rolling-origin backtest.

Everything that turns predictions into numbers lives here, so every model is
scored by exactly the same code on exactly the same rows.

The baselines now include a real published benchmark. OPSD ships a day-ahead
load forecast column derived from ENTSO-E Transparency data, in the same file as
the demand figures. Seasonal naive tells you whether the model learned anything;
the published benchmark tells you whether the answer is in the right ballpark.

Be careful how that benchmark is described. It is OPSD's aggregation, not a raw
untouched TSO series, and the file keeps target timestamps but no forecast
vintage - so there is no way to know whether a value is the first issuance or a
later revision. Its information cutoff is earlier than this model's assumed
midnight either way, so beating it is not a like-for-like win.
"""

import numpy as np
import pandas as pd

TZ = "Europe/Berlin"


# --------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------
def seasonal_naive(load):
    """Same hour, same weekday, last week. The one to beat first.

    168 rows back is 168 UTC hours, which is the same LOCAL clock hour on every
    week except the two containing a daylight-saving change - there it lands an
    hour out. Two weeks a year, left as is because changing it would move the
    published baseline, but worth knowing it is not exactly "same local hour".
    """
    return load.shift(168)

