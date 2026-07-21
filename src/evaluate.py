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


def yesterday(load):
    return load.shift(24)


def mean_last_4_weeks(load):
    return sum(load.shift(168 * (w + 1)) for w in range(4)) / 4


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mape(a, b):
    return float(np.mean(np.abs((a - b) / a)) * 100)


def bias(pred, y):
    """Mean signed error. Positive means the forecast runs high. MAE hides this
    completely - a forecast can have a fine MAE and still be systematically
    over, which matters if you're buying generation against it."""
    return float(np.mean(pred - y))


def skill(y, pred, base):
    """Fraction of the baseline's error removed. 0 = no better than the baseline."""
    return 1.0 - mae(y, pred) / mae(y, base)


def predict(model, X):
    return pd.Series(model.predict(X), index=X.index)

