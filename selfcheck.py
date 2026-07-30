"""
Self-checks on synthetic data. No download, no network.

    python selfcheck.py

The numbers these print are meaningless - the data is fake. What matters is
that the assertions hold: nothing reaches forward in time, the calendar features
are on German clocks, weather behaves the way the chosen mode says it does, the
MLP's scalers never see the test period, and every model is scored on the same
rows.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.data import _from_netcdf, fake_frame, fake_temperature
from src.evaluate import (assert_same_rows, baseline_preds, mae,
                          mae_by_target_hour, predict, seasonal_naive, skill)
from src.features import (build_features, chronological_split, degree_hours,
                          usable_temperature)
from src.models import ALL_MODELS, fit_mlp

VAL_START, TEST_START = "2016-09-01", "2016-11-01"


def _poke(series, pos, amount):
    out = series.copy()
    out.iloc[pos] += amount
    return out, out.index[pos]


def _changed_rows(before, after):
    common = before.index.intersection(after.index)
    return common[(before.loc[common] != after.loc[common]).any(axis=1)]


def check_no_load_leakage(load):
    # 1) no feature may react to a load change less than 24h before it.
    # Note the >= t: a feature that used the value AT t would be the worst leak
    # of all, and the old version of this test let it through by only looking
    # strictly after t.
    Xb, _ = build_features(load)
    poked, t = _poke(load, 3000, 50000)
    Xa, _ = build_features(poked)

    changed = _changed_rows(Xb, Xa)
    too_soon = changed[(changed >= t) & (changed < t + pd.Timedelta("24h"))]
    assert len(too_soon) == 0, f"LEAKAGE: features reacted within 24h at {list(too_soon)[:3]}"
    assert (changed >= t + pd.Timedelta("24h")).any(), "lags look broken - nothing reacted"
    print("  [ok] no feature uses load data newer than 24h")

