"""
The three models. Same features, same split, same protocol, same interface.

Each one exposes exactly:

    fit_ridge / fit_gbm / fit_mlp
        (Xtr, ytr, Xva, yva, Xfit, yfit, verbose) -> (predict_fn, info)

which runs a small four-configuration grid on the validation fold, refits once
on train+val, and hands back something that predicts. They sit in one file so
you can see at a glance that the signatures really are identical - that claim is
the whole point of the comparison, and it's easier to check when they're
adjacent than when they're in three separate files.

None of them is ever handed the test set, so none of them can touch it. That's
structural, not a promise.

Ridge and the MLP both standardise their inputs. Ridge does it inside a
scikit-learn pipeline, so the scaler is fitted on whatever fold the pipeline is
fitted on; the MLP does the same thing by hand in _train. Neither can see the
test fold because neither is given it.

The MLP is deliberately plain - dense layers, ReLU, Adam, early stopping, fixed
seed, no recurrence or attention or schedulers. It's here to be a fair second
data point next to the GBM, not to win, and it isn't tuned until it does. Its
loss is MSE because HistGradientBoostingRegressor minimises squared error; if
they minimised different things, a gap between them would be about the loss
function rather than the model.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .evaluate import mae, predict

SEED = 0


# --------------------------------------------------------------------------
# ridge
# --------------------------------------------------------------------------
ALPHAS = [0.1, 1.0, 10.0, 100.0]


def fit_ridge(Xtr, ytr, Xva, yva, Xfit, yfit, verbose=True):
    best, best_score = 1.0, float("inf")
    for a in ALPHAS:
        m = make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(Xtr, ytr)
        v = mae(yva, predict(m, Xva))
        if verbose:
            print(f"  ridge alpha={a:>6}  val MAE {v:8.1f}")
        if v < best_score:
            best, best_score = a, v

    model = make_pipeline(StandardScaler(), Ridge(alpha=best)).fit(Xfit, yfit)
    info = {"name": "ridge", "params": {"alpha": best},
            "val_mae": best_score, "loss": "squared_error"}
    return (lambda X: predict(model, X)), info
