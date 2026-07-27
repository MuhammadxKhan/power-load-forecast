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

# --------------------------------------------------------------------------
# gradient boosting
#
# early_stopping is forced off. The default is "auto", which switches itself ON
# above 10,000 rows and carves an internal 10% validation slice out of whatever
# you hand it. With 25,800 training rows that was silently active, so max_iter
# =600 was really running about 95 iterations and the [300, 600] grid was tuning
# a number the model ignored. Off means max_iter means max_iter, and the
# external validation fold is the only one - which is what the README claimed
# all along.
# --------------------------------------------------------------------------
GBM_LEARNING_RATES = [0.05, 0.1]
MAX_ITERS = [300, 600]


def _make(lr, iters):
    return HistGradientBoostingRegressor(
        learning_rate=lr, max_iter=iters, early_stopping=False, random_state=SEED)


def fit_gbm(Xtr, ytr, Xva, yva, Xfit, yfit, verbose=True):
    best, best_score = None, float("inf")
    for lr in GBM_LEARNING_RATES:
        for it in MAX_ITERS:
            m = _make(lr, it).fit(Xtr, ytr)
            v = mae(yva, predict(m, Xva))
            if verbose:
                print(f"  gbm lr={lr} iters={it:>4}  val MAE {v:8.1f}")
            if v < best_score:
                best, best_score = (lr, it), v

    model = _make(*best).fit(Xfit, yfit)
    info = {"name": "gbm", "params": {"learning_rate": best[0], "max_iter": best[1]},
            "val_mae": best_score, "loss": "squared_error"}
    return (lambda X: predict(model, X)), info
