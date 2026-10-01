"""
The three models, all with the same interface:

    fit_ridge / fit_gbm / fit_mlp
        (Xtr, ytr, Xva, yva, Xfit, yfit, verbose) -> (predict_fn, info)

Each tries four settings on the validation year, refits the best one on
train+val, and returns a predict function. None of them is passed the test set.

- ridge: the linear reference. Regularised because the lag features are highly
  correlated with each other.
- gradient boosting (HistGradientBoostingRegressor): picks up interactions such
  as hour x weekday x holiday without hand-made features, needs no scaling, and
  is fast on ~35k rows.
- MLP: a plain feed-forward network as a smooth non-linear alternative. Trained
  on MSE, the same loss the booster uses, so a gap between them is about the
  model rather than the loss.
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
# early_stopping is off. sklearn's default ("auto") turns it on above 10,000
# rows and holds back 10% of the training data internally, which meant max_iter
# was being ignored and the grid below wasn't tuning anything. With it off, the
# validation year is the only validation set.
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

# --------------------------------------------------------------------------
# MLP
# --------------------------------------------------------------------------
HIDDEN_SIZES = [(64, 64), (256, 128)]
MLP_LEARNING_RATES = [1e-3, 3e-3]
BATCH = 256
MAX_EPOCHS = 100
PATIENCE = 10


class _Scaler:
    # standardise inputs and target; networks train badly on raw MW values
    def fit(self, a):
        self.mu = a.mean(axis=0)
        self.sd = a.std(axis=0)
        self.sd = np.where(self.sd == 0, 1.0, self.sd)  # constant columns
        return self

    def transform(self, a):
        return (a - self.mu) / self.sd

    def inverse(self, a):
        return a * self.sd + self.mu


def _make_net(n_in, hidden, seed):
    torch.manual_seed(seed)
    layers, prev = [], n_in
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def _predict(net, xs, ys, X):
    net.eval()
    with torch.no_grad():
        z = torch.from_numpy(xs.transform(X.to_numpy(dtype=np.float64)).astype(np.float32))
        out = net(z).numpy().astype(np.float64).ravel()
    return pd.Series(ys.inverse(out), index=X.index)


def _train(Xa, ya, hidden, lr, epochs, Xva=None, yva=None, seed=SEED):
    """Fit on (Xa, ya), with scalers fitted on that data only.

    With a validation set, stop early and return the best epoch. Without one,
    train for exactly `epochs`; that is how the refit reuses the tuned number,
    the same way the booster reuses max_iter.
    """
    xa = Xa.to_numpy(dtype=np.float64)
    yv = ya.to_numpy(dtype=np.float64).reshape(-1, 1)

    xs = _Scaler().fit(xa)
    ys = _Scaler().fit(yv)

    xt = torch.from_numpy(xs.transform(xa).astype(np.float32))
    yt = torch.from_numpy(ys.transform(yv).astype(np.float32))

    net = _make_net(xt.shape[1], hidden, seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    g = torch.Generator().manual_seed(seed)  # fixed batch order, so runs repeat exactly

    best_state, best_val, best_epoch, stale = None, float("inf"), epochs, 0
    n = xt.shape[0]

    for ep in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss_fn(net(xt[idx]), yt[idx]).backward()
            opt.step()

        if Xva is None:
            continue

        # stop on validation MAE, the same metric that picks the other models
        v = mae(yva, _predict(net, xs, ys, Xva))
        if v < best_val:
            best_val, best_epoch, stale = v, ep, 0
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    if best_state is not None:
        net.load_state_dict(best_state)
    return net, xs, ys, best_epoch, best_val


def fit_mlp(Xtr, ytr, Xva, yva, Xfit, yfit, verbose=True):
    best, best_score, best_epochs = None, float("inf"), MAX_EPOCHS
    for hidden in HIDDEN_SIZES:
        for lr in MLP_LEARNING_RATES:
            _, _, _, ep, v = _train(Xtr, ytr, hidden, lr, MAX_EPOCHS, Xva, yva)
            if verbose:
                print(f"  mlp hidden={str(hidden):>10} lr={lr:<6} "
                      f"epochs={ep:>3}  val MAE {v:8.1f}")
            if v < best_score:
                best, best_score, best_epochs = (hidden, lr), v, ep

    hidden, lr = best
    net, xs, ys, _, _ = _train(Xfit, yfit, hidden, lr, best_epochs)

    info = {
        "name": "mlp",
        "params": {"hidden": hidden, "lr": lr, "epochs": best_epochs,
                   "batch": BATCH, "seed": SEED},
        "val_mae": best_score,
        "loss": "mse",
        # returned so selfcheck.py can check the scalers only saw the fit data
        "scaler_x_mean": xs.mu.copy(),
        "scaler_y_mean": float(ys.mu[0]),
        "scaler_y_std": float(ys.sd[0]),
        "scaler_fit_rows": len(Xfit),
    }
    return (lambda X: _predict(net, xs, ys, X)), info


# every model in the comparison; run_comparison.py and selfcheck.py loop over this
ALL_MODELS = [fit_ridge, fit_gbm, fit_mlp]
