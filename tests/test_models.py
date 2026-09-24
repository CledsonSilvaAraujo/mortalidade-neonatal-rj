import numpy as np
import pandas as pd

from src import models


def _dados(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"peso": rng.normal(3000, 600, n).astype("float32"),
                      "cat": pd.Categorical(rng.choice(["1", "2", "ausente"], n))})
    X.loc[::17, "peso"] = np.nan
    logit = -4 - (X["peso"].fillna(3000) - 3000) / 300 + (X["cat"] == "2") * 1.0
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


def test_lr_lida_com_ausentes_e_categorias():
    X, y = _dados()
    p = models.ajustar_lr(X, y).predict_proba(X)[:, 1]
    assert p.shape == (len(X),) and np.isfinite(p).all()


def test_busca_xgb_e_recalibracao():
    X, y = _dados()
    Xv, yv = _dados(seed=1)
    estudo = models.buscar_xgb(X, y, Xv, yv, n_tentativas=2)
    params, spw = models.params_do_estudo(estudo, y)
    assert "peso_classe" not in params and spw >= 1
    m = models.ajustar_xgb(X, y, Xv, yv, params, spw, curvas_treino=True)
    curvas = models.curvas_iteracao(m)
    assert {"treino_logloss", "valid_aucpr"} <= set(curvas.columns)
    cal = models.Recalibrado(m, Xv, yv)
    p = cal.predict_proba(Xv)[:, 1]
    assert 0 <= p.min() and p.max() <= 1
    assert abs(p.mean() - yv.mean()) < 0.01  # isotônica reproduz a prevalência onde foi ajustada
