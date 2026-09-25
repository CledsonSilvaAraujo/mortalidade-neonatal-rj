import numpy as np
import pandas as pd

from src import explicabilidade


def _R():
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame({"SEMAGESTAC": rng.normal(38, 2, n), "CONSPRENAT": rng.integers(0, 12, n).astype(float),
                      "IDADEMAE": rng.normal(27, 6, n),
                      "RACACORMAE": pd.Categorical(rng.choice(["1", "2", "4"], n)),
                      "ESCMAE2010": pd.Categorical(rng.choice(["2", "3", "5"], n))})
    valores = np.column_stack([-(X["SEMAGESTAC"] - 38) / 2, rng.normal(0, 0.1, n), rng.normal(0, 0.05, n),
                               np.where(X["RACACORMAE"] == "2", 0.2, -0.05), rng.normal(0, 0.02, n)])
    return {"shap": {"B_pre_parto": {"valores": valores, "X": X}}}


def test_atributos_sensiveis_fracao_e_media_por_categoria():
    out = explicabilidade.atributos_sensiveis(_R(), "B_pre_parto")
    assert 0 < out["fracao_importancia"]["RACACORMAE"] < 1
    assert np.isclose(out["RACACORMAE"].loc["2", "mean"], 0.2)


def test_dependencia_devolve_x_e_shap_das_variaveis_escolhidas():
    dep = explicabilidade.dependencia(_R(), "B_pre_parto")
    assert set(dep) == set(explicabilidade.DEPENDENCIA["B_pre_parto"])
    d = dep["SEMAGESTAC"]
    assert list(d.columns) == ["x", "shap"] and d["shap"].corr(d["x"]) < -0.99


def test_shap_nativo_soma_o_logodds_do_modelo_com_early_stopping():
    from src import models
    from tests.test_models import _dados
    X, y = _dados(n=4000)
    Xv, yv = _dados(n=4000, seed=1)
    m = models.ajustar_xgb(X, y, Xv, yv, {"learning_rate": 0.3, "max_depth": 4}, 1.0)
    assert m.best_iteration + 1 < m.get_booster().num_boosted_rounds()  # há árvores após a melhor
    v, b = explicabilidade.shap_nativo(m, Xv)
    p = m.predict_proba(Xv)[:, 1]
    assert np.abs(v.sum(1) + b - np.log(p / (1 - p))).max() < 1e-4
