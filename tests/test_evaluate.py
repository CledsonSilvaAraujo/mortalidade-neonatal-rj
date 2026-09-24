import numpy as np
import pandas as pd

from src import evaluate


def _yp(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.05).astype(int)
    p = np.clip(0.05 + 0.3 * y + rng.normal(0, 0.1, n), 0, 1)
    return y, p


def test_metricas_perfeitas():
    y = np.array([0, 0, 0, 1])
    m = evaluate.metricas(y, np.array([0.1, 0.2, 0.3, 0.9]), limiar=0.5)
    assert m["AUROC"] == 1 and m["F1"] == 1 and m["Sensibilidade"] == 1 and m["Especificidade"] == 1


def test_ic_contem_ponto_e_bootstrap_mantem_obitos():
    y, p = _yp()
    tab = evaluate.com_ic(y, p, limiar=0.2, n=50)
    assert (tab["ic_inf"] <= tab["valor"]).all() and (tab["valor"] <= tab["ic_sup"]).all()
    idx = next(evaluate._reamostras(y, 1, 0))
    assert y[idx].sum() == y.sum()


def test_delta_pareado_de_modelo_com_ele_mesmo_e_zero():
    y, p = _yp()
    d = evaluate.delta_pareado(y, p, p, n=20)
    assert d["delta"] == 0 and d["ic_inf"] == 0 and d["ic_sup"] == 0


def test_limiar_max_f1_e_captura_exata_com_empates():
    y, p = _yp()
    assert 0 < evaluate.limiar_max_f1(y, p) < 1
    # Escore em degraus (como a isotônica): o top 5% tem exatamente 5% dos registros.
    y2 = np.array([1, 0, 0, 0] * 5)
    assert evaluate.captura_top(y2, np.ones(20), fracao=0.25) == 0.4  # 5 primeiros: 2 dos 5 óbitos


def test_metricas_usam_escore_para_ordenar():
    y = np.array([0, 0, 1, 1])
    p_degrau = np.array([0.2, 0.5, 0.5, 0.5])  # empata negativo e positivos
    escore = np.array([0.1, 0.2, 0.3, 0.4])
    assert evaluate.metricas(y, p_degrau, 0.5, escore)["AUROC"] == 1.0
    assert evaluate.metricas(y, p_degrau, 0.5)["AUROC"] < 1.0


def test_calibracao_e_subgrupos():
    y, p = _yp()
    cal = evaluate.calibracao(y, p)
    assert len(cal) == 10 and cal["n"].sum() == len(y)
    df = pd.DataFrame({"y": y, "p": p, "g": np.where(np.arange(len(y)) % 2, "a", "b")})
    sub = evaluate.por_subgrupo(df, "g")
    assert set(sub["g"]) == {"a", "b"}
    com_ic = evaluate.por_subgrupo(df, "g", n_bootstrap=30)
    assert (com_ic["AUROC_inf"] <= com_ic["AUROC"]).all() and (com_ic["AUROC"] <= com_ic["AUROC_sup"]).all()
