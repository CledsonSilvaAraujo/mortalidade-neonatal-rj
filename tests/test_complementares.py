import numpy as np
import pandas as pd

from src import complementares


def _calibrado(n=40_000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.beta(0.5, 40, n)
    return (rng.random(n) < p).astype(int), p


def test_calibracao_perfeita_tem_intercepto_zero_e_inclinacao_um():
    y, p = _calibrado()
    c = complementares.intercepto_inclinacao(y, p)
    assert c["intercepto_inf"] < 0 < c["intercepto_sup"]
    assert c["inclinacao_inf"] < 1 < c["inclinacao_sup"]


def test_subestimacao_da_intercepto_positivo():
    y, p = _calibrado()
    c = complementares.intercepto_inclinacao(y, p / 2)
    assert c["intercepto_inf"] > 0


def test_beneficio_liquido():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.6, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    nb = complementares.beneficio_liquido(y, p, limiares=[0.5]).iloc[0]
    assert np.isclose(nb["modelo"], 2 / 10 - 1 / 10 * 1.0)      # VP=2, FP=1, pt/(1-pt)=1
    assert np.isclose(nb["tratar todos"], 0.2 - 0.8 * 1.0)
    assert nb["tratar nenhum"] == 0


def test_taxa_pareamento_por_subgrupo():
    obitos = pd.DataFrame(index=[10, 11, 12, 13])
    pareado = pd.Series([True, False, True, True], index=obitos.index)
    g = {"Residência": pd.Series(["Capital", "Capital", "Interior", "Interior"], index=obitos.index)}
    t = complementares.taxa_pareamento_por(obitos, pareado, g).set_index("Categoria")
    assert t.loc["Capital", "Pareados (%)"] == 50 and t.loc["Interior", "Pareados (%)"] == 100


def test_completude_explica_oe_razao_teste_sobre_treino():
    idx = pd.MultiIndex.from_tuples([("Interior", 2021), ("Interior", 2022), ("Interior", 2023),
                                     ("Capital", 2021), ("Capital", 2022), ("Capital", 2023)], names=["região", "ano"])
    t = pd.DataFrame({"óbitos": [100, 100, 100, 50, 50, 50], "pareados_pct": [80, 90, 94.5, 95, 95, 95]}, index=idx)
    r = complementares.completude_explica_oe(t, ano_teste=2023)
    assert np.isclose(r.loc["Interior", "completude treino (%)"], 85)
    assert np.isclose(r.loc["Interior", "O/E esperado só pelo linkage"], 94.5 / 85)
    assert np.isclose(r.loc["Capital", "O/E esperado só pelo linkage"], 1.0)
