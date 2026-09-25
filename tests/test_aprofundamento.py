import numpy as np
import pandas as pd

from src import aprofundamento as ap
from src import linkage


def _nv(n=400, seed=0):
    """DN sintéticas com os campos de chave e de conferência, em texto como no DATASUS."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "DTNASC": rng.choice(["01012020", "02012020", "03012020"], n),
        "SEXO": rng.choice(["1", "2"], n),
        "CODMUNRES": rng.choice(["330455", "330330", "330490"], n),
        "PESO": pd.Series(rng.integers(500, 4500, n)).astype(str).str.zfill(4).to_numpy(),
        "IDADEMAE": pd.Series(rng.integers(15, 45, n)).astype(str).to_numpy(),
        "SEMAGESTAC": pd.Series(rng.integers(24, 42, n)).astype(str).to_numpy(),
        "GRAVIDEZ": "1", "PARTO": rng.choice(["1", "2"], n),
        "CODESTAB": rng.choice(["111", "222", "333", "444"], n),
        "ESCMAE": rng.choice(["1", "2", "3", "4", "5"], n),
        "QTDFILMORT": rng.choice(["0", "1", "2"], n),
    }, index=pd.RangeIndex(1000, 1000 + n))


def test_simulacao_sem_ruido_pareia_tudo_corretamente():
    nv = _nv()
    verdadeiros = nv.sample(40, random_state=1)
    do = ap.simular_obitos(verdadeiros, ausencia={}, erro={})
    pares = linkage.parear(nv, do)
    tab = ap.avaliar_pareamento(pares, do)
    assert tab.loc["total", "falsos_reais"] == 0
    assert set(do.columns) >= set(ap.CAMPOS_DO) | {"idx_nv_verdadeiro"}


def test_simulacao_com_ausencia_total_do_peso_esvazia_chaves_com_peso():
    nv = _nv()
    do = ap.simular_obitos(nv.sample(40, random_state=2), ausencia={"PESO": 1.0}, erro={})
    assert do["PESO"].isna().all()
    pares = linkage.parear(nv, do)
    assert not pares["passada"].isin([1, 2, 3, 4]).any()  # as quatro primeiras chaves exigem o peso


def test_ruido_numerico_mantem_formato():
    nv = _nv()
    do = ap.simular_obitos(nv, ausencia={}, erro={"PESO": 1.0, "SEMAGESTAC": 1.0})
    diff = (pd.to_numeric(do["PESO"]) - pd.to_numeric(nv["PESO"].reset_index(drop=True))).abs()
    assert diff.between(10, 100).all()


def test_comparar_niveis():
    a = pd.DataFrame({"CODMUNRES": ["1", "1", None], "PESO": [1000.0, 1000.0, 1000.0],
                      "IDADEMAE": [20.0, 20.0, np.nan], "SEMAGESTAC": [30.0, 31.0, 30.0],
                      "GRAVIDEZ": ["1", "1", "1"], "PARTO": ["1", "2", "1"], "CODESTAB": ["9", "9", "9"]})
    b = pd.DataFrame({"CODMUNRES": ["1", "2", "1"], "PESO": [1000.0, 1040.0, 2000.0],
                      "IDADEMAE": [20.0, 22.0, 20.0], "SEMAGESTAC": [30.0, 30.0, 30.0],
                      "GRAVIDEZ": ["1", "1", "1"], "PARTO": ["1", "1", "1"], "CODESTAB": ["9", "8", "9"]})
    niv = ap.comparar(a, b)
    assert list(niv["PESO"]) == [2, 1, 0]
    assert np.isnan(niv.loc[2, "CODMUNRES"]) and np.isnan(niv.loc[2, "IDADEMAE"])
    assert list(niv["IDADEMAE"][:2]) == [2, 0]
    assert list(niv["PARTO"]) == [2, 0, 2]


def test_em_separa_pares_verdadeiros():
    rng = np.random.default_rng(0)
    n_true, n_false = 300, 30_000
    campos = ["A", "B", "C"]
    verd = pd.DataFrame({c: rng.choice([2, 1, 0], n_true, p=[0.9, 0.05, 0.05]) for c in campos})
    fals = pd.DataFrame({c: rng.choice([2, 1, 0], n_false, p=[0.05, 0.05, 0.9]) for c in campos})
    niv = pd.concat([verd, fals], ignore_index=True).astype(float)
    u = {c: {0: 0.9, 1: 0.05, 2: 0.05} for c in campos}
    par = ap.em_fellegi_sunter(niv, u)
    assert par["m"]["A"][2] > 0.8
    assert abs(par["lambda"] - n_true / (n_true + n_false)) < 0.005
    _, post = ap.pesos(niv, par)
    assert post[:n_true].mean() > 0.8 > post[n_true:].mean()


def test_etapa_probabilistica_recupera_par_com_erro():
    nv = _nv(seed=3)
    verdadeiros = nv.sample(60, random_state=3)
    do = ap.simular_obitos(verdadeiros, ausencia={}, erro={"PESO": 1.0, "IDADEMAE": 1.0, "SEMAGESTAC": 1.0})
    det = linkage.parear(nv, do)
    fs = ap.etapa_probabilistica(nv, do, det, limiar_post=0.5)
    if len(fs["pares"]):
        certo = do.loc[fs["pares"]["idx_do"], "idx_nv_verdadeiro"].to_numpy() == fs["pares"]["idx_nv"].to_numpy()
        assert certo.mean() > 0.5
    assert fs["pares"]["idx_do"].is_unique and fs["pares"]["idx_nv"].is_unique


def test_particao_temporal():
    base = pd.DataFrame({"ano": [2018, 2019, 2020, 2021, 2022], "obito_neonatal": 0})
    aj, va, te = ap._particao(base, 2021)
    assert list(aj["ano"]) == [2018, 2019] and list(va["ano"]) == [2020] and list(te["ano"]) == [2021]


def test_m_de_pares_usa_passadas_em_que_o_campo_fica_fora_da_chave():
    nv = _nv(seed=5)
    do = ap.simular_obitos(nv.sample(80, random_state=5), ausencia={}, erro={})
    pares = linkage.parear(nv, do)
    m = ap.m_de_pares(linkage.chaves(nv), linkage.chaves(do), pares)
    assert m["CODESTAB"]["passadas"] == [1, 2]      # o estabelecimento não entra nas duas primeiras chaves
    assert m["PESO"]["passadas"] == [5, 6]          # o peso está nas quatro primeiras
    assert m["CODESTAB"][2] > 0.99                  # sem ruído, os pares verdadeiros concordam


def test_declaracao_pos_obito_separa_por_data():
    teste = pd.DataFrame({"obito_neonatal": [1, 1, 0, 0, 0, 0]}, index=[10, 11, 12, 13, 14, 15])
    nv_rot = pd.DataFrame({"DTNASC": ["01012024"] * 6, "dias_vida": [0, 3, np.nan, np.nan, np.nan, np.nan],
                           "DTDECLARAC": ["02012024", "01012024", "01012024", "01012024", "01012024", "01012024"],
                           "DTCADASTRO": ["10012024"] * 6}, index=teste.index)
    R = {"predicoes": {(c, "XGBoost"): {"bruto": np.array([.9, .8, .1, .2, .3, .05])}
                       for c in ["A_completo", "B_pre_parto"]}}
    t = ap.declaracao_pos_obito(nv_rot, teste, R, n_min=100)
    d = t[(t["Campo"] == "declarada") & (t["Conjunto"] == "A_completo") & (t["Faixa"] == "todas")].set_index("DN")
    assert d.loc["depois do óbito", "Óbitos"] == 1 and d.loc["antes ou no dia", "Óbitos"] == 1
    g = t[(t["Campo"] == "digitada") & (t["Conjunto"] == "A_completo") & (t["Faixa"] == "todas")].set_index("DN")
    assert g.loc["depois do óbito", "Óbitos"] == 2
