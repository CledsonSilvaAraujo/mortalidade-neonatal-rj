import pandas as pd

from src import preprocess


def _nv(**kw):
    base = {c: "1" for c in preprocess.CATEGORICAS}
    base.update({c: "10" for c in preprocess.NUMERICAS})
    base.update(DTNASC="01012020", CODMUNRES="330455", obito_neonatal=0, passada_linkage=pd.NA,
                RACACORMAE="4", ESCMAE2010="3", SEMAGESTAC="39")
    return {**base, **kw}


def test_limpar_converte_ignorados_e_categorias():
    nv = pd.DataFrame([_nv(APGAR5="99", PESO="9999", IDADEMAE="32", TPROBSON="05", GRAVIDEZ=None)])
    df = preprocess.limpar(nv)
    assert pd.isna(df.loc[0, "APGAR5"]) and pd.isna(df.loc[0, "PESO"])
    assert df.loc[0, "IDADEMAE"] == 32
    assert df.loc[0, "TPROBSON"] == "5"          # zero à esquerda removido
    assert df.loc[0, "GRAVIDEZ"] == "ausente"
    assert df.loc[0, "capital"] == 1 and df.loc[0, "ano"] == 2020
    assert df.loc[0, "raca_mae"] == "Parda" and df.loc[0, "escol_mae"] == "Médio"


def test_excluir_idade_gestacional_implausivel():
    nv = pd.DataFrame([_nv(SEMAGESTAC="39"), _nv(SEMAGESTAC="12"), _nv(SEMAGESTAC=None),
                       _nv(DTNASC="99999999")])
    df, fluxo = preprocess.excluir(preprocess.limpar(nv))
    assert len(df) == 2
    assert fluxo["n"].tolist() == [4, 3, 2]


def test_conjunto_pre_parto_nao_tem_variaveis_do_parto():
    b = set(preprocess.CONJUNTOS["B_pre_parto"])
    assert not b & {"APGAR1", "APGAR5", "PESO", "IDANOMAL", "PARTO"}
    assert {"SEMAGESTAC", "CONSPRENAT", "IDADEMAE"} <= b
    assert set(preprocess.CONJUNTOS["A_completo"]) >= b


def test_dividir_e_temporal():
    t = preprocess.ANO_TESTE
    df = pd.DataFrame({"ano": [2017, t - 1, t], "obito_neonatal": [0, 1, 0]})
    treino, teste = preprocess.dividir(df)
    assert treino["ano"].max() < teste["ano"].min() == t
