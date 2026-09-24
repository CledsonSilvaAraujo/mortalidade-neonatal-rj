import pandas as pd

from src import linkage


def _nv(**kw):
    base = dict(DTNASC="01012020", SEXO="1", CODMUNRES="330455", CODESTAB="111", PESO="3000",
                IDADEMAE="25", SEMAGESTAC="39", GRAVIDEZ="1", PARTO="1",
                ESCMAE="4", QTDFILMORT="0")
    return {**base, **kw}


def _do(dtobito="05012020", **kw):
    return {**_nv(**kw), "TIPOBITO": "2", "DTOBITO": dtobito}


def test_obitos_neonatais_filtra_idade_tipo_e_ano():
    do = pd.DataFrame([
        _do(),                                        # 4 dias: entra
        _do(dtobito="29012020"),                      # 28 dias: pós-neonatal
        {**_do(), "TIPOBITO": "1"},                   # óbito fetal
        _do(DTNASC="01012016", dtobito="02012016"),   # nascido fora do período
    ])
    neo = linkage.obitos_neonatais(do, [2020])
    assert list(neo.index) == [0]
    assert neo["DIAS_VIDA"].tolist() == [4]


def test_chaves_padroniza_ausentes_e_numeros():
    k = linkage.chaves(pd.DataFrame([_nv(PESO="0665", GRAVIDEZ="9", IDADEMAE="99", CODMUNRES="3304557")]))
    assert k.loc[0, "PESO"] == 665
    assert pd.isna(k.loc[0, "GRAVIDEZ"]) and pd.isna(k.loc[0, "IDADEMAE"])
    assert k.loc[0, "CODMUNRES"] == "330455"


def test_parear_exige_unicidade_e_usa_passadas_tolerantes():
    nv = pd.DataFrame([
        _nv(PESO="3000"),                 # 0: par exato do óbito 0
        _nv(PESO="2500", IDADEMAE="30"),  # 1: gêmeos idênticos na chave → ambíguos
        _nv(PESO="2500", IDADEMAE="30"),  # 2
        _nv(PESO="1800", IDADEMAE="19"),  # 3: óbito 2 sem semanas de gestação
    ])
    do = pd.DataFrame([
        _do(PESO="3000"),
        _do(PESO="2500", IDADEMAE="30"),
        _do(PESO="1800", IDADEMAE="19", SEMAGESTAC=None),
    ])
    pares = linkage.parear(nv, do)
    got = dict(zip(pares["idx_do"], pares["idx_nv"]))
    assert got[0] == 0 and got[2] == 3
    assert 1 not in got  # ambíguo: não pareia
    assert pares.set_index("idx_do").loc[2, "passada"] > 1


def test_rotular_marca_desfecho_e_relatorio():
    nv = pd.DataFrame([_nv(), _nv(PESO="3200", IDADEMAE="31")])
    do = pd.DataFrame([_do(), _do()])  # óbito duplicado no SIM conta uma vez
    rot, rel = linkage.rotular(nv, do, [2020])
    assert rot["obito_neonatal"].tolist() == [1, 0]
    assert rel["n_obitos"] == 1 and rel["n_pareados"] == 1
    assert rel["por_ano"].loc[2020, "pct_pareado"] == 100.0
