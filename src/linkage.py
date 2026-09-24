"""Linkage SINASC × SIM sem identificadores, por chaves compostas em cascata.

Os microdados públicos não trazem o número da DN nem o nome da mãe. Como a
Declaração de Óbito de menores de 1 ano repete dados do nascimento (data,
sexo, peso, idade da mãe, semanas de gestação, tipo de gravidez e de parto),
cada óbito neonatal é pareado a um nascido vivo pela concordância exata
nesses campos. As chaves vão da mais restritiva à mais tolerante a campos
ausentes; em cada passada só são aceitos pares 1:1 (a chave é única entre os
nascidos ainda não pareados e entre os óbitos ainda não pareados).
"""
import numpy as np
import pandas as pd

NEONATAL_MAX_DIAS = 27  # óbito com 0 a 27 dias completos de vida

# DO.CODESTAB é o estabelecimento do óbito; para o neonato costuma ser o mesmo
# hospital do parto (DN.CODESTAB), por isso vale como campo de pareamento.
PASSADAS = [
    ["DTNASC", "SEXO", "CODMUNRES", "PESO", "IDADEMAE", "SEMAGESTAC", "GRAVIDEZ", "PARTO"],
    ["DTNASC", "SEXO", "CODMUNRES", "PESO", "IDADEMAE"],
    ["DTNASC", "SEXO", "CODMUNRES", "PESO", "CODESTAB"],
    ["DTNASC", "SEXO", "PESO", "IDADEMAE"],
    ["DTNASC", "SEXO", "CODMUNRES", "IDADEMAE", "SEMAGESTAC"],
    ["DTNASC", "SEXO", "CODMUNRES", "CODESTAB", "IDADEMAE"],
]
# Duas chaves foram testadas e descartadas por excesso estimado de falsos pares
# (ver `taxa_falsos_pares`): [DTNASC, SEXO, CODMUNRES, PESO] e
# [DTNASC, SEXO, CODMUNRES, CODESTAB]. A última pareia neonatos transferidos com
# outro nascido do hospital onde morreram.
PASSADAS_ESTRITAS = 2  # sensibilidade: só as duas primeiras chaves

# Campos que nunca entram em chave, usados para estimar falsos pares. Um campo que
# já esteve numa chave anterior discorda por construção nas passadas seguintes.
CONFERENCIA = ["ESCMAE", "QTDFILMORT"]

NUMERICAS = {"PESO": (100, 7000), "IDADEMAE": (9, 60), "SEMAGESTAC": (15, 45)}
IGNORADO_9 = ["SEXO", "GRAVIDEZ", "PARTO"]


def _data(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, format="%d%m%Y", errors="coerce")


def chaves(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza os campos de pareamento das duas bases (mesmo tipo e mesmos ausentes)."""
    out = pd.DataFrame(index=df.index)
    out["DTNASC"] = _data(df["DTNASC"])
    for c in ["CODMUNRES", "CODESTAB"]:
        out[c] = df[c].astype("string")
    # O município vem com 6 dígitos nas duas bases, mas registros antigos podem ter 7.
    out["CODMUNRES"] = out["CODMUNRES"].str[:6]
    for c in IGNORADO_9:
        out[c] = df[c].astype("string").mask(df[c].isin(["9", "0"]))
    for c, (lo, hi) in NUMERICAS.items():
        v = pd.to_numeric(df[c], errors="coerce")
        out[c] = v.where(v.between(lo, hi))
    return out


def obitos_neonatais(do: pd.DataFrame, anos_nascimento) -> pd.DataFrame:
    """Óbitos não fetais com 0–27 dias de vida, de nascidos nos anos analisados."""
    do = do[do["TIPOBITO"] == "2"]
    dias = (_data(do["DTOBITO"]) - _data(do["DTNASC"])).dt.days
    ano_nasc = _data(do["DTNASC"]).dt.year
    sel = dias.between(0, NEONATAL_MAX_DIAS) & ano_nasc.isin(list(anos_nascimento))
    return do[sel].assign(DIAS_VIDA=dias[sel])


def _unicos(df: pd.DataFrame, chave: list[str]) -> pd.DataFrame:
    df = df.dropna(subset=chave)
    return df[~df.duplicated(chave, keep=False)]


def parear(nv: pd.DataFrame, obitos: pd.DataFrame, passadas=PASSADAS) -> pd.DataFrame:
    """Devolve os pares (índice do NV, índice do óbito, passada) aceitos em cascata."""
    a, b = chaves(nv), chaves(obitos)
    pares = []
    for i, chave in enumerate(passadas, start=1):
        ua, ub = _unicos(a, chave), _unicos(b, chave)
        m = (ub.rename_axis("idx_do").reset_index()[["idx_do", *chave]]
               .merge(ua.rename_axis("idx_nv").reset_index()[["idx_nv", *chave]], on=chave))
        pares.append(m[["idx_nv", "idx_do"]].assign(passada=i))
        a, b = a.drop(m["idx_nv"]), b.drop(m["idx_do"])
    return pd.concat(pares, ignore_index=True)


def _concorda(x: pd.Series, y: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce").mask(lambda v: v == 9).reset_index(drop=True)
    y = pd.to_numeric(y, errors="coerce").mask(lambda v: v == 9).reset_index(drop=True)
    return (x == y).where(x.notna() & y.notna())


def taxa_falsos_pares(nv: pd.DataFrame, obitos: pd.DataFrame, pares: pd.DataFrame,
                      semente: int = 0) -> pd.DataFrame:
    """Estima a fração de falsos pares de cada passada por um modelo de mistura.

    Para um campo fora das chaves, a concordância observada na passada k é
    c_k = (1 − f_k)·c_verd + f_k·c_acaso, em que c_verd vem da passada 1 (chave
    completa, praticamente sem falsos pares) e c_acaso de pares aleatórios com a
    mesma data de nascimento e sexo. Logo f_k = (c_verd − c_k)/(c_verd − c_acaso).
    """
    nv_p, ob_p = nv.loc[pares["idx_nv"]], obitos.loc[pares["idx_do"]]
    rng = np.random.default_rng(semente)
    grupos = nv.groupby(["DTNASC", "SEXO"]).indices
    sorteio = [rng.choice(grupos[k]) for k in zip(ob_p["DTNASC"], ob_p["SEXO"])]
    linhas = {}
    for c in CONFERENCIA:
        conc = _concorda(nv_p[c], ob_p[c]).groupby(pares["passada"].to_numpy()).mean() * 100
        acaso = _concorda(nv.iloc[sorteio][c], ob_p[c]).mean() * 100
        linhas[f"conc_{c}"] = conc.round(1)
        linhas[f"falsos_{c}"] = (100 * (conc.loc[1] - conc) / (conc.loc[1] - acaso)).clip(lower=0).round(0)
        linhas[f"acaso_{c}"] = round(acaso, 1)
    tab = pd.DataFrame(linhas)
    tab.insert(0, "n_pares", pares["passada"].value_counts().sort_index())
    return tab


def rotular(nv: pd.DataFrame, do: pd.DataFrame, anos_nascimento) -> tuple[pd.DataFrame, dict]:
    """Marca `obito_neonatal` nos nascidos vivos e devolve o relatório de qualidade."""
    obitos = obitos_neonatais(do, anos_nascimento)
    obitos = obitos[~obitos.duplicated()]  # registros repetidos no SIM
    pares = parear(nv, obitos)
    nv = nv.copy()
    nv["obito_neonatal"] = 0
    nv.loc[pares["idx_nv"], "obito_neonatal"] = 1
    nv["passada_linkage"] = pd.NA
    nv.loc[pares["idx_nv"], "passada_linkage"] = pares["passada"].to_numpy()
    nv.loc[pares["idx_nv"], "dias_vida"] = obitos.loc[pares["idx_do"], "DIAS_VIDA"].to_numpy()
    ano_nv = _data(nv["DTNASC"]).dt.year
    ano_ob = _data(obitos["DTNASC"]).dt.year
    por_ano = pd.DataFrame({
        "nascidos_vivos": ano_nv.value_counts(),
        "obitos_neonatais_sim": ano_ob.value_counts(),
        "obitos_pareados": ano_nv[nv["obito_neonatal"] == 1].value_counts(),
    }).sort_index()
    por_ano["pct_pareado"] = (100 * por_ano["obitos_pareados"] / por_ano["obitos_neonatais_sim"]).round(1)
    por_ano["tmn_sim_por_mil"] = (1000 * por_ano["obitos_neonatais_sim"] / por_ano["nascidos_vivos"]).round(2)
    por_ano["tmn_pareada_por_mil"] = (1000 * por_ano["obitos_pareados"] / por_ano["nascidos_vivos"]).round(2)
    relatorio = {
        "por_ano": por_ano,
        "por_passada": pares["passada"].value_counts().sort_index(),
        "falsos_pares": taxa_falsos_pares(nv, obitos, pares),
        "n_obitos": len(obitos),
        "n_pareados": len(pares),
        "nao_pareados": obitos.drop(pares["idx_do"]),
    }
    return nv, relatorio
