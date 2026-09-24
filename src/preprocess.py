"""Base analítica: linkage, exclusões, limpeza de códigos, variáveis derivadas e divisão temporal."""
from pathlib import Path

import numpy as np
import pandas as pd

from src import download, linkage

MUNICIPIO_RIO = "330455"
ANO_TESTE = 2023      # validação temporal: último ano de nascimentos
ANO_VALIDACAO = 2022  # holdout temporal para ajustar hiperparâmetros (dentro do treino)

# Numéricas e o código de "ignorado" de cada uma (valores legítimos variam por campo).
NUMERICAS = {
    "IDADEMAE": 99, "IDADEPAI": 99, "QTDFILVIVO": 99, "QTDFILMORT": 99, "QTDGESTANT": 99,
    "QTDPARTNOR": 99, "QTDPARTCES": 99, "SEMAGESTAC": 99, "CONSPRENAT": 99, "MESPRENAT": 99,
    "APGAR1": 99, "APGAR5": 99, "PESO": 9999,
}
# Categóricas: o "9" (ignorado) vira uma categoria própria, porque o não
# preenchimento da DN costuma estar ligado à qualidade da assistência.
CATEGORICAS = [
    "LOCNASC", "ESTCIVMAE", "ESCMAE2010", "GRAVIDEZ", "PARTO", "SEXO", "RACACORMAE",
    "IDANOMAL", "TPAPRESENT", "STTRABPART", "STCESPARTO", "TPNASCASSI", "TPROBSON", "KOTELCHUCK",
]

# (A) Completo: todas as variáveis da DN, como no artigo de referência.
VARS_COMPLETO = [*NUMERICAS, *CATEGORICAS, "capital"]
# (B) Admissão para o parto: o que se sabe antes do nascimento. Saem Apgar, peso,
# anomalia congênita, sexo, local e tipo de parto, indução/cesárea antes do
# trabalho de parto, quem assistiu e o grupo de Robson (que usa o início do trabalho de parto).
POS_PARTO = ["APGAR1", "APGAR5", "PESO", "IDANOMAL", "SEXO", "LOCNASC", "PARTO",
             "STTRABPART", "STCESPARTO", "TPNASCASSI", "TPROBSON"]
VARS_PRE_PARTO = [v for v in VARS_COMPLETO if v not in POS_PARTO]

CONJUNTOS = {"A_completo": VARS_COMPLETO, "B_pre_parto": VARS_PRE_PARTO}

ROTULOS = {
    "IDADEMAE": "Idade da mãe", "IDADEPAI": "Idade do pai", "QTDFILVIVO": "Filhos vivos",
    "QTDFILMORT": "Perdas fetais/abortos", "QTDGESTANT": "Gestações anteriores",
    "QTDPARTNOR": "Partos vaginais anteriores", "QTDPARTCES": "Cesáreas anteriores",
    "SEMAGESTAC": "Semanas de gestação", "CONSPRENAT": "Consultas de pré-natal",
    "MESPRENAT": "Mês de início do pré-natal", "APGAR1": "Apgar 1º min", "APGAR5": "Apgar 5º min",
    "PESO": "Peso ao nascer", "LOCNASC": "Local do nascimento", "ESTCIVMAE": "Estado civil da mãe",
    "ESCMAE2010": "Escolaridade da mãe", "GRAVIDEZ": "Tipo de gravidez", "PARTO": "Tipo de parto",
    "SEXO": "Sexo", "RACACORMAE": "Raça/cor da mãe", "IDANOMAL": "Anomalia congênita",
    "TPAPRESENT": "Apresentação", "STTRABPART": "Trabalho de parto induzido",
    "STCESPARTO": "Cesárea antes do trabalho de parto", "TPNASCASSI": "Quem assistiu o parto",
    "TPROBSON": "Grupo de Robson", "KOTELCHUCK": "Adequação do pré-natal (Kotelchuck)",
    "capital": "Residente na capital",
}


def limpar(nv: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(index=nv.index)
    for c, ignorado in NUMERICAS.items():
        v = pd.to_numeric(nv[c], errors="coerce")
        df[c] = v.mask(v == ignorado).astype("float32")
    for c in CATEGORICAS:
        s = nv[c].astype("string").str.lstrip("0").replace("", "0")
        df[c] = s.fillna("ausente").astype("category")
    df["capital"] = (nv["CODMUNRES"].str[:6] == MUNICIPIO_RIO).astype("float32")
    df["ano"] = pd.to_datetime(nv["DTNASC"], format="%d%m%Y", errors="coerce").dt.year
    df["obito_neonatal"] = nv["obito_neonatal"].astype("int8")
    df["passada_linkage"] = nv["passada_linkage"]
    # Variáveis descritivas (fora dos modelos), usadas nas análises de subgrupo.
    df["raca_mae"] = nv["RACACORMAE"].map({"1": "Branca", "2": "Preta", "3": "Amarela",
                                            "4": "Parda", "5": "Indígena"}).fillna("Ignorada")
    df["escol_mae"] = pd.to_numeric(nv["ESCMAE2010"], errors="coerce").map(
        {0: "Fund. incompleto ou menos", 1: "Fund. incompleto ou menos",
         2: "Fund. incompleto ou menos", 3: "Médio", 4: "Superior",
         5: "Superior"}).fillna("Ignorada")
    return df


def excluir(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exclusões do artigo de referência; devolve também o fluxo (n por etapa)."""
    fluxo = [("Nascidos vivos residentes no RJ, 2017–2023", len(df))]
    df = df[df["ano"].notna()]
    fluxo.append(("Com data de nascimento válida", len(df)))
    ig = df["SEMAGESTAC"]
    df = df[ig.isna() | ig.between(15, 45)]
    fluxo.append(("Idade gestacional entre 15 e 45 semanas (ou ausente)", len(df)))
    return df, pd.DataFrame(fluxo, columns=["Etapa", "n"])


def construir(pasta: Path = download.RAW, cache: Path = download.DADOS / "base_analitica.parquet"):
    """Base analítica completa + relatório do linkage (cacheado em parquet)."""
    nv = download.carregar("DN", download.ANOS_NASCIMENTO, pasta)
    do = download.carregar("DO", download.ANOS_OBITO, pasta)
    nv, relatorio = linkage.rotular(nv, do, download.ANOS_NASCIMENTO)
    base, fluxo = excluir(limpar(nv))
    base.to_parquet(cache)
    return base, fluxo, relatorio


def dividir(df: pd.DataFrame, ano_teste: int = ANO_TESTE):
    treino, teste = df[df["ano"] < ano_teste], df[df["ano"] == ano_teste]
    return treino, teste


def xy(df: pd.DataFrame, conjunto: str):
    return df[CONJUNTOS[conjunto]], df["obito_neonatal"].to_numpy()
