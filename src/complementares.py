"""Análises complementares: viés do linkage, sensibilidade, calibração, utilidade clínica e premissas.

Usam os hiperparâmetros já escolhidos no experimento principal, então podem ser
refeitas sem repetir a busca: `python -m src.experimento --complementares`.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src import download, evaluate, linkage, models, preprocess

MUNICIPIO_RIO = preprocess.MUNICIPIO_RIO
LIMIARES_DECISAO = np.round(np.arange(0.005, 0.301, 0.005), 3)


# ------------------------------------------------------------------ viés do linkage
def perfil_obitos(obitos: pd.DataFrame) -> dict:
    """Características dos óbitos a partir da própria Declaração de Óbito (sem depender do pareamento)."""
    k = linkage.chaves(obitos)
    dias = obitos["DIAS_VIDA"]
    return {
        "n": len(obitos),
        "Peso ausente (%)": 100 * k["PESO"].isna().mean(),
        "Peso < 1.500 g (%)": 100 * (k["PESO"] < 1500).sum() / max(k["PESO"].notna().sum(), 1),
        "Peso mediano (g)": k["PESO"].median(),
        "Semanas ausentes (%)": 100 * k["SEMAGESTAC"].isna().mean(),
        "Semanas < 32 (%)": 100 * (k["SEMAGESTAC"] < 32).sum() / max(k["SEMAGESTAC"].notna().sum(), 1),
        "Idade da mãe ausente (%)": 100 * k["IDADEMAE"].isna().mean(),
        "Óbito no 1º dia (%)": 100 * (dias == 0).mean(),
        "Óbito com 7–27 dias (%)": 100 * (dias >= 7).mean(),
        "Residente na capital (%)": 100 * (obitos["CODMUNRES"].str[:6] == MUNICIPIO_RIO).mean(),
        "Óbito em hospital (%)": 100 * (obitos["LOCOCOR"] == "1").mean(),
    }


def taxa_pareamento_por(obitos: pd.DataFrame, pareado: pd.Series, grupos: dict) -> pd.DataFrame:
    """% de óbitos pareados por subgrupo definido na DO."""
    linhas = []
    for nome, serie in grupos.items():
        for cat, idx in serie.groupby(serie).groups.items():
            linhas.append({"Variável": nome, "Categoria": cat, "Óbitos": len(idx),
                           "Pareados (%)": 100 * pareado.loc[idx].mean()})
    return pd.DataFrame(linhas)


def vies_linkage(nv: pd.DataFrame, do: pd.DataFrame) -> dict:
    """Compara óbitos pareados e não pareados e a taxa de pareamento por subgrupo."""
    obitos = linkage.obitos_neonatais(do, download.ANOS_NASCIMENTO)
    obitos = obitos[~obitos.duplicated()]
    pares = linkage.parear(nv, obitos)
    pareado = pd.Series(obitos.index.isin(pares["idx_do"]), index=obitos.index)
    perfil = pd.DataFrame({"Pareados": perfil_obitos(obitos[pareado]),
                           "Não pareados": perfil_obitos(obitos[~pareado])})
    k = linkage.chaves(obitos)
    raca = obitos["RACACOR"].map({"1": "Branca", "2": "Preta", "4": "Parda"}).fillna("Outra/ignorada")
    grupos = {
        "Residência": pd.Series(np.where(obitos["CODMUNRES"].str[:6] == MUNICIPIO_RIO, "Capital", "Interior"),
                                index=obitos.index),
        "Raça/cor do recém-nascido (DO)": raca,
        "Peso na DO": pd.Series(np.where(k["PESO"].isna(), "ausente", "preenchido"), index=obitos.index),
    }
    return {"perfil": perfil, "por_subgrupo": taxa_pareamento_por(obitos, pareado, grupos)}


# ------------------------------------------------------------------ sensibilidade
def sensibilidade_linkage(ajuste, valid, teste, R) -> dict:
    """Refaz o XGBoost com rótulos alternativos no treino e avalia no MESMO teste de 2023.

    - "estrito": os nascimentos pareados nas passadas 3–6 saem do ajuste e da validação;
    - "pior caso": esses nascimentos ficam, mas rotulados como sobreviventes (0).
    O teste é o completo, com os rótulos principais, para que as métricas sejam comparáveis
    (inclusive por bootstrap pareado) às do modelo principal.
    """
    tolerante = lambda df: df["passada_linkage"].notna() & (df["passada_linkage"] > linkage.PASSADAS_ESTRITAS)
    cenarios = {
        "estrito": lambda df: df[~tolerante(df)],
        "pior caso": lambda df: df.assign(obito_neonatal=np.where(tolerante(df), 0, df["obito_neonatal"])),
    }
    y_te = teste["obito_neonatal"].to_numpy()
    saida = {}
    for conj in preprocess.CONJUNTOS:
        params = {k: v for k, v in R["params"][(conj, "XGBoost")].items() if k not in ("scale_pos_weight", "n_arvores")}
        spw = R["params"][(conj, "XGBoost")]["scale_pos_weight"]
        X_te, _ = preprocess.xy(teste, conj)
        base_escore = R["predicoes"][(conj, "XGBoost")]["bruto"]
        for nome, f in cenarios.items():
            (X_aj, y_aj), (X_va, y_va) = preprocess.xy(f(ajuste), conj), preprocess.xy(f(valid), conj)
            cal = models.Recalibrado(models.ajustar_xgb(X_aj, y_aj, X_va, y_va, params, spw), X_va, y_va)
            limiar = evaluate.limiar_max_f1(y_va, cal.predict_proba(X_va)[:, 1])
            escore = cal.bruto(X_te)
            saida[(conj, nome)] = {
                "metricas": evaluate.com_ic(y_te, cal.predict_proba(X_te)[:, 1], limiar, escore=escore),
                "delta_auroc": evaluate.delta_pareado(y_te, escore, base_escore),
                "positivos_treino": int(y_aj.sum()),
            }
    return saida


# ------------------------------------------------------------------ calibração
def intercepto_inclinacao(y, p) -> dict:
    """Calibração fraca (Van Calster et al., 2019).

    Inclinação: coeficiente de logit(p) em y ~ logit(p) (ideal 1; < 1 indica previsões extremas demais).
    Intercepto (calibration-in-the-large): de y ~ 1 com offset logit(p) (ideal 0; > 0 indica subestimação).
    """
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    inc = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit()
    itc = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit()
    ci_i, ci_s = inc.conf_int()[1], itc.conf_int()[0]
    return {"intercepto": itc.params[0], "intercepto_inf": ci_s[0], "intercepto_sup": ci_s[1],
            "inclinacao": inc.params[1], "inclinacao_inf": ci_i[0], "inclinacao_sup": ci_i[1]}


# ------------------------------------------------------------------ utilidade clínica
def beneficio_liquido(y, p, limiares=LIMIARES_DECISAO) -> pd.DataFrame:
    """Curva de decisão (Vickers e Elkin, 2006): NB = VP/n − FP/n · pt/(1 − pt)."""
    y, p = np.asarray(y), np.asarray(p)
    n, prev = len(y), y.mean()
    linhas = []
    for pt in limiares:
        pred = p >= pt
        vp, fp = (pred & (y == 1)).sum(), (pred & (y == 0)).sum()
        peso = pt / (1 - pt)
        linhas.append({"limiar": pt, "modelo": vp / n - fp / n * peso,
                       "tratar todos": prev - (1 - prev) * peso, "tratar nenhum": 0.0,
                       "fração sinalizada": pred.mean()})
    return pd.DataFrame(linhas)


# ------------------------------------------------------------------ premissas
def colinearidade(df: pd.DataFrame, n_amostra: int = 200_000, semente: int = 42) -> pd.DataFrame:
    """VIF das variáveis numéricas (após imputação pela mediana) numa amostra do ajuste."""
    X = df[list(preprocess.NUMERICAS)].sample(min(n_amostra, len(df)), random_state=semente)
    X = X.fillna(X.median()).astype(float)
    Xc = sm.add_constant(X)
    vif = [variance_inflation_factor(Xc.values, i + 1) for i in range(X.shape[1])]
    return pd.DataFrame({"VIF": vif}, index=[preprocess.ROTULOS[c] for c in X.columns]).sort_values("VIF", ascending=False)


def ausencia_informativa(df: pd.DataFrame) -> pd.DataFrame:
    """TMN quando o campo está ausente × preenchido: diferença grande indica ausência não aleatória."""
    linhas = []
    for c in ["IDADEPAI", "CONSPRENAT", "MESPRENAT", "SEMAGESTAC", "APGAR5", "PESO"]:
        aus = df[c].isna()
        linhas.append({"Variável": preprocess.ROTULOS[c], "Ausente (%)": 100 * aus.mean(),
                       "TMN ausente (‰)": 1000 * df.loc[aus, "obito_neonatal"].mean() if aus.any() else np.nan,
                       "TMN preenchido (‰)": 1000 * df.loc[~aus, "obito_neonatal"].mean()})
    for c, cod in [("IDANOMAL", "9"), ("ESCMAE2010", "9"), ("RACACORMAE", "ausente")]:
        aus = df[c].astype(str).isin([cod, "ausente"])
        linhas.append({"Variável": preprocess.ROTULOS[c] + " (ignorado)", "Ausente (%)": 100 * aus.mean(),
                       "TMN ausente (‰)": 1000 * df.loc[aus, "obito_neonatal"].mean(),
                       "TMN preenchido (‰)": 1000 * df.loc[~aus, "obito_neonatal"].mean()})
    return pd.DataFrame(linhas)


def gemelares(teste: pd.DataFrame, R: dict) -> dict:
    """Peso dos nascimentos múltiplos e AUROC só entre únicos (não independência entre gêmeos)."""
    multiplo = teste["GRAVIDEZ"].astype(str).isin(["2", "3"]).to_numpy()
    y = teste["obito_neonatal"].to_numpy()
    out = {"% múltiplos": 100 * multiplo.mean(), "% dos óbitos em múltiplos": 100 * y[multiplo].sum() / y.sum()}
    for conj in preprocess.CONJUNTOS:
        e = R["predicoes"][(conj, "XGBoost")]["bruto"]
        out[f"AUROC {conj} (todos)"] = roc_auc_score(y, e)
        out[f"AUROC {conj} (só únicos)"] = roc_auc_score(y[~multiplo], e[~multiplo])
    return out


# ------------------------------------------------------------------ orquestração
def rodar(R: dict, base: pd.DataFrame, nv: pd.DataFrame | None = None, do: pd.DataFrame | None = None) -> dict:
    treino, teste = preprocess.dividir(base)
    ajuste = treino[treino["ano"] < preprocess.ANO_VALIDACAO]
    valid = treino[treino["ano"] == preprocess.ANO_VALIDACAO]
    y_te = teste["obito_neonatal"].to_numpy()
    if nv is None:
        nv = download.carregar("DN", download.ANOS_NASCIMENTO)
    if do is None:
        do = download.carregar("DO", download.ANOS_OBITO)
    C = {"vies_linkage": vies_linkage(nv, do),
         "sensibilidade": sensibilidade_linkage(ajuste, valid, teste, R),
         "calibracao": {}, "decisao": {},
         "colinearidade": colinearidade(ajuste),
         "ausencia": ausencia_informativa(base),
         "gemelares": gemelares(teste, R),
         "tmn_por_amostra": {k: v["TMN (‰)"] for k, v in R["amostras"].items()}}
    for conj in preprocess.CONJUNTOS:
        pr = R["predicoes"][(conj, "XGBoost")]
        C["calibracao"][conj] = {t: intercepto_inclinacao(y_te, pr[t]) for t in ("bruto", "calibrado")}
        C["decisao"][conj] = beneficio_liquido(y_te, pr["calibrado"])
    return C
