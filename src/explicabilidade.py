"""Explicabilidade (XAI) com SHAP: global, dependência, local, estabilidade e atributos sensíveis.

Os modelos finais são reajustados com os hiperparâmetros gravados. O ajuste é
determinístico (mesmas predições do experimento, diferença máxima 0) e fica em
cache em data/modelos/ para não ser repetido.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from lightgbm import Booster
from scipy.stats import spearmanr
import xgboost as xgb
from xgboost import XGBClassifier

from src import download, evaluate, models, preprocess

PASTA_MODELOS = download.DADOS / "modelos"
DEPENDENCIA = {"A_completo": ["APGAR5", "PESO", "SEMAGESTAC"], "B_pre_parto": ["SEMAGESTAC", "CONSPRENAT", "IDADEMAE"]}
SENSIVEIS = ["RACACORMAE", "ESCMAE2010"]
N_TOP = 6


def shap_nativo(modelo, X) -> tuple[np.ndarray, float]:
    """SHAP exato (TreeSHAP) calculado pela própria biblioteca do modelo.

    Para o XGBoost, restringe às árvores da melhor iteração (ver `_arvores`); sem isso, as
    contribuições somam o log-odds de outro modelo. `verificar_aditividade` confere o resultado.
    """
    if isinstance(modelo, XGBClassifier):
        c = modelo.get_booster().predict(xgb.DMatrix(X, enable_categorical=True), pred_contribs=True,
                                         iteration_range=_arvores(modelo))
    else:  # lightgbm.Booster
        c = modelo.predict(X, pred_contrib=True)
    return c[:, :-1], float(c[0, -1])


def _arvores(modelo) -> tuple[int, int]:
    """Só as árvores até a melhor iteração do early stopping, como no predict_proba.

    Sem isso, o booster usa todas as árvores treinadas (inclusive as 50 posteriores à melhor
    iteração) e o SHAP explicaria um modelo diferente do avaliado.
    """
    return (0, modelo.best_iteration + 1)


def verificar_aditividade(modelo, X) -> dict:
    """Erro máximo de Σφ + φ0 contra o log-odds do modelo avaliado (o do predict_proba).

    Compara o SHAP nativo e o do `shap.TreeExplainer`; os dois devem reproduzir o log-odds.
    """
    p = np.clip(modelo.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
    logit = np.log(p / (1 - p))
    v, b = shap_nativo(modelo, X)
    te = shap.TreeExplainer(modelo)
    v_te = te.shap_values(X)
    return {"nativo": float(np.abs(v.sum(1) + b - logit).max()),
            "TreeExplainer": float(np.abs(v_te.sum(1) + np.ravel(te.expected_value)[0] - logit).max())}


def _params(R, conj, nome):
    p = dict(R["params"][(conj, nome)])
    return {k: v for k, v in p.items() if k not in ("scale_pos_weight", "n_arvores")}, p["scale_pos_weight"]


def modelos_finais(R, ajuste, valid) -> dict:
    """XGBoost e LightGBM finais de cada conjunto (reajuste determinístico, com cache)."""
    PASTA_MODELOS.mkdir(parents=True, exist_ok=True)
    saida = {}
    for conj in preprocess.CONJUNTOS:
        (X_aj, y_aj), (X_va, y_va) = preprocess.xy(ajuste, conj), preprocess.xy(valid, conj)
        arq = PASTA_MODELOS / f"xgb_{conj}.json"
        if arq.exists():
            m = XGBClassifier(); m.load_model(arq)
        else:
            m = models.ajustar_xgb(X_aj, y_aj, X_va, y_va, *_params(R, conj, "XGBoost"))
            m.save_model(arq)
        saida[(conj, "XGBoost")] = m
        arq = PASTA_MODELOS / f"lgbm_{conj}.txt"
        if arq.exists():
            saida[(conj, "LightGBM")] = Booster(model_file=str(arq))
        else:
            lg = models.ajustar_lgbm(X_aj, y_aj, X_va, y_va, *_params(R, conj, "LightGBM"))
            lg.booster_.save_model(str(arq))
            saida[(conj, "LightGBM")] = lg.booster_
    return saida


def _importancia(valores, colunas) -> pd.Series:
    return pd.Series(np.abs(valores).mean(0), index=colunas)


def estabilidade(R, modelos: dict, conj: str, n_particoes: int = 2, semente: int = 42) -> dict:
    """Concordância dos rankings de importância: XGBoost × LightGBM e entre metades da amostra."""
    s = R["shap"][conj]
    imp_xgb = _importancia(s["valores"], s["X"].columns)
    val_lgbm, _ = shap_nativo(modelos[(conj, "LightGBM")], s["X"])
    imp_lgbm = _importancia(val_lgbm, s["X"].columns)
    idx = np.random.default_rng(semente).permutation(len(s["X"]))
    metades = np.array_split(idx, n_particoes)
    imp_m = [_importancia(s["valores"][m], s["X"].columns) for m in metades]
    top = lambda imp: set(imp.sort_values(ascending=False).index[:5])
    return {
        "spearman_xgb_lgbm": spearmanr(imp_xgb, imp_lgbm).statistic,
        "top5_comum_xgb_lgbm": len(top(imp_xgb) & top(imp_lgbm)),
        "spearman_metades": spearmanr(imp_m[0], imp_m[1]).statistic,
        "top5_comum_metades": len(top(imp_m[0]) & top(imp_m[1])),
        "importancia_lgbm": imp_lgbm.sort_values(ascending=False),
    }


def atributos_sensiveis(R, conj: str) -> dict:
    """Peso de raça/cor e escolaridade da mãe nas explicações e SHAP médio por categoria."""
    s = R["shap"][conj]
    imp = _importancia(s["valores"], s["X"].columns)
    out = {"fracao_importancia": {c: imp[c] / imp.sum() for c in SENSIVEIS}}
    for c in SENSIVEIS:
        j = list(s["X"].columns).index(c)
        out[c] = (pd.DataFrame({"categoria": s["X"][c].astype(str).to_numpy(), "shap": s["valores"][:, j]})
                  .groupby("categoria")["shap"].agg(["mean", "size"]))
    return out


def explicacoes_locais(R, modelos: dict, teste: pd.DataFrame, conj: str = "A_completo") -> dict:
    """Explicação individual de três casos típicos do ano de teste.

    - óbito de maior risco previsto (verdadeiro positivo);
    - óbito não sinalizado no limiar escolhido na validação (falso negativo mediano);
    - sobrevivente de maior risco previsto (falso positivo).
    """
    y = teste["obito_neonatal"].to_numpy()
    p = R["predicoes"][(conj, "XGBoost")]["calibrado"]
    e = R["predicoes"][(conj, "XGBoost")]["bruto"]
    limiar = R["limiares"][(conj, "XGBoost")]
    fn = np.flatnonzero((y == 1) & (p < limiar))
    casos = {
        "Verdadeiro positivo": int(np.flatnonzero(y == 1)[np.argmax(e[y == 1])]),
        "Falso negativo": int(fn[np.argsort(e[fn])[len(fn) // 2]]),
        "Falso positivo": int(np.flatnonzero(y == 0)[np.argmax(e[y == 0])]),
    }
    X, _ = preprocess.xy(teste, conj)
    _, base = shap_nativo(modelos[(conj, "XGBoost")], X.iloc[:1])
    saida = {"base_logodds": base, "base_prob": 1 / (1 + np.exp(-base)), "casos": {}}
    for nome, i in casos.items():
        linha = X.iloc[[i]]
        v = shap_nativo(modelos[(conj, "XGBoost")], linha)[0][0]
        contrib = (pd.DataFrame({"variavel": [preprocess.ROTULOS[c] for c in X.columns],
                                 "valor": [str(linha.iloc[0][c]) for c in X.columns], "shap": v})
                   .assign(abs=lambda d: d["shap"].abs()).sort_values("abs", ascending=False).drop(columns="abs"))
        logit = base + v.sum()
        saida["casos"][nome] = {
            "indice": i, "obito": int(y[i]), "prob_calibrada": float(p[i]), "prob_bruta": float(e[i]),
            "aditividade_erro": abs(1 / (1 + np.exp(-logit)) - e[i]),  # SHAP soma exatamente a predição
            "top": contrib.head(N_TOP), "resto": float(contrib["shap"].iloc[N_TOP:].sum()),
        }
    return saida


def dependencia(R, conj: str) -> dict:
    """Valores (x, SHAP) para os gráficos de dependência das principais variáveis."""
    s = R["shap"][conj]
    cols = list(s["X"].columns)
    return {v: pd.DataFrame({"x": s["X"][v].to_numpy(dtype=float), "shap": s["valores"][:, cols.index(v)]})
            for v in DEPENDENCIA[conj]}


def rodar(R, base) -> dict:
    treino, teste = preprocess.dividir(base)
    ajuste = treino[treino["ano"] < preprocess.ANO_VALIDACAO]
    valid = treino[treino["ano"] == preprocess.ANO_VALIDACAO]
    modelos = modelos_finais(R, ajuste, valid)
    aditividade = {}
    for conj in preprocess.CONJUNTOS:  # substitui o SHAP do experimento pelo exato, na mesma amostra
        X = R["shap"][conj]["X"]
        aditividade[conj] = verificar_aditividade(modelos[(conj, "XGBoost")], X.iloc[:3000])
        R["shap"][conj]["valores"], R["shap"][conj]["base"] = shap_nativo(modelos[(conj, "XGBoost")], X)
    return {
        "aditividade": aditividade,
        "estabilidade": {c: estabilidade(R, modelos, c) for c in preprocess.CONJUNTOS},
        "sensiveis": {c: atributos_sensiveis(R, c) for c in preprocess.CONJUNTOS},
        "locais": explicacoes_locais(R, modelos, teste),
        "dependencia": {c: dependencia(R, c) for c in preprocess.CONJUNTOS},
    }
