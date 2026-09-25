"""Executa o experimento completo e guarda tudo em `data/resultados.pkl`.

Notebook, relatório e slides leem esse arquivo, então os números citados são
sempre os mesmos. Uso: `python -m src.experimento` (≈ 1 h com 32 núcleos).
"""
import pickle
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.metrics import average_precision_score, roc_auc_score

from src import complementares, download, evaluate, explicabilidade, models, preprocess

RESULTADOS = download.DADOS / "resultados.pkl"
C_LR = [0.01, 0.1, 1.0]
N_SHAP = 20_000


def _registrar(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.0f}s] {msg}", flush=True)


def _n_arvores(modelo) -> int:
    """Nº de árvores após o early stopping (XGBoost e LightGBM nomeiam diferente)."""
    it = modelo.best_iteration if hasattr(modelo, "get_booster") else modelo.best_iteration_ - 1
    return int(it) + 1


def rodar(caminho: Path = RESULTADOS) -> dict:
    t0 = time.time()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    base, fluxo, rel_linkage = preprocess.construir()
    rel_linkage = {k: v for k, v in rel_linkage.items() if k != "nao_pareados"}
    treino, teste = preprocess.dividir(base)
    ajuste, valid = treino[treino["ano"] < preprocess.ANO_VALIDACAO], treino[treino["ano"] == preprocess.ANO_VALIDACAO]
    _registrar(f"base: ajuste={len(ajuste)} valid={len(valid)} teste={len(teste)}", t0)

    R = {"fluxo": fluxo, "linkage": rel_linkage, "amostras": {}, "estudos": {}, "params": {},
         "metricas": {}, "predicoes": {}, "limiares": {}, "curvas": {}, "calibracao": {},
         "subgrupos": {}, "shap": {}, "deltas": {}}
    a0 = download.ANOS_NASCIMENTO[0]
    rotulos = [f"ajuste {a0}–{preprocess.ANO_VALIDACAO - 1}", f"validação {preprocess.ANO_VALIDACAO}",
               f"teste {preprocess.ANO_TESTE}"]
    for nome, df in zip(rotulos, [ajuste, valid, teste]):
        R["amostras"][nome] = {"n": len(df), "óbitos": int(df["obito_neonatal"].sum()),
                               "TMN (‰)": 1000 * df["obito_neonatal"].mean()}
    y_te = teste["obito_neonatal"].to_numpy()

    for conj in preprocess.CONJUNTOS:
        X_aj, y_aj = preprocess.xy(ajuste, conj)
        X_va, y_va = preprocess.xy(valid, conj)
        X_te, _ = preprocess.xy(teste, conj)
        finais = {}

        # Regressão logística: C escolhido pela AUPRC na validação.
        melhor = max(C_LR, key=lambda C: average_precision_score(
            y_va, models.ajustar_lr(X_aj, y_aj, C).predict_proba(X_va)[:, 1]))
        finais["LR"] = models.ajustar_lr(X_aj, y_aj, melhor)
        R["params"][(conj, "LR")] = {"C": melhor}
        _registrar(f"{conj} LR C={melhor}", t0)

        for nome, buscar, ajustar in [("XGBoost", models.buscar_xgb, models.ajustar_xgb),
                                      ("LightGBM", models.buscar_lgbm, models.ajustar_lgbm)]:
            estudo = buscar(X_aj, y_aj, X_va, y_va)
            params, spw = models.params_do_estudo(estudo, y_aj)
            extra = {"curvas_treino": True} if nome == "XGBoost" else {}
            finais[nome] = ajustar(X_aj, y_aj, X_va, y_va, params, spw, **extra)
            R["estudos"][(conj, nome)] = estudo.trials_dataframe()
            R["params"][(conj, nome)] = {**params, "scale_pos_weight": spw,
                                         "n_arvores": _n_arvores(finais[nome])}
            _registrar(f"{conj} {nome} AUPRC valid={estudo.best_value:.4f}", t0)
        R["curvas"][conj] = models.curvas_iteracao(finais["XGBoost"])

        for nome, modelo in finais.items():
            cal = models.Recalibrado(modelo, X_va, y_va)
            p_va, p_te = cal.predict_proba(X_va)[:, 1], cal.predict_proba(X_te)[:, 1]
            limiar = evaluate.limiar_max_f1(y_va, p_va)
            R["limiares"][(conj, nome)] = limiar
            R["predicoes"][(conj, nome)] = {"calibrado": p_te, "bruto": cal.bruto(X_te)}
            R["metricas"][(conj, nome)] = evaluate.com_ic(y_te, p_te, limiar, escore=cal.bruto(X_te))
            _registrar(f"{conj} {nome} teste AUROC={roc_auc_score(y_te, p_te):.4f}", t0)

        p_bruto = R["predicoes"][(conj, "XGBoost")]["bruto"]
        p_cal = R["predicoes"][(conj, "XGBoost")]["calibrado"]
        R["calibracao"][conj] = {"bruto": evaluate.calibracao(y_te, p_bruto),
                                 "calibrado": evaluate.calibracao(y_te, p_cal),
                                 "OE_bruto": evaluate.razao_obs_esp(y_te, p_bruto),
                                 "OE_calibrado": evaluate.razao_obs_esp(y_te, p_cal)}
        df_sub = teste[["raca_mae", "escol_mae", "capital"]].assign(y=y_te, p=p_cal, escore=p_bruto)
        df_sub["capital"] = df_sub["capital"].map({1.0: "Capital", 0.0: "Interior"})
        R["subgrupos"][conj] = {g: evaluate.por_subgrupo(df_sub, g, col_escore="escore", n_bootstrap=500)
                                for g in ["raca_mae", "escol_mae", "capital"]}

        amostra = X_te.sample(N_SHAP, random_state=models.SEMENTE)
        valores, base_shap = explicabilidade.shap_nativo(finais["XGBoost"], amostra)
        R["shap"][conj] = {"valores": valores, "base": base_shap, "X": amostra}
        _registrar(f"{conj} SHAP", t0)

    for m in ["AUROC", "AUPRC"]:
        metrica = roc_auc_score if m == "AUROC" else average_precision_score
        R["deltas"][f"XGBoost A − B ({m})"] = evaluate.delta_pareado(
            y_te, R["predicoes"][("A_completo", "XGBoost")]["bruto"],
            R["predicoes"][("B_pre_parto", "XGBoost")]["bruto"], metrica)
        for conj in preprocess.CONJUNTOS:
            R["deltas"][f"XGBoost − LR, {conj} ({m})"] = evaluate.delta_pareado(
                y_te, R["predicoes"][(conj, "XGBoost")]["bruto"],
                R["predicoes"][(conj, "LR")]["bruto"], metrica)

    R["y_teste"] = y_te
    R["complementares"] = complementares.rodar(R, base)
    _registrar("análises complementares", t0)
    R["tempo_total_s"] = time.time() - t0
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "wb") as f:
        pickle.dump(R, f)
    return R


def carregar(caminho: Path = RESULTADOS) -> dict:
    with open(caminho, "rb") as f:
        return pickle.load(f)


def rodar_complementares(caminho: Path = RESULTADOS) -> dict:
    """Refaz só as análises complementares sobre um experimento já gravado (sem repetir a busca)."""
    R = carregar(caminho)
    R["complementares"] = complementares.rodar(R, pd.read_parquet(download.DADOS / "base_analitica.parquet"))
    with open(caminho, "wb") as f:
        pickle.dump(R, f)
    return R


def atualizar_complementares(caminho: Path = RESULTADOS) -> dict:
    """Refaz só as partes baratas das complementares e a explicabilidade (sem reajustar a sensibilidade)."""
    R = carregar(caminho)
    complementares.atualizar(R, pd.read_parquet(download.DADOS / "base_analitica.parquet"))
    with open(caminho, "wb") as f:
        pickle.dump(R, f)
    return R


if __name__ == "__main__":
    import sys
    if "--complementares" in sys.argv:
        rodar_complementares()
    elif "--atualizar" in sys.argv:
        atualizar_complementares()
    else:
        rodar()
