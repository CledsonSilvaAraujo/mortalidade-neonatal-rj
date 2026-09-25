"""Pipelines, busca bayesiana de hiperparâmetros (Optuna) e recalibração.

Esquema temporal, espelhando o artigo de referência: ajuste nos anos iniciais,
escolha de hiperparâmetros, early stopping e recalibração no penúltimo ano
(preprocess.ANO_VALIDACAO) e teste final intocado no último (preprocess.ANO_TESTE).
"""
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

SEMENTE = 42
N_TENTATIVAS = {"xgb": 15, "lgbm": 10}
MAX_ARVORES = 1500
PARADA_ANTECIPADA = 50


def razao_neg_pos(y) -> float:
    y = np.asarray(y)
    return float((y == 0).sum() / max((y == 1).sum(), 1))


def preprocessador_lr() -> ColumnTransformer:
    """Imputação (com indicador de ausência), padronização e one-hot, ajustados só no treino."""
    num = Pipeline([("imputa", SimpleImputer(strategy="median", add_indicator=True)),
                    ("padroniza", StandardScaler())])
    cat = OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=False)
    return ColumnTransformer([
        ("num", num, make_column_selector(dtype_include="number")),
        ("cat", cat, make_column_selector(dtype_include="category")),
    ])


def ajustar_lr(X, y, C: float = 1.0) -> Pipeline:
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000)
    return Pipeline([("prep", preprocessador_lr()), ("clf", clf)]).fit(X, y)


def _xgb(params: dict, spw: float) -> XGBClassifier:
    # O early stopping usa a última métrica da lista: a AUPRC, a mesma que a busca otimiza.
    return XGBClassifier(
        n_estimators=MAX_ARVORES, early_stopping_rounds=PARADA_ANTECIPADA,
        tree_method="hist", enable_categorical=True, max_cat_to_onehot=1,
        eval_metric=["logloss", "aucpr"], scale_pos_weight=spw,
        n_jobs=-1, random_state=SEMENTE, **params,
    )


def espaco_xgb(trial: optuna.Trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 50, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "gamma": trial.suggest_float("gamma", 1e-3, 5, log=True),
    }


def _peso_classe(trial: optuna.Trial, y) -> float:
    # sqrt da razão é um meio-termo entre ignorar e compensar todo o desbalanceamento.
    opcao = trial.suggest_categorical("peso_classe", ["nenhum", "raiz", "total"])
    r = razao_neg_pos(y)
    return {"nenhum": 1.0, "raiz": float(np.sqrt(r)), "total": r}[opcao]


def ajustar_xgb(X_tr, y_tr, X_va, y_va, params: dict, spw: float,
                curvas_treino: bool = False) -> XGBClassifier:
    """Com `curvas_treino`, também registra as métricas no ajuste a cada árvore (bem mais lento)."""
    avaliacao = [(X_tr, y_tr), (X_va, y_va)] if curvas_treino else [(X_va, y_va)]
    return _xgb(params, spw).fit(X_tr, y_tr, eval_set=avaliacao, verbose=False)


def buscar_xgb(X_tr, y_tr, X_va, y_va, n_tentativas: int = N_TENTATIVAS["xgb"]) -> optuna.Study:
    """Otimização bayesiana (TPE) da AUPRC no ano de validação."""
    def objetivo(trial):
        m = ajustar_xgb(X_tr, y_tr, X_va, y_va, espaco_xgb(trial), _peso_classe(trial, y_tr))
        trial.set_user_attr("n_arvores", m.best_iteration + 1)
        return average_precision_score(y_va, m.predict_proba(X_va)[:, 1])
    estudo = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=SEMENTE))
    estudo.optimize(objetivo, n_trials=n_tentativas, callbacks=[_log_tentativa])
    return estudo


def espaco_lgbm(trial: optuna.Trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 128, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 500, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
    }


def ajustar_lgbm(X_tr, y_tr, X_va, y_va, params: dict, spw: float) -> LGBMClassifier:
    m = LGBMClassifier(n_estimators=MAX_ARVORES, scale_pos_weight=spw, n_jobs=-1,
                       random_state=SEMENTE, verbose=-1, **params)
    return m.fit(X_tr, y_tr, eval_X=(X_va,), eval_y=(y_va,), eval_metric="average_precision",
                 callbacks=[early_stopping(PARADA_ANTECIPADA, verbose=False), log_evaluation(0)])


def buscar_lgbm(X_tr, y_tr, X_va, y_va, n_tentativas: int = N_TENTATIVAS["lgbm"]) -> optuna.Study:
    def objetivo(trial):
        m = ajustar_lgbm(X_tr, y_tr, X_va, y_va, espaco_lgbm(trial), _peso_classe(trial, y_tr))
        return average_precision_score(y_va, m.predict_proba(X_va)[:, 1])
    estudo = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=SEMENTE))
    estudo.optimize(objetivo, n_trials=n_tentativas, callbacks=[_log_tentativa])
    return estudo


def _log_tentativa(estudo: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    print(f"    tentativa {trial.number}: AUPRC={trial.value:.4f} "
          f"(melhor {estudo.best_value:.4f}, {trial.duration.total_seconds():.0f}s)", flush=True)


def params_do_estudo(estudo: optuna.Study, y_tr) -> tuple[dict, float]:
    """Separa os hiperparâmetros do modelo do peso de classe escolhido."""
    p = dict(estudo.best_params)
    r = razao_neg_pos(y_tr)
    spw = {"nenhum": 1.0, "raiz": float(np.sqrt(r)), "total": r}[p.pop("peso_classe")]
    return p, spw


class Recalibrado:
    """Modelo + regressão isotônica ajustada no ano de validação.

    O peso de classe distorce as probabilidades; a isotônica é monótona, então
    preserva a ordenação (AUROC/AUPRC quase iguais) e corrige a calibração.
    """

    def __init__(self, modelo, X_va, y_va):
        self.modelo = modelo
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        self.iso.fit(self.bruto(X_va), y_va)

    def bruto(self, X) -> np.ndarray:
        return self.modelo.predict_proba(X)[:, 1]

    def predict_proba(self, X) -> np.ndarray:
        p = self.iso.predict(self.bruto(X))
        return np.column_stack([1 - p, p])


def curvas_iteracao(modelo: XGBClassifier) -> pd.DataFrame:
    """Log-loss e AUPRC por árvore, em treino e validação (para o notebook)."""
    r = modelo.evals_result()
    return pd.DataFrame({
        "treino_logloss": r["validation_0"]["logloss"], "valid_logloss": r["validation_1"]["logloss"],
        "treino_aucpr": r["validation_0"]["aucpr"], "valid_aucpr": r["validation_1"]["aucpr"],
    }).rename_axis("iteracao")
