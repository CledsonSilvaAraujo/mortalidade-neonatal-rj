"""Métricas no teste, intervalos por bootstrap, calibração e análises de subgrupo."""
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_recall_curve, roc_auc_score)

N_BOOTSTRAP = 500
SEMENTE = 42
FRACAO_ALTO_RISCO = 0.05  # o artigo reporta a captura de óbitos entre os 5% de maior risco


def limiar_max_f1(y, p) -> float:
    """Limiar que maximiza o F1, escolhido na validação e aplicado ao teste."""
    prec, rec, lim = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(lim[np.argmax(f1[:-1])])


def captura_top(y, escore, fracao: float = FRACAO_ALTO_RISCO) -> float:
    """Fração dos óbitos entre exatamente os `fracao` com maior escore."""
    k = int(round(fracao * len(escore)))
    topo = np.argsort(-np.asarray(escore), kind="stable")[:k]
    return float(np.asarray(y)[topo].sum() / max(np.sum(y), 1))


def metricas(y, p, limiar: float, escore=None) -> dict:
    """Discriminação (AUROC, AUPRC, captura) pelo `escore` bruto do modelo; o resto pela probabilidade `p`.

    A isotônica é monótona, mas cria empates (é uma função em degraus), o que achata
    AUROC/AUPRC e torna ambíguo o "top 5%". Por isso a ordenação usa o escore bruto,
    e as métricas que dependem do valor da probabilidade (Brier, limiar) usam `p`.
    """
    y, p = np.asarray(y), np.asarray(p)
    escore = p if escore is None else np.asarray(escore)
    pred = p >= limiar
    vp, fp = (pred & (y == 1)).sum(), (pred & (y == 0)).sum()
    fn, vn = (~pred & (y == 1)).sum(), (~pred & (y == 0)).sum()
    return {
        "AUROC": roc_auc_score(y, escore),
        "AUPRC": average_precision_score(y, escore),
        "Brier": brier_score_loss(y, p),
        "F1": f1_score(y, pred, zero_division=0),
        "Precisão": vp / max(vp + fp, 1),
        "Sensibilidade": vp / max(vp + fn, 1),
        "Especificidade": vn / max(vn + fp, 1),
        "VPN": vn / max(vn + fn, 1),
        "Óbitos no top 5%": captura_top(y, escore),
        "O/E": razao_obs_esp(y, p),
    }


def _reamostras(y, n: int, semente: int):
    """Índices bootstrap estratificados (mantêm o nº de óbitos, raro no teste)."""
    rng = np.random.default_rng(semente)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    for _ in range(n):
        yield np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])


def com_ic(y, p, limiar: float, escore=None, n: int = N_BOOTSTRAP, semente: int = SEMENTE) -> pd.DataFrame:
    """Estimativa pontual e IC 95% (percentis do bootstrap) de cada métrica."""
    y, p = np.asarray(y), np.asarray(p)
    escore = p if escore is None else np.asarray(escore)
    ponto = metricas(y, p, limiar, escore)
    boot = pd.DataFrame([metricas(y[i], p[i], limiar, escore[i]) for i in _reamostras(y, n, semente)])
    return pd.DataFrame({"valor": ponto, "ic_inf": boot.quantile(0.025), "ic_sup": boot.quantile(0.975)})


def delta_pareado(y, p_a, p_b, metrica=roc_auc_score, n: int = N_BOOTSTRAP,
                  semente: int = SEMENTE) -> dict:
    """Diferença A − B com bootstrap pareado (mesmas reamostras para os dois modelos)."""
    y, p_a, p_b = map(np.asarray, (y, p_a, p_b))
    d = np.array([metrica(y[i], p_a[i]) - metrica(y[i], p_b[i]) for i in _reamostras(y, n, semente)])
    return {"delta": metrica(y, p_a) - metrica(y, p_b),
            "ic_inf": np.quantile(d, 0.025), "ic_sup": np.quantile(d, 0.975),
            "p_bilateral": min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))}


def calibracao(y, p, n_faixas: int = 10) -> pd.DataFrame:
    """Observado × previsto por decil de risco (o risco é muito concentrado no último)."""
    df = pd.DataFrame({"y": y, "p": p})
    df["faixa"] = pd.qcut(df["p"].rank(method="first"), n_faixas, labels=False) + 1
    return df.groupby("faixa").agg(n=("y", "size"), previsto=("p", "mean"), observado=("y", "mean"))


def razao_obs_esp(y, p) -> float:
    return float(np.sum(y) / np.sum(p))


def por_subgrupo(df: pd.DataFrame, grupo: str, col_y: str = "y", col_p: str = "p",
                 col_escore: str | None = None, min_obitos: int = 20, n_bootstrap: int = 0,
                 semente: int = SEMENTE) -> pd.DataFrame:
    """AUROC e AUPRC (pelo escore) e O/E (pela probabilidade) por subgrupo; com `n_bootstrap`, IC 95%."""
    col_escore = col_escore or col_p
    linhas = []
    for nome, g in df.groupby(grupo, observed=True):
        if g[col_y].sum() < min_obitos:
            continue
        y, p, e = g[col_y].to_numpy(), g[col_p].to_numpy(), g[col_escore].to_numpy()
        linha = {grupo: nome, "n": len(g), "óbitos": int(y.sum()), "TMN (‰)": 1000 * y.mean(),
                 "AUROC": roc_auc_score(y, e), "AUPRC": average_precision_score(y, e),
                 "O/E": razao_obs_esp(y, p)}
        if n_bootstrap:
            # Reamostragem simples (não estratificada), para o O/E refletir a variação do nº de óbitos.
            rng = np.random.default_rng(semente)
            auc, oe = [], []
            for _ in range(n_bootstrap):
                i = rng.integers(0, len(y), len(y))
                auc.append(roc_auc_score(y[i], e[i]))
                oe.append(razao_obs_esp(y[i], p[i]))
            linha.update({"AUROC_inf": np.quantile(auc, 0.025), "AUROC_sup": np.quantile(auc, 0.975),
                          "O/E_inf": np.quantile(oe, 0.025), "O/E_sup": np.quantile(oe, 0.975)})
        linhas.append(linha)
    return pd.DataFrame(linhas)
