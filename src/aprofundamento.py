"""V2: questionamentos para publicação, sobre o experimento principal já gravado.

Cada análise responde a uma pergunta que um revisor faria:

- baseline clínico: um escore com peso, idade gestacional e Apgar 5 faz o mesmo que o XGBoost?
- idade no óbito: o modelo acerta demais os óbitos do dia 0 (sinal de vazamento na DN)?
- pandemia: tirar 2020–2021 do ajuste muda o teste?
- chaves do linkage: quanto o desempenho depende das variáveis que também servem de chave?
- janelas deslizantes: a calibração se degrada ano a ano (modelo congelado × reajustado)?
- simulação de padrão-ouro: o estimador de falsos pares acerta quando a verdade é conhecida?
- etapa probabilística (Fellegi–Sunter): quantos óbitos não pareados dá para recuperar, e com que erro?

Todas usam os hiperparâmetros escolhidos no experimento principal, sem repetir a busca.
Uso: `python -m src.aprofundamento [parte ...]` (grava `data/resultados_v2.pkl`; sem partes, roda todas).
"""
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src import complementares, download, evaluate, experimento, linkage, models, preprocess

RESULTADOS_V2 = download.DADOS / "resultados_v2.pkl"
SEMENTE = 42
N_BOOT = 200

VARS_BASELINE = ["PESO", "SEMAGESTAC", "APGAR5"]
VARS_BASELINE_ANOMALIA = [*VARS_BASELINE, "IDANOMAL"]
ANOS_PANDEMIA = [2020, 2021]
# Variáveis da DN que também entram nas chaves do linkage (CODMUNRES e CODESTAB não são preditores).
CHAVES_PREDITORAS = ["PESO", "IDADEMAE", "SEMAGESTAC", "GRAVIDEZ", "PARTO"]
FAIXAS_IDADE_OBITO = {"dia 0": (0, 0), "1–6 dias": (1, 6), "7–27 dias": (7, 27)}


def _registrar(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.0f}s] {msg}", flush=True)


def _particao(base: pd.DataFrame, ano_teste: int = preprocess.ANO_TESTE):
    ajuste = base[base["ano"] < ano_teste - 1]
    valid = base[base["ano"] == ano_teste - 1]
    teste = base[base["ano"] == ano_teste]
    return ajuste, valid, teste


def _auc_ic(y, e, n: int = N_BOOT, semente: int = SEMENTE) -> tuple[float, float, float]:
    """AUROC e IC 95% por bootstrap estratificado."""
    y, e = np.asarray(y), np.asarray(e)
    boot = [roc_auc_score(y[i], e[i]) for i in evaluate._reamostras(y, n, semente)]
    return roc_auc_score(y, e), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


# ------------------------------------------------------------------ baseline clínico
def baseline_clinico(ajuste, valid, teste, R) -> dict:
    """Regressão logística só com peso, idade gestacional e Apgar 5 (e + anomalia) × XGBoost completo."""
    y_aj, y_va = ajuste["obito_neonatal"].to_numpy(), valid["obito_neonatal"].to_numpy()
    y_te = teste["obito_neonatal"].to_numpy()
    ref = R["predicoes"][("A_completo", "XGBoost")]["bruto"]
    saida = {}
    for nome, cols in [("peso + IG + Apgar 5", VARS_BASELINE),
                       ("peso + IG + Apgar 5 + anomalia", VARS_BASELINE_ANOMALIA)]:
        C = max([0.01, 0.1, 1.0], key=lambda C: average_precision_score(
            y_va, models.ajustar_lr(ajuste[cols], y_aj, C).predict_proba(valid[cols])[:, 1]))
        cal = models.Recalibrado(models.ajustar_lr(ajuste[cols], y_aj, C), valid[cols], y_va)
        p_va = cal.predict_proba(valid[cols])[:, 1]
        escore = cal.bruto(teste[cols])
        saida[nome] = {
            "C": C,
            "metricas": evaluate.com_ic(y_te, cal.predict_proba(teste[cols])[:, 1],
                                        evaluate.limiar_max_f1(y_va, p_va), escore=escore),
            "delta_auroc": evaluate.delta_pareado(y_te, ref, escore),
            "delta_auprc": evaluate.delta_pareado(y_te, ref, escore, average_precision_score),
        }
    return saida


# ------------------------------------------------------------------ idade no óbito
def por_idade_obito(teste: pd.DataFrame, dias_vida: pd.Series, R: dict) -> pd.DataFrame:
    """AUROC e captura no top 5% de cada faixa de idade no óbito contra todos os sobreviventes.

    Se a DN fosse preenchida ou corrigida depois do óbito, o modelo acertaria desproporcionalmente
    os óbitos do dia 0, que acontecem antes do fechamento da declaração.
    """
    y = teste["obito_neonatal"].to_numpy()
    dias = dias_vida.reindex(teste.index).to_numpy()
    k = int(round(evaluate.FRACAO_ALTO_RISCO * len(y)))
    linhas = []
    for conj in preprocess.CONJUNTOS:
        e = R["predicoes"][(conj, "XGBoost")]["bruto"]
        corte = np.sort(e)[::-1][k - 1]
        for faixa, (lo, hi) in FAIXAS_IDADE_OBITO.items():
            pos = (y == 1) & (dias >= lo) & (dias <= hi)
            sel = pos | (y == 0)
            auc, inf, sup = _auc_ic(y[sel], e[sel])
            linhas.append({"Conjunto": conj, "Faixa": faixa, "Óbitos": int(pos.sum()),
                           "AUROC": auc, "AUROC_inf": inf, "AUROC_sup": sup,
                           "Top 5%": float((e[pos] >= corte).mean())})
    return pd.DataFrame(linhas)


def declaracao_pos_obito(nv_rot: pd.DataFrame, teste: pd.DataFrame, R: dict, n_min: int = 20) -> pd.DataFrame:
    """Teste direto de vazamento: DN declarada (DTDECLARAC) ou digitada (DTCADASTRO) depois do óbito.

    Dentro de cada faixa de idade no óbito (que controla a gravidade), compara a AUROC dos óbitos cuja DN
    foi preenchida depois do óbito com a dos preenchidos antes, sempre contra todos os sobreviventes.
    Se a DN "soubesse" do óbito, o primeiro grupo seria previsto bem melhor.
    """
    t = nv_rot.reindex(teste.index)
    dias = t["dias_vida"].to_numpy(dtype=float)
    obito = linkage._data(t["DTNASC"]) + pd.to_timedelta(t["dias_vida"], unit="D")
    y = teste["obito_neonatal"].to_numpy()
    k = int(round(evaluate.FRACAO_ALTO_RISCO * len(y)))
    linhas = []
    for campo, rotulo in [("DTDECLARAC", "declarada"), ("DTCADASTRO", "digitada")]:
        depois = (linkage._data(t[campo]) > obito).to_numpy()
        for conj in preprocess.CONJUNTOS:
            e = R["predicoes"][(conj, "XGBoost")]["bruto"]
            corte = np.sort(e)[::-1][k - 1]
            for faixa, (lo, hi) in {"todas": (0, 27), **FAIXAS_IDADE_OBITO}.items():
                na_faixa = (y == 1) & (dias >= lo) & (dias <= hi)
                for momento, sel_pos in [("depois do óbito", na_faixa & depois), ("antes ou no dia", na_faixa & ~depois)]:
                    linha = {"Campo": rotulo, "Conjunto": conj, "Faixa": faixa, "DN": momento,
                             "Óbitos": int(sel_pos.sum()), "% da faixa": 100 * sel_pos.sum() / max(na_faixa.sum(), 1)}
                    if sel_pos.sum() >= n_min:
                        auc, inf, sup = _auc_ic(y[sel_pos | (y == 0)], e[sel_pos | (y == 0)])
                        linha.update({"AUROC": auc, "AUROC_inf": inf, "AUROC_sup": sup,
                                      "Top 5%": float((e[sel_pos] >= corte).mean())})
                    linhas.append(linha)
    return pd.DataFrame(linhas)


# ------------------------------------------------------------------ reajustes com os mesmos hiperparâmetros
def _params(R: dict, conj: str) -> tuple[dict, float]:
    p = R["params"][(conj, "XGBoost")]
    return {k: v for k, v in p.items() if k not in ("scale_pos_weight", "n_arvores")}, p["scale_pos_weight"]


def _vars(conj: str, remover=()) -> list[str]:
    return [v for v in preprocess.CONJUNTOS[conj] if v not in remover]


def reajustar(ajuste, valid, teste, R, conj: str, remover=()) -> dict:
    """XGBoost com os hiperparâmetros principais, avaliado no MESMO teste; Δ por bootstrap pareado."""
    cols = _vars(conj, remover)
    params, spw = _params(R, conj)
    y_va, y_te = valid["obito_neonatal"].to_numpy(), teste["obito_neonatal"].to_numpy()
    modelo = models.ajustar_xgb(ajuste[cols], ajuste["obito_neonatal"].to_numpy(), valid[cols], y_va, params, spw)
    cal = models.Recalibrado(modelo, valid[cols], y_va)
    escore = cal.bruto(teste[cols])
    ref = R["predicoes"][(conj, "XGBoost")]["bruto"]
    return {
        "n_ajuste": len(ajuste), "variaveis": cols,
        "metricas": evaluate.com_ic(y_te, cal.predict_proba(teste[cols])[:, 1],
                                    evaluate.limiar_max_f1(y_va, cal.predict_proba(valid[cols])[:, 1]),
                                    escore=escore),
        "delta_auroc": evaluate.delta_pareado(y_te, escore, ref),
        "delta_auprc": evaluate.delta_pareado(y_te, escore, ref, average_precision_score),
    }


def sem_pandemia(ajuste, valid, teste, R) -> dict:
    fora = ajuste[~ajuste["ano"].isin(ANOS_PANDEMIA)]
    return {conj: reajustar(fora, valid, teste, R, conj) for conj in preprocess.CONJUNTOS}


def sem_chaves(ajuste, valid, teste, R) -> dict:
    return {conj: reajustar(ajuste, valid, teste, R, conj, remover=CHAVES_PREDITORAS)
            for conj in preprocess.CONJUNTOS}


# ------------------------------------------------------------------ janelas deslizantes
def janelas_deslizantes(base: pd.DataFrame, R: dict, anos_teste=None) -> pd.DataFrame:
    """Para cada ano t: ajuste < t−1, validação t−1, teste t.

    "reajustado" refaz o modelo a cada ano; "congelado" é o modelo da primeira janela aplicado a todos
    os anos seguintes, sem reajuste nem recalibração, para medir a degradação por drift.
    """
    anos_teste = list(anos_teste or range(2020, preprocess.ANO_TESTE + 1))
    linhas = []
    for conj in preprocess.CONJUNTOS:
        cols = _vars(conj)
        params, spw = _params(R, conj)
        congelado = None
        for t in anos_teste:
            aj, va, te = _particao(base, t)
            y_va, y_te = va["obito_neonatal"].to_numpy(), te["obito_neonatal"].to_numpy()
            cal = models.Recalibrado(models.ajustar_xgb(aj[cols], aj["obito_neonatal"].to_numpy(),
                                                        va[cols], y_va, params, spw), va[cols], y_va)
            congelado = congelado or cal
            for nome, m in [("reajustado", cal), ("congelado", congelado)]:
                p = m.predict_proba(te[cols])[:, 1]
                cal_fraca = complementares.intercepto_inclinacao(y_te, p)
                linhas.append({"Conjunto": conj, "Modelo": nome, "Ano teste": t,
                               "Anos ajuste": f"{int(aj['ano'].min())}–{int(aj['ano'].max())}",
                               "TMN ajuste (‰)": 1000 * aj["obito_neonatal"].mean(),
                               "TMN teste (‰)": 1000 * y_te.mean(),
                               "AUROC": roc_auc_score(y_te, m.bruto(te[cols])),
                               "AUPRC": average_precision_score(y_te, m.bruto(te[cols])),
                               "O/E": evaluate.razao_obs_esp(y_te, p),
                               "Intercepto": cal_fraca["intercepto"], "Inclinação": cal_fraca["inclinacao"]})
    return pd.DataFrame(linhas)


# ------------------------------------------------------------------ simulação de padrão-ouro
CAMPOS_DO = ["DTNASC", "SEXO", "CODMUNRES", "PESO", "IDADEMAE", "SEMAGESTAC", "GRAVIDEZ", "PARTO",
             "CODESTAB", *linkage.CONFERENCIA]

# Probabilidade de erro de digitação/transcrição por campo, além da ausência observada na DO real.
# São premissas, variadas em cenários; o estabelecimento "erra" quando o neonato é transferido.
CENARIOS_RUIDO = {
    "baixo": {"PESO": 0.02, "IDADEMAE": 0.01, "SEMAGESTAC": 0.05, "GRAVIDEZ": 0.005, "PARTO": 0.01,
              "CODMUNRES": 0.01, "CODESTAB": 0.10, "ESCMAE": 0.10, "QTDFILMORT": 0.10},
    "realista": {"PESO": 0.05, "IDADEMAE": 0.03, "SEMAGESTAC": 0.10, "GRAVIDEZ": 0.01, "PARTO": 0.03,
                 "CODMUNRES": 0.03, "CODESTAB": 0.20, "ESCMAE": 0.20, "QTDFILMORT": 0.20},
    "alto": {"PESO": 0.10, "IDADEMAE": 0.06, "SEMAGESTAC": 0.20, "GRAVIDEZ": 0.02, "PARTO": 0.06,
             "CODMUNRES": 0.06, "CODESTAB": 0.35, "ESCMAE": 0.35, "QTDFILMORT": 0.35},
}


# Cenários em que parte das DN verdadeiras não está na base (a simulação básica supõe que todas estão).
CENARIOS_SEM_DN = [("realista", 0.05), ("realista", 0.10)]


def ausencia_do(obitos: pd.DataFrame) -> dict:
    """Fração de ausentes de cada campo nas DO reais (depois da padronização das chaves)."""
    k = linkage.chaves(obitos)
    out = {c: float(k[c].isna().mean()) for c in k.columns if c != "DTNASC"}
    for c in linkage.CONFERENCIA:
        out[c] = float(pd.to_numeric(obitos[c], errors="coerce").mask(lambda v: v == 9).isna().mean())
    return out


def _perturbar(s: pd.Series, campo: str, erro: np.ndarray, rng, doadores: np.ndarray) -> pd.Series:
    s = s.copy()
    if not erro.any():
        return s
    if campo in ("PESO", "IDADEMAE", "SEMAGESTAC"):
        v = pd.to_numeric(s[erro], errors="coerce")
        passo = {"PESO": rng.choice([-100, -50, -10, 10, 50, 100], erro.sum()),
                 "IDADEMAE": rng.choice([-1, 1], erro.sum()),
                 "SEMAGESTAC": rng.choice([-2, -1, 1, 2], erro.sum())}[campo]
        largura = 4 if campo == "PESO" else 2
        s[erro] = (v + passo).astype("Int64").astype("string").str.zfill(largura)
    else:
        s[erro] = rng.choice(doadores, erro.sum())  # outro valor observado do mesmo campo
    return s


def simular_obitos(nv_verdadeiros: pd.DataFrame, ausencia: dict, erro: dict, semente: int = SEMENTE) -> pd.DataFrame:
    """DO sintéticas copiadas das DN verdadeiras, com ausência e erros por campo.

    O índice da DN de origem fica em `idx_nv_verdadeiro`, o que dá o padrão-ouro do pareamento.
    """
    rng = np.random.default_rng(semente)
    n = len(nv_verdadeiros)
    do = pd.DataFrame({"idx_nv_verdadeiro": nv_verdadeiros.index.to_numpy()})
    for c in CAMPOS_DO:
        s = nv_verdadeiros[c].astype("string").reset_index(drop=True)
        if c != "DTNASC" and c != "SEXO":
            doadores = s.dropna().to_numpy()
            s = _perturbar(s, c, rng.random(n) < erro.get(c, 0), rng, doadores)
            s[rng.random(n) < ausencia.get(c, 0)] = pd.NA
        do[c] = s
    return do


def avaliar_pareamento(pares: pd.DataFrame, do_sint: pd.DataFrame) -> pd.DataFrame:
    """Pares e fração real de falsos pares por passada, e cobertura de pares verdadeiros."""
    verdade = do_sint.loc[pares["idx_do"], "idx_nv_verdadeiro"].to_numpy()
    p = pares.assign(correto=(pares["idx_nv"].to_numpy() == verdade))
    tab = p.groupby("passada").agg(n_pares=("correto", "size"), falsos_reais=("correto", lambda c: 100 * (1 - c.mean())))
    tab.loc["total", "n_pares"] = len(p)
    tab.loc["total", "falsos_reais"] = 100 * (1 - p["correto"].mean())
    tab["cobertura_verdadeira_%"] = np.nan
    tab.loc["total", "cobertura_verdadeira_%"] = 100 * p["correto"].sum() / len(do_sint)
    return tab


def simulacao_padrao_ouro(nv: pd.DataFrame, obitos: pd.DataFrame, cenarios=CENARIOS_RUIDO,
                          semente: int = SEMENTE) -> dict:
    """Roda o linkage determinístico (e a etapa probabilística) sobre óbitos sintéticos com verdade conhecida.

    Os "óbitos" copiam as DN dos nascidos efetivamente pareados, para ter a mesma distribuição de peso,
    idade gestacional e hospital dos óbitos reais; o restante da base serve de distrator.
    """
    pares_reais = linkage.parear(nv, obitos)
    verdadeiros = nv.loc[pares_reais["idx_nv"]]
    aus = ausencia_do(obitos)
    saida = {"ausencia_do": aus}
    rng = np.random.default_rng(semente)
    todos = {nome: (erro, 0.0) for nome, erro in cenarios.items()}
    todos.update({f"{base} + {int(100 * f)}% sem DN": (cenarios[base], f)
                  for base, f in CENARIOS_SEM_DN if base in cenarios})
    for nome, (erro, frac_sem_dn) in todos.items():
        do_sint = simular_obitos(verdadeiros, aus, erro, semente)
        # DN ausente do pool (sub-registro, residência diferente na DN e na DO): todo par desse óbito é falso.
        sem_dn = rng.choice(verdadeiros.index, int(round(frac_sem_dn * len(verdadeiros))), replace=False)
        pool = nv.drop(sem_dn)
        pares = linkage.parear(pool, do_sint)
        estimado = linkage.taxa_falsos_pares(pool, do_sint, pares)
        tab = avaliar_pareamento(pares, do_sint)
        fs = etapa_probabilistica(pool, do_sint, pares)
        fs_verdade = do_sint.loc[fs["pares"]["idx_do"], "idx_nv_verdadeiro"].to_numpy()
        saida[nome] = {"real": tab, "estimado": estimado, "frac_sem_dn": frac_sem_dn,
                       "fs_pares": len(fs["pares"]),
                       "fs_falsos_reais": 100 * (1 - (fs["pares"]["idx_nv"].to_numpy() == fs_verdade).mean())
                       if len(fs["pares"]) else np.nan}
    return saida


# ------------------------------------------------------------------ etapa probabilística (Fellegi–Sunter)
CAMPOS_FS = ["CODMUNRES", "PESO", "IDADEMAE", "SEMAGESTAC", "GRAVIDEZ", "PARTO", "CODESTAB"]
TOLERANCIA = {"PESO": 50, "IDADEMAE": 1, "SEMAGESTAC": 1}  # nível intermediário de concordância


def comparar(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Nível de concordância por campo: 2 exato, 1 próximo (numéricos), 0 discorda, NaN ausente."""
    a, b = a.reset_index(drop=True), b.reset_index(drop=True)
    out = {}
    for c in CAMPOS_FS:
        falta = (a[c].isna() | b[c].isna()).to_numpy()
        if c in TOLERANCIA:
            d = np.abs(pd.to_numeric(a[c], errors="coerce").to_numpy(float)
                       - pd.to_numeric(b[c], errors="coerce").to_numpy(float))
            nivel = np.where(d == 0, 2, np.where(d <= TOLERANCIA[c], 1, 0)).astype(float)
        else:
            nivel = np.where((a[c] == b[c]).fillna(False).to_numpy(bool), 2, 0).astype(float)
        nivel[falta] = np.nan
        out[c] = nivel
    return pd.DataFrame(out)


def candidatos(nv_k: pd.DataFrame, ob_k: pd.DataFrame) -> pd.DataFrame:
    """Pares candidatos com bloqueio por data de nascimento e sexo."""
    a = nv_k.dropna(subset=["DTNASC", "SEXO"]).rename_axis("idx_nv").reset_index()
    b = ob_k.dropna(subset=["DTNASC", "SEXO"]).rename_axis("idx_do").reset_index()
    return b[["idx_do", "DTNASC", "SEXO"]].merge(a[["idx_nv", "DTNASC", "SEXO"]], on=["DTNASC", "SEXO"])


def em_fellegi_sunter(niveis: pd.DataFrame, u: dict, max_iter: int = 200, tol: float = 1e-6) -> dict:
    """EM com independência condicional; u fixo (pares aleatórios), m e a proporção de pares λ estimados."""
    campos = list(niveis.columns)
    m = {c: {0: 0.05, 1: 0.10, 2: 0.85} for c in campos}
    lam = 1e-3
    X = {c: niveis[c].to_numpy() for c in campos}
    for _ in range(max_iter):
        lm = np.full(len(niveis), np.log(lam))
        lu = np.full(len(niveis), np.log(1 - lam))
        for c in campos:
            for k in (0, 1, 2):
                sel = X[c] == k
                lm[sel] += np.log(max(m[c].get(k, 1e-6), 1e-6))
                lu[sel] += np.log(max(u[c].get(k, 1e-6), 1e-6))
        g = 1 / (1 + np.exp(lu - lm))
        novo_lam = float(g.mean())
        novo_m = {}
        for c in campos:
            ok = ~np.isnan(X[c])
            tot = g[ok].sum()
            novo_m[c] = {k: float((g[ok] * (X[c][ok] == k)).sum() / max(tot, 1e-12)) for k in (0, 1, 2)}
        mudou = abs(novo_lam - lam) + sum(abs(novo_m[c][k] - m[c][k]) for c in campos for k in (0, 1, 2))
        lam, m = novo_lam, novo_m
        if mudou < tol:
            break
    return {"m": m, "u": u, "lambda": lam}


def pesos(niveis: pd.DataFrame, par: dict) -> tuple[np.ndarray, np.ndarray]:
    """Peso de concordância (log2 m/u somado) e probabilidade a posteriori de ser par verdadeiro."""
    w = np.zeros(len(niveis))
    for c in niveis.columns:
        x = niveis[c].to_numpy()
        for k in (0, 1, 2):
            sel = x == k
            w[sel] += np.log2(max(par["m"][c][k], 1e-6) / max(par["u"][c].get(k, 1e-6), 1e-6))
    prior = np.log2(par["lambda"] / (1 - par["lambda"]))
    post = 1 / (1 + 2 ** -(w + prior))
    return w, post


def estimar_u(nv_k: pd.DataFrame, ob_k: pd.DataFrame, n: int = 200_000, semente: int = SEMENTE) -> dict:
    """Distribuição dos níveis entre pares aleatórios do mesmo bloco (quase todos falsos)."""
    cand = candidatos(nv_k, ob_k)
    amostra = cand.sample(min(n, len(cand)), random_state=semente)
    niv = comparar(nv_k.loc[amostra["idx_nv"]].reset_index(drop=True), ob_k.loc[amostra["idx_do"]].reset_index(drop=True))
    return {c: {k: float((niv[c] == k).sum() / max(niv[c].notna().sum(), 1)) for k in (0, 1, 2)} for c in niv}


def m_de_pares(nv_k: pd.DataFrame, ob_k: pd.DataFrame, pares: pd.DataFrame) -> dict:
    """m de cada campo medido nos pares determinísticos em cujas chaves o campo NÃO entra.

    Prefere as passadas estritas (1–2); se o campo está nas duas, usa as tolerantes em que ele fica de fora
    (com a contaminação por falsos pares estimada na primeira parte). Evita o EM, que num bloco com ~1 par
    verdadeiro em centenas pode convergir para outra partição (ex.: mesmo município × outro município).
    """
    m = {}
    for c in CAMPOS_FS:
        fora = [i for i, chave in enumerate(linkage.PASSADAS, start=1) if c not in chave]
        usar = [i for i in fora if i <= linkage.PASSADAS_ESTRITAS] or fora
        p = pares[pares["passada"].isin(usar)]
        niv = comparar(nv_k.loc[p["idx_nv"]], ob_k.loc[p["idx_do"]])[c]
        tot = max(int(niv.notna().sum()), 1)
        m[c] = {k: max(float((niv == k).sum()) / tot, 1e-4) for k in (0, 1, 2)}
        m[c]["passadas"] = usar
    return m


def etapa_probabilistica(nv: pd.DataFrame, obitos: pd.DataFrame, pares_det: pd.DataFrame,
                         limiar_post: float = 0.9, metodo: str = "pares") -> dict:
    """Fellegi–Sunter sobre os óbitos que o determinístico não pareou, com atribuição 1:1 gulosa.

    `metodo="pares"`: m pelos pares determinísticos e λ = óbitos restantes / candidatos (cada óbito tem no
    máximo um par verdadeiro no bloco). `metodo="em"`: m e λ pelo EM, só para comparação.
    """
    nv_k, ob_k = linkage.chaves(nv), linkage.chaves(obitos)
    u = estimar_u(nv_k, ob_k)
    resto_nv = nv_k.drop(pares_det["idx_nv"])
    resto_ob = ob_k.drop(pares_det["idx_do"])
    cand = candidatos(resto_nv, resto_ob)
    vazio = pd.DataFrame(columns=["idx_nv", "idx_do", "passada", "post"])
    if cand.empty:
        return {"pares": vazio, "parametros": None, "candidatos": 0, "post_max": pd.Series(dtype=float)}
    niv = comparar(resto_nv.loc[cand["idx_nv"]].reset_index(drop=True), resto_ob.loc[cand["idx_do"]].reset_index(drop=True))
    if metodo == "em":
        par = em_fellegi_sunter(niv, u)
    else:
        m = m_de_pares(nv_k, ob_k, pares_det)
        par = {"m": {c: {k: m[c][k] for k in (0, 1, 2)} for c in m}, "u": u,
               "lambda": cand["idx_do"].nunique() / len(cand), "passadas_m": {c: m[c]["passadas"] for c in m}}
    _, post = pesos(niv, par)
    post_max = pd.Series(post).groupby(cand["idx_do"].to_numpy()).max()
    cand = cand.assign(post=post).sort_values("post", ascending=False)
    cand = cand[cand["post"] >= limiar_post]
    usados_nv, usados_do, aceitos = set(), set(), []
    for i_nv, i_do, p in cand[["idx_nv", "idx_do", "post"]].itertuples(index=False):
        if i_nv in usados_nv or i_do in usados_do:
            continue
        usados_nv.add(i_nv); usados_do.add(i_do)
        aceitos.append((i_nv, i_do, p))
    pares = pd.DataFrame(aceitos, columns=["idx_nv", "idx_do", "post"]).assign(passada=len(linkage.PASSADAS) + 1)
    return {"pares": pares if len(pares) else vazio, "parametros": par, "candidatos": len(niv),
            "post_max": post_max}


def linkage_probabilistico(nv: pd.DataFrame, obitos: pd.DataFrame, teste: pd.DataFrame, R: dict) -> dict:
    """Etapa FS sobre os óbitos reais não pareados; falsos pares estimados pelo mesmo modelo de mistura."""
    pares_det = linkage.parear(nv, obitos)
    fs = etapa_probabilistica(nv, obitos, pares_det)
    fs_em = etapa_probabilistica(nv, obitos, pares_det, metodo="em")
    todos = pd.concat([pares_det, fs["pares"][["idx_nv", "idx_do", "passada"]]], ignore_index=True)
    todos["passada"] = todos["passada"].astype(int)
    estimado = linkage.taxa_falsos_pares(nv, obitos, todos)
    # Perfil dos óbitos restantes: quantos campos informativos (fora data e sexo) cada um tem na DO.
    resto = linkage.chaves(obitos).drop(pares_det["idx_do"])
    n_campos = resto[CAMPOS_FS].notna().sum(axis=1)
    pm = fs["post_max"]
    # Efeito nos rótulos do teste: os óbitos recuperados passam a positivos, sem reajustar o modelo.
    novos = set(fs["pares"]["idx_nv"]) & set(teste.index)
    y_novo = teste["obito_neonatal"].to_numpy().copy()
    y_novo[teste.index.isin(list(novos))] = 1
    efeito = {}
    for conj in preprocess.CONJUNTOS:
        e = R["predicoes"][(conj, "XGBoost")]["bruto"]
        efeito[conj] = {"AUROC principal": roc_auc_score(teste["obito_neonatal"], e),
                        "AUROC com recuperados": roc_auc_score(y_novo, e),
                        "AUPRC principal": average_precision_score(teste["obito_neonatal"], e),
                        "AUPRC com recuperados": average_precision_score(y_novo, e)}
    n_ob = len(obitos)
    return {"n_obitos": n_ob, "n_deterministico": len(pares_det), "n_probabilistico": len(fs["pares"]),
            "pct_total": 100 * (len(pares_det) + len(fs["pares"])) / n_ob,
            "candidatos": fs["candidatos"], "parametros": fs["parametros"],
            "falsos_estimados": estimado, "recuperados_no_teste": len(novos), "efeito_teste": efeito,
            "n_restantes": len(resto), "restantes_campos": n_campos.value_counts().sort_index(),
            "post_max_faixas": pd.cut(pm, [0, 0.1, 0.5, 0.9, 1.0], include_lowest=True).value_counts().sort_index(),
            "em": {"n_pares": len(fs_em["pares"]), "lambda": fs_em["parametros"]["lambda"] if fs_em["parametros"] else None}}


# ------------------------------------------------------------------ orquestração
PARTES = ["baseline", "idade_obito", "declaracao", "sem_pandemia", "sem_chaves", "janelas",
          "probabilistico", "simulacao"]


def rodar(caminho=RESULTADOS_V2, partes=None) -> dict:
    """Roda todas as análises ou só `partes`, reaproveitando as demais de um resultado já gravado."""
    t0 = time.time()
    partes = list(partes or PARTES)
    V = carregar(caminho) if caminho.exists() and set(partes) != set(PARTES) else {}
    R = experimento.carregar()
    base = pd.read_parquet(download.DADOS / "base_analitica.parquet")
    ajuste, valid, teste = _particao(base)
    nv = download.carregar("DN", download.ANOS_NASCIMENTO)
    do = download.carregar("DO", download.ANOS_OBITO)
    obitos = linkage.obitos_neonatais(do, download.ANOS_NASCIMENTO)
    obitos = obitos[~obitos.duplicated()]
    nv_rot, _ = linkage.rotular(nv, do, download.ANOS_NASCIMENTO)
    _registrar("dados carregados", t0)

    V["ano_teste"] = preprocess.ANO_TESTE
    tarefas = {
        "baseline": lambda: baseline_clinico(ajuste, valid, teste, R),
        "idade_obito": lambda: por_idade_obito(teste, nv_rot["dias_vida"], R),
        "declaracao": lambda: declaracao_pos_obito(nv_rot, teste, R),
        "sem_pandemia": lambda: sem_pandemia(ajuste, valid, teste, R),
        "sem_chaves": lambda: sem_chaves(ajuste, valid, teste, R),
        "janelas": lambda: janelas_deslizantes(base, R),
        "probabilistico": lambda: linkage_probabilistico(nv, obitos, teste, R),
        "simulacao": lambda: simulacao_padrao_ouro(nv, obitos),
    }
    for nome in PARTES:
        if nome in partes:
            V[nome] = tarefas[nome]()
            _registrar(nome, t0)
    if set(partes) == set(PARTES):
        V["tempo_total_s"] = time.time() - t0
    with open(caminho, "wb") as f:
        pickle.dump(V, f)
    return V


def carregar(caminho=RESULTADOS_V2) -> dict:
    with open(caminho, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    import sys
    rodar(partes=sys.argv[1:] or None)
