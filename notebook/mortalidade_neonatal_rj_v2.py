# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Predição de mortalidade neonatal no RJ com dados abertos (SINASC × SIM, 2017–2024) — v2
#
# **Disciplina:** Introdução ao Aprendizado de Máquina para a Saúde (UFF, 2026.2).
#
# **Artigo de referência:** Batista AFM, Diniz CSG, Bonilha EA, Kawachi I, Chiavegatto Filho ADP.
# *Neonatal mortality prediction with routinely collected data: a machine learning approach.*
# BMC Pediatrics 21:322 (2021).
#
# **Perguntas:**
# 1. O desempenho do artigo (XGBoost, AUROC 0,971) se reproduz no estado do RJ só com
#    microdados públicos, que não trazem identificadores?
# 2. Quanto se perde com um modelo que usa apenas informações disponíveis na admissão para o parto?
# 3. O modelo é calibrado e funciona igualmente por raça/cor e escolaridade da mãe e entre capital e interior?
#
# **Versão 2:** as seções 1–5 são o projeto da disciplina (v1, tag `v1.0`); a seção 6 acrescenta o aprofundamento
# para publicação (perguntas Q1–Q7 de um revisor).
#
# Este notebook é a Parte 2 (Demonstração) do projeto e executa o fluxo inteiro:
# download → linkage → pré-processamento → ajuste → avaliação → figuras.

# %% [markdown]
# ## 1. Dependências do ambiente de execução
# As versões exatas estão em `requirements.txt`. Para reproduzir:
# ```bash
# uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
# .venv/bin/python -m src.experimento        # ≈ 1 h com 32 núcleos; grava data/resultados.pkl
# .venv/bin/jupyter nbconvert --to notebook --execute --inplace notebook/mortalidade_neonatal_rj.ipynb
# ```
# O experimento pesado (busca de hiperparâmetros, bootstrap e SHAP) roda em `src/experimento.py`.
# O notebook reaproveita o resultado gravado e só executa o experimento se esse arquivo não existir.

# %%
import sys
from pathlib import Path

RAIZ = Path.cwd().parent if Path.cwd().name == "notebook" else Path.cwd()
sys.path.insert(0, str(RAIZ))

import lightgbm, optuna, shap, sklearn, xgboost
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.metrics import precision_recall_curve, roc_curve

from src import download, evaluate, experimento, linkage, models, preprocess

FIG = RAIZ / "figuras"
FIG.mkdir(exist_ok=True)
pd.set_option("display.width", 160)
pd.set_option("display.precision", 3)

# Paleta categórica validada (slots 1–3 passam em todos os pares, inclusive daltonismo).
COR = {"LR": "#2a78d6", "XGBoost": "#eb6834", "LightGBM": "#1baf7a"}
TINTA, TINTA2, GRADE = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": TINTA2, "axes.labelcolor": TINTA, "xtick.color": TINTA2, "ytick.color": TINTA2,
    "axes.grid": True, "grid.color": GRADE, "grid.linewidth": 0.6, "lines.linewidth": 2,
    "font.size": 10, "legend.frameon": False,
})
NOME_CONJ = {"A_completo": "A · completo", "B_pre_parto": "B · pré-parto"}
print("pandas", pd.__version__, "| scikit-learn", sklearn.__version__, "| xgboost", xgboost.__version__,
      "| lightgbm", lightgbm.__version__, "| optuna", optuna.__version__, "| shap", shap.__version__)

# %% [markdown]
# ## 2. Carregamento do dataset
# São 8 arquivos do SINASC (`DNRJ2017–2024.dbc`) e 9 do SIM (`DORJ2017–2025.dbc`), baixados do FTP público do
# DATASUS: <ftp://ftp.datasus.gov.br/dissemin/publicos/>.
# - O SIM vai até 2025 para incluir os óbitos de quem nasceu no fim de 2024.
# - O SIM 2025 ainda é **preliminar** (pasta `SIM/PRELIM`) e só contribui com esses óbitos.
# - O download é idempotente: arquivos já baixados são reaproveitados.

# %%
arquivos = download.baixar_tudo(RAIZ / "data" / "raw")
nv_bruto = download.carregar("DN", download.ANOS_NASCIMENTO, RAIZ / "data" / "raw")
do_bruto = download.carregar("DO", download.ANOS_OBITO, RAIZ / "data" / "raw")
print(f"SINASC: {len(nv_bruto):,} nascidos vivos, {nv_bruto.shape[1]} colunas")
print(f"SIM:    {len(do_bruto):,} óbitos, {do_bruto.shape[1]} colunas")
print("NUMERODN nos arquivos públicos?", "NUMERODN" in nv_bruto, "NUMERODN" in do_bruto)

# %% [markdown]
# ## 3. Pré-processamento
# ### 3.1 Linkage SINASC × SIM sem identificadores
# Os arquivos públicos não trazem o número da DN. A Declaração de Óbito de menores de 1 ano repete dados
# do nascimento, e o pareamento usa essa redundância:
# - chaves compostas em cascata, da mais restritiva à mais tolerante a campos ausentes;
# - em cada passada, só pares 1:1 são aceitos.
#
# Para estimar os falsos pares, usamos campos que nunca entram nas chaves (escolaridade da mãe e perdas
# fetais anteriores). A concordância na passada 1 serve de referência para pares verdadeiros, e a
# concordância entre pares aleatórios com a mesma data e o mesmo sexo serve de referência para o acaso.

# %%
for i, chave in enumerate(linkage.PASSADAS, 1):
    print(i, " + ".join(chave))

# %%
R = experimento.carregar() if experimento.RESULTADOS.exists() else experimento.rodar()
rel = R["linkage"]
print(f"Óbitos neonatais no SIM: {rel['n_obitos']:,} · pareados: {rel['n_pareados']:,} "
      f"({100 * rel['n_pareados'] / rel['n_obitos']:.1f}%)")
rel["por_ano"]

# %%
rel["falsos_pares"]

# %%
fig, ax = plt.subplots(figsize=(6.5, 3.2))
pa = rel["por_ano"]
ax.plot(pa.index, pa["tmn_sim_por_mil"], marker="o", ms=5, color=COR["LR"], label="SIM (todos os óbitos)")
ax.plot(pa.index, pa["tmn_pareada_por_mil"], marker="o", ms=5, color=COR["XGBoost"], label="Óbitos pareados")
for x, y, t in zip(pa.index, pa["tmn_pareada_por_mil"], pa["pct_pareado"]):
    ax.annotate(f"{t:.0f}%", (x, y), textcoords="offset points", xytext=(0, -14), ha="center",
                fontsize=8, color=TINTA2)
ax.set_ylim(6, 10); ax.set_ylabel("Óbitos neonatais por mil NV")
ax.set_title("Taxa de mortalidade neonatal, RJ: SIM × base pareada (rótulo: % pareado)", loc="left", fontsize=10)
ax.legend(loc="upper left", ncol=2)
fig.savefig(FIG / "linkage_por_ano.png")

# %% [markdown]
# ### 3.2 Limpeza, exclusões e variáveis
# - **Numéricas:** códigos de "ignorado" (99, ou 9999 no peso) viram ausentes. Mantemos idade da mãe e
#   do pai, histórico obstétrico, semanas de gestação, consultas e mês de início do pré-natal, Apgar e peso.
# - **Categóricas:** o "9" (ignorado) e o campo vazio viram categorias próprias, porque o não
#   preenchimento da DN também informa sobre a qualidade da assistência.
# - **Exclusão do artigo:** idade gestacional fora de 15–45 semanas. Nenhum registro do RJ caiu nela,
#   porque o SINASC já valida esse campo.
# - **Imputação e encoding:** para o XGBoost e o LightGBM, os ausentes e as categorias são tratados
#   nativamente pelas árvores. Na regressão logística, imputação pela mediana com indicador de ausência,
#   padronização e one-hot são ajustados só no treino.

# %%
R["fluxo"]

# %%
base = pd.read_parquet(RAIZ / "data" / "base_analitica.parquet")
print("Conjunto A (completo):", len(preprocess.VARS_COMPLETO), "variáveis")
print("Conjunto B (pré-parto):", len(preprocess.VARS_PRE_PARTO), "variáveis:",
      ", ".join(preprocess.ROTULOS[v] for v in preprocess.VARS_PRE_PARTO))
aus = (base[[*preprocess.NUMERICAS]].isna().mean() * 100).round(1).rename(preprocess.ROTULOS)
print("\n% ausente nas numéricas:\n", aus.sort_values(ascending=False).to_string())

# %% [markdown]
# ### 3.3 Estatística descritiva
# Taxa de mortalidade neonatal (TMN, por mil nascidos vivos) nas principais variáveis da DN.

# %%
def tmn(df, col, rotulos=None):
    g = df.groupby(col, observed=True)["obito_neonatal"].agg(["size", "sum", "mean"])
    g.columns = ["NV", "óbitos", "TMN (‰)"]
    g["TMN (‰)"] *= 1000
    return g.rename(index=rotulos) if rotulos else g

faixa_peso = pd.cut(base["PESO"], [0, 1500, 2500, 4000, 8000], right=False,
                    labels=["< 1.500 g", "1.500–2.499 g", "2.500–3.999 g", "≥ 4.000 g"])
faixa_ig = pd.cut(base["SEMAGESTAC"], [0, 32, 37, 42, 46], right=False,
                  labels=["< 32 sem", "32–36 sem", "37–41 sem", "≥ 42 sem"])
desc = pd.concat({
    "Peso ao nascer": tmn(base.assign(f=faixa_peso), "f"),
    "Idade gestacional": tmn(base.assign(f=faixa_ig), "f"),
    "Apgar 5º min < 7": tmn(base.assign(f=np.where(base["APGAR5"] < 7, "sim", "não")), "f"),
    "Anomalia congênita": tmn(base, "IDANOMAL", {"1": "sim", "2": "não", "9": "ignorado", "ausente": "ausente"}),
    "Raça/cor da mãe": tmn(base, "raca_mae"),
    "Escolaridade da mãe": tmn(base, "escol_mae"),
})
desc.to_csv(FIG / "descritiva_tmn.csv")
desc

# %% [markdown]
# ### 3.4 Amostras de treino, validação e teste (validação temporal)
# | Uso | Anos | Papel |
# |---|---|---|
# | Ajuste | 2017–2022 | treino dos modelos |
# | Validação | 2023 | busca de hiperparâmetros (Optuna), *early stopping*, recalibração isotônica e escolha do limiar |
# | Teste | 2024 | avaliação final, usada uma única vez |
#
# O esquema espelha o do artigo, que treinou em 2012–2016 e testou em 2017. A pergunta avaliada é se um
# modelo treinado no passado funciona no ano seguinte.

# %%
pd.DataFrame(R["amostras"]).T

# %% [markdown]
# ## 4. Modelos
# **Regressão logística:** $P(y=1\mid\mathbf{x}) = \sigma(\beta_0 + \boldsymbol\beta^\top\mathbf{x})$, com
# $\sigma(z) = 1/(1+e^{-z})$. Ajuste por máxima verossimilhança com penalização L2 ($C$ escolhido na validação) e
# `class_weight="balanced"`.
#
# **XGBoost / LightGBM:** $F(\mathbf{x}) = \sum_{k=1}^{K} f_k(\mathbf{x})$, em que cada $f_k$ é uma árvore de regressão, e
# $P(y=1\mid\mathbf{x}) = \sigma(F(\mathbf{x}))$. As árvores entram em sequência, e a árvore $k$ minimiza a expansão
# de 2ª ordem da log-loss mais $\Omega(f) = \gamma T + \tfrac12\lambda\lVert w\rVert^2 + \alpha\lVert w\rVert_1$.
#
# **Peso de classe (`scale_pos_weight`):** é tratado como hiperparâmetro, com três opções: nenhum,
# $\sqrt{n_0/n_1}$ ou $n_0/n_1$. Como qualquer peso distorce as probabilidades, uma **regressão isotônica**
# ajustada em 2023 recalibra as saídas. Ela é monótona, então não altera a ordenação.
#
# **Complexidade:** o XGBoost com histogramas custa $O(K \cdot n \cdot d)$ por ajuste ($K$ árvores, $n$ registros,
# $d$ variáveis), e a predição custa $O(K \cdot \text{profundidade})$ por registro. A regressão logística custa
# $O(n \cdot d')$ por iteração do L-BFGS, com $d'$ colunas após o one-hot.
#
# ### 4.1 Hiperparâmetros
# A busca bayesiana (TPE, Optuna) teve 30 tentativas para o XGBoost e 20 para o LightGBM. O objetivo
# foi a AUPRC na validação de 2023, com *early stopping* de 50 árvores.

# %%
params = pd.DataFrame({f"{NOME_CONJ[c]} · {m}": pd.Series(v) for (c, m), v in R["params"].items()})
params.to_csv(FIG / "hiperparametros.csv")
params

# %%
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
for ax, conj in zip(axes, preprocess.CONJUNTOS):
    for nome, cor in [("XGBoost", COR["XGBoost"]), ("LightGBM", COR["LightGBM"])]:
        t = R["estudos"][(conj, nome)]
        ax.plot(t["number"], t["value"].cummax(), color=cor, label=f"{nome} (melhor até aqui)")
        ax.scatter(t["number"], t["value"], s=12, color=cor, alpha=0.35, linewidths=0)
    ax.set_title(NOME_CONJ[conj], loc="left", fontsize=10); ax.set_xlabel("Tentativa do Optuna")
axes[0].set_ylabel(f"AUPRC na validação ({preprocess.ANO_VALIDACAO})"); axes[0].legend(loc="lower right")
fig.savefig(FIG / "optuna.png")

# %% [markdown]
# ## 5. Execução
# ### 5.1 Curvas por iteração (XGBoost final)
# Log-loss e AUPRC a cada árvore, no ajuste (2017–2022) e na validação (2023). A linha vertical marca a
# árvore escolhida pelo *early stopping*. Log-loss e AUPRC ficam em painéis separados porque têm escalas diferentes.

# %%
fig, axes = plt.subplots(2, 2, figsize=(9, 5.5), sharex="col")
for j, conj in enumerate(preprocess.CONJUNTOS):
    c = R["curvas"][conj]
    melhor = R["params"][(conj, "XGBoost")]["n_arvores"] - 1
    for i, (met, rot) in enumerate([("logloss", "Log-loss"), ("aucpr", "AUPRC")]):
        ax = axes[i, j]
        ax.plot(c.index, c[f"treino_{met}"], color=COR["LR"], label=f"ajuste {download.ANOS_NASCIMENTO[0]}–{preprocess.ANO_VALIDACAO - 1}")
        ax.plot(c.index, c[f"valid_{met}"], color=COR["XGBoost"], label=f"validação {preprocess.ANO_VALIDACAO}")
        ax.axvline(melhor, color=TINTA2, lw=1, ls="--")
        ax.set_ylabel(rot)
        if i == 0:
            ax.set_title(NOME_CONJ[conj], loc="left", fontsize=10)
        else:
            ax.set_xlabel("Árvores")
axes[0, 0].legend()
fig.savefig(FIG / "curvas_iteracao.png")

# %% [markdown]
# ### 5.2 Desempenho no teste (2024)
# O limiar de classificação foi escolhido em 2023 para maximizar o F1 e só depois aplicado a 2024. Os IC 95%
# vêm de 500 reamostras bootstrap estratificadas.
#
# A regressão isotônica é monótona, mas é uma função em degraus e cria empates. Por isso as métricas de
# **ordenação** (AUROC, AUPRC e óbitos no top 5%) usam o escore bruto do modelo. As métricas que dependem do
# **valor** da probabilidade (Brier, O/E e as métricas no limiar) usam a probabilidade recalibrada.

# %%
def tabela_metricas(R):
    linhas = []
    for (conj, nome), t in R["metricas"].items():
        linha = {"Conjunto": NOME_CONJ[conj], "Modelo": nome}
        for m, r in t.iterrows():
            linha[m] = f"{r.valor:.3f} ({r.ic_inf:.3f}–{r.ic_sup:.3f})"
        linhas.append(linha)
    return pd.DataFrame(linhas).set_index(["Conjunto", "Modelo"])

tab = tabela_metricas(R)
tab.to_csv(FIG / "metricas_teste.csv")
tab.T

# %% [markdown]
# **Comparação com o artigo (XGBoost, conjunto completo):**

# %%
artigo = {"AUROC": 0.971, "AUPRC": 0.586, "F1": 0.548, "Precisão": 0.729, "Sensibilidade": 0.440,
          "Especificidade": 0.99, "VPN": 0.997, "Óbitos no top 5%": 0.90}
nosso = R["metricas"][("A_completo", "XGBoost")]
comp = pd.DataFrame({"Batista et al. (SP, 2017)": artigo, f"Este trabalho (RJ, {preprocess.ANO_TESTE})": nosso["valor"]}).loc[list(artigo)]
comp.to_csv(FIG / "comparacao_artigo.csv")
comp.round(3)

# %%
pd.DataFrame(R["deltas"]).T

# %%
y_te = R["y_teste"]
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
for (conj, nome), pr in R["predicoes"].items():
    p = pr["bruto"]  # curvas de ordenação: escore bruto (a isotônica cria empates)
    ls = "-" if conj == "A_completo" else "--"
    fpr, tpr, _ = roc_curve(y_te, p)
    prec, rec, _ = precision_recall_curve(y_te, p)
    auc = R["metricas"][(conj, nome)].loc["AUROC", "valor"]
    ap = R["metricas"][(conj, nome)].loc["AUPRC", "valor"]
    axes[0].plot(fpr, tpr, ls=ls, color=COR[nome], lw=1.6, label=f"{nome} · {NOME_CONJ[conj][0]} ({auc:.3f})")
    axes[1].plot(rec, prec, ls=ls, color=COR[nome], lw=1.6, label=f"{nome} · {NOME_CONJ[conj][0]} ({ap:.3f})")
axes[0].plot([0, 1], [0, 1], color=GRADE, lw=1)
axes[0].set(xlabel="1 − especificidade", ylabel="Sensibilidade", title="ROC")
axes[1].axhline(y_te.mean(), color=GRADE, lw=1)
axes[1].set(xlabel="Sensibilidade (recall)", ylabel="Precisão", title="Precisão × sensibilidade")
axes[0].legend(title="Linha cheia: A · tracejada: B", fontsize=8, title_fontsize=8, loc="lower right")
axes[1].legend(fontsize=8, loc="upper right")
for ax in axes:
    ax.title.set_fontsize(10); ax.title.set_ha("left"); ax.title.set_x(0)
fig.savefig(FIG / "roc_pr.png")

# %% [markdown]
# **Captura de óbitos por fração da população priorizada.** Ao ordenar os recém-nascidos pelo risco
# previsto, a curva mostra que fração dos óbitos está entre os x% de maior risco. O artigo reporta mais de
# 90% dos óbitos nos 5% de maior risco.

# %%
fig, ax = plt.subplots(figsize=(6, 3.6))
for conj, ls in [("A_completo", "-"), ("B_pre_parto", "--")]:
    p = R["predicoes"][(conj, "XGBoost")]["bruto"]
    ordem = np.argsort(-p, kind="stable")
    capt = np.cumsum(y_te[ordem]) / y_te.sum()
    frac = np.arange(1, len(p) + 1) / len(p)
    ax.plot(frac * 100, capt * 100, ls=ls, color=COR["XGBoost"], label=f"XGBoost · {NOME_CONJ[conj]}")
    i5 = int(0.05 * len(p)) - 1
    ax.annotate(f"{capt[i5] * 100:.0f}%", (5, capt[i5] * 100), textcoords="offset points",
                xytext=(8, -4), fontsize=8, color=TINTA2)
ax.axvline(5, color=TINTA2, lw=1, ls=":")
ax.set(xlim=(0, 30), ylim=(0, 100), xlabel="% dos nascidos vivos (maior risco primeiro)",
       ylabel="% dos óbitos neonatais capturados")
ax.legend(loc="lower right")
fig.savefig(FIG / "captura_top.png")

# %% [markdown]
# ### 5.3 Calibração
# Observado × previsto por decil de risco no teste, antes e depois da recalibração isotônica. A escala é
# logarítmica porque o risco vai de menos de 0,1% a mais de 50%. O/E é a razão entre óbitos observados e
# esperados, e o ideal é 1.

# %%
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
for ax, conj in zip(axes, preprocess.CONJUNTOS):
    cal = R["calibracao"][conj]
    for tipo, cor in [("bruto", COR["LR"]), ("calibrado", COR["XGBoost"])]:
        t = cal[tipo]
        ax.plot(t["previsto"], t["observado"], marker="o", ms=5, color=cor,
                label=f"{tipo} (O/E = {cal[f'OE_{tipo}']:.2f})")
    lim = [1e-4, 1]
    ax.plot(lim, lim, color=GRADE, lw=1)
    ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="Risco previsto (média do decil)")
    ax.set_title(f"XGBoost · {NOME_CONJ[conj]}", loc="left", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
axes[0].set_ylabel("Mortalidade observada")
fig.savefig(FIG / "calibracao.png")

# %% [markdown]
# ### 5.4 Interpretação (SHAP)
# Os valores SHAP (TreeSHAP exato, seção 5.10) foram calculados numa amostra de 20.000 nascimentos de 2024. O
# gráfico mostra a importância média |SHAP| das 12 principais variáveis. No artigo, as cinco primeiras
# foram Apgar 5, peso, Apgar 1, anomalia congênita e idade gestacional.

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
imp_todas = {}
for ax, conj in zip(axes, preprocess.CONJUNTOS):
    s = R["shap"][conj]
    imp = pd.Series(np.abs(s["valores"]).mean(0), index=s["X"].columns).rename(preprocess.ROTULOS)
    imp = imp.sort_values().tail(12)
    imp_todas[conj] = imp[::-1]
    ax.barh(imp.index, imp.values, color=COR["XGBoost"], height=0.6)
    ax.set_title(f"XGBoost · {NOME_CONJ[conj]}", loc="left", fontsize=10)
    ax.set_xlabel("média |SHAP| (log-odds)"); ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(FIG / "shap_importancia.png")
pd.DataFrame({NOME_CONJ[c]: v.index[:5].tolist() for c, v in imp_todas.items()})

# %%
s = R["shap"]["A_completo"]
shap.summary_plot(s["valores"], s["X"].rename(columns=preprocess.ROTULOS), max_display=12, show=False,
                  plot_size=(7, 5))
plt.gcf().savefig(FIG / "shap_beeswarm_A.png")
plt.show()

# %% [markdown]
# ### 5.5 Subgrupos: discriminação e calibração
# Resultados do XGBoost recalibrado no teste de 2024, com IC 95% de 500 reamostras bootstrap dentro de cada
# subgrupo. Subgrupos com menos de 20 óbitos (mães amarelas e indígenas) foram omitidos. Se os IC se
# sobrepõem, a diferença entre os subgrupos não é distinguível do acaso.

# %%
teste = base[base["ano"] == preprocess.ANO_TESTE]
assert (teste["obito_neonatal"].to_numpy() == y_te).all()  # mesma ordem das predições gravadas
GRUPOS = {"raca_mae": "Raça/cor da mãe", "escol_mae": "Escolaridade da mãe", "capital": "Residência"}
ORDEM = {"raca_mae": ["Branca", "Parda", "Preta"],
         "escol_mae": ["Fund. incompleto ou menos", "Médio", "Superior"], "capital": ["Capital", "Interior"]}
sub = []
for conj in preprocess.CONJUNTOS:
    pr = R["predicoes"][(conj, "XGBoost")]
    d = teste[list(GRUPOS)].assign(y=y_te, p=pr["calibrado"], escore=pr["bruto"])
    d["capital"] = d["capital"].map({1.0: "Capital", 0.0: "Interior"})
    for g in GRUPOS:
        t = evaluate.por_subgrupo(d, g, col_escore="escore", n_bootstrap=500).rename(columns={g: "Subgrupo"})
        sub.append(t.assign(Conjunto=NOME_CONJ[conj], Variável=GRUPOS[g]))
sub = pd.concat(sub).set_index(["Conjunto", "Variável", "Subgrupo"])
sub.to_csv(FIG / "subgrupos.csv")
sub.round(3)

# %%
linhas = [(GRUPOS[g], nome) for g in GRUPOS for nome in ORDEM[g]]
ypos, y, rotulos, blocos = {}, 0, [], []
for i, (var, nome) in enumerate(linhas):
    if i and var != linhas[i - 1][0]:
        y += 0.8  # espaço entre blocos
        blocos.append(y - 0.9)
    ypos[(var, nome)] = y; rotulos.append(nome); y += 1
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
for ax, met, ref in [(axes[0], "AUROC", None), (axes[1], "O/E", 1.0)]:
    for conj, cor, desl in [("A · completo", COR["XGBoost"], -0.17), ("B · pré-parto", COR["LR"], 0.17)]:
        t = sub.loc[conj]
        yy = [ypos[k] + desl for k in t.index]
        ax.errorbar(t[met], yy, xerr=[t[met] - t[f"{met}_inf"], t[f"{met}_sup"] - t[met]], fmt="o", ms=6,
                    color=cor, ecolor=cor, elinewidth=1.5, capsize=0, label=conj)
    if ref:
        ax.axvline(ref, color=TINTA2, lw=1, ls="--")
    for yb in blocos:
        ax.axhline(yb, color=GRADE, lw=1)
    ax.set_title(met + (" (ideal = 1)" if ref else ""), loc="left", fontsize=10); ax.grid(axis="y", visible=False)
axes[0].set_yticks(list(ypos.values()), rotulos); axes[0].invert_yaxis()
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, bbox_to_anchor=(0.55, 1.03))
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(FIG / "subgrupos.png")

# %% [markdown]
# ### 5.6 Calibração fraca: intercepto e inclinação
# Seguimos Van Calster et al. (2019).
# - **Intercepto** (*calibration-in-the-large*): ideal 0; um valor positivo indica que o modelo subestima o risco.
# - **Inclinação:** ideal 1; um valor menor que 1 indica previsões extremas demais.
#
# A TMN de cada amostra aparece abaixo. Se ela muda entre o treino e o teste (*drift*), isso desloca
# o O/E para cima de 1 mesmo num modelo bem calibrado no passado.

# %%
C = R["complementares"]
cal_tab = pd.DataFrame({(NOME_CONJ[c], t): v for c, d in C["calibracao"].items() for t, v in d.items()}).T
cal_tab.to_csv(FIG / "calibracao_intercepto_inclinacao.csv")
print("TMN por amostra (‰):", {k: round(v, 2) for k, v in C["tmn_por_amostra"].items()})
cal_tab.round(3)

# %% [markdown]
# ### 5.7 Curva de calibração e utilidade clínica
# - **Calibração** (painel da esquerda): mortalidade observada × prevista em 20 faixas de risco (quantis), em
#   escala log, para a probabilidade bruta e a recalibrada. Intercepto e inclinação são calibração "fraca"; a
#   curva é o nível recomendado por Van Calster et al.
# - **Curva de decisão** (painel da direita): benefício líquido $NB(p_t) = \frac{VP}{n} - \frac{FP}{n}\cdot\frac{p_t}{1-p_t}$
#   (Vickers e Elkin, 2006).
#   - Com prevalência de 0,8%, "tratar todos" tem benefício negativo acima de $p_t$ ≈ 0,8%, então superá-la é
#     trivial. A comparação relevante é com **tratar nenhum**.
#   - A faixa sombreada, 2%–10%, corresponde a aceitar de 9 a 49 falsos alarmes por óbito identificado. É uma
#     faixa plausível para decidir uma transferência *in utero*, que tem custo e risco, mas é muito menos grave
#     do que um óbito evitável sem UTI.

# %%
def curva_cal(y, p, n=20):
    d = pd.DataFrame({"y": y, "p": p})
    d["f"] = pd.qcut(d["p"].rank(method="first"), n, labels=False)
    return d.groupby("f").agg(previsto=("p", "mean"), observado=("y", "mean"))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.9))
for conj, cor in [("A_completo", COR["XGBoost"]), ("B_pre_parto", COR["LR"])]:
    for t, ls, mk in [("calibrado", "-", "o"), ("bruto", ":", "")]:
        c = curva_cal(y_te, R["predicoes"][(conj, "XGBoost")][t])
        ax1.plot(c["previsto"], c["observado"], ls=ls, marker=mk, ms=3.5, color=cor, lw=1.6,
                 label=f"{NOME_CONJ[conj]} · {t}")
lim = [3e-4, 0.6]
ax1.plot(lim, lim, color=GRADE, lw=1)
ax1.set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="Risco previsto (média da faixa)",
        ylabel="Mortalidade observada")
ax1.set_title("Curva de calibração (20 faixas)", loc="left", fontsize=10); ax1.legend(fontsize=7.5)
ax2.axvspan(2, 10, color=GRADE, alpha=0.5, lw=0)
for conj, cor in [("A_completo", COR["XGBoost"]), ("B_pre_parto", COR["LR"])]:
    d = C["decisao"][conj]; d = d[d["limiar"] <= 0.2]
    ax2.plot(d["limiar"] * 100, d["modelo"] * 1000, color=cor, label=f"XGBoost · {NOME_CONJ[conj]}")
d = C["decisao"]["A_completo"]; d = d[d["limiar"] <= 0.2]
ax2.plot(d["limiar"] * 100, d["tratar todos"] * 1000, color=TINTA2, ls="--", lw=1.2, label="tratar todos")
ax2.axhline(0, color=TINTA, lw=1, label="tratar nenhum")
ax2.set(xlabel="Limiar de risco $p_t$ (%)", ylabel="Benefício líquido (por mil NV)", ylim=(-1, 9), xlim=(0, 20))
ax2.text(6, 8.3, "faixa clínica\n(9–49 falsos alarmes\npor óbito)", ha="center", fontsize=7.5, color=TINTA2, va="top")
ax2.set_title("Curva de decisão", loc="left", fontsize=10); ax2.legend(fontsize=7.5, loc="upper right", bbox_to_anchor=(1, 0.8))
fig.tight_layout()
fig.savefig(FIG / "calibracao_decisao.png")
pd.concat({NOME_CONJ[c]: d.set_index("limiar").loc[[0.02, 0.05, 0.1]] for c, d in C["decisao"].items()}).round(5)

# %% [markdown]
# ### 5.8 Viés do linkage: óbitos pareados × não pareados
# Os óbitos não pareados continuam na base rotulados como sobreviventes. Para ver quem eles são, comparamos as
# características que a própria Declaração de Óbito informa, sem depender do pareamento. As chaves exigem peso,
# idade da mãe e semanas preenchidos, então um óbito com esses campos ausentes tem menos chance de ser pareado.

# %%
vl = C["vies_linkage"]
vl["perfil"].to_csv(FIG / "vies_linkage_perfil.csv")
vl["por_subgrupo"].to_csv(FIG / "vies_linkage_subgrupos.csv", index=False)
vl["perfil"].round(1)

# %%
vl["por_subgrupo"].round(1)

# %% [markdown]
# ### 5.9 Sensibilidade ao linkage
# O XGBoost foi refeito com os mesmos hiperparâmetros e rótulos alternativos **no treino e na validação**.
# - **Estrito:** os nascimentos pareados nas passadas 3–6 saem.
# - **Pior caso:** esses nascimentos ficam, mas rotulados como sobreviventes.
#
# A avaliação é no **mesmo teste de 2024 completo**, com os rótulos principais. Assim, a diferença de AUROC em
# relação ao modelo principal pode ser testada por bootstrap pareado.

# %%
linhas = []
for (conj, cen), v in C["sensibilidade"].items():
    m, d = v["metricas"]["valor"], v["delta_auroc"]
    linhas.append({"Conjunto": NOME_CONJ[conj], "Cenário": cen, "Óbitos no treino": v["positivos_treino"],
                   "AUROC": m["AUROC"], "AUPRC": m["AUPRC"], "Top 5%": m["Óbitos no top 5%"], "O/E": m["O/E"],
                   "ΔAUROC vs principal": d["delta"], "IC inf": d["ic_inf"], "IC sup": d["ic_sup"]})
for conj in preprocess.CONJUNTOS:
    m = R["metricas"][(conj, "XGBoost")]["valor"]
    linhas.append({"Conjunto": NOME_CONJ[conj], "Cenário": "principal", "AUROC": m["AUROC"], "AUPRC": m["AUPRC"],
                   "Top 5%": m["Óbitos no top 5%"], "O/E": m["O/E"]})
sens = pd.DataFrame(linhas).sort_values(["Conjunto", "Cenário"]).set_index(["Conjunto", "Cenário"])
sens.to_csv(FIG / "sensibilidade_linkage.csv")
sens.round(4)

# %% [markdown]
# ### 5.10 Explicabilidade (XAI)
# O SHAP (TreeSHAP exato) decompõe o log-odds previsto de cada nascimento em contribuições aditivas das
# variáveis: $\text{logit}\,\hat p = \phi_0 + \sum_j \phi_j$. A análise tem quatro partes:
# 1. **Global:** importância média |SHAP| (seção 5.4).
# 2. **Dependência:** SHAP em função do valor da variável. O efeito aprendido faz sentido clinicamente?
# 3. **Local:** explicação de casos individuais, isto é, por que o modelo acertou ou errou.
# 4. **Estabilidade e atributos sensíveis:** o ranking muda entre algoritmos ou amostras? Quanto raça/cor e
#    escolaridade pesam nas explicações?

# %%
X_ = C["xai"]
fig, axes = plt.subplots(2, 3, figsize=(11, 5.6))
for i, conj in enumerate(preprocess.CONJUNTOS):
    for j, (v, d) in enumerate(X_["dependencia"][conj].items()):
        ax = axes[i, j]
        d = d.dropna()
        amostra = d.sample(min(6000, len(d)), random_state=0)
        ax.scatter(amostra["x"], amostra["shap"], s=4, alpha=0.25, color=COR["LR"], linewidths=0)
        m = d.groupby(d["x"].round(0 if v != "PESO" else -2))["shap"].median()
        m = m[d.groupby(d["x"].round(0 if v != "PESO" else -2)).size() >= 30]
        ax.plot(m.index, m.values, color=COR["XGBoost"], lw=2)
        ax.axhline(0, color=GRADE, lw=1)
        ax.set_xlabel(preprocess.ROTULOS[v]); ax.set_ylabel("SHAP (log-odds)" if j == 0 else "")
        ax.set_title(NOME_CONJ[conj] if j == 0 else "", loc="left", fontsize=10)
fig.tight_layout()
fig.savefig(FIG / "xai_dependencia.png")

# %% [markdown]
# **Explicações locais.** São três casos do teste de 2024, no modelo completo: o óbito de maior risco previsto,
# um óbito que o modelo não sinalizou (falso negativo mediano) e o sobrevivente de maior risco previsto (falso
# positivo). O ponto de partida é o risco médio ($\phi_0$), e cada barra soma ou subtrai log-odds.

# %%
CODIGOS = {"Anomalia congênita": {"1": "sim", "2": "não", "9": "ignorado"},
           "Sexo": {"1": "masculino", "2": "feminino"}, "Tipo de gravidez": {"1": "única", "2": "dupla"},
           "Tipo de parto": {"1": "vaginal", "2": "cesáreo"}}


def legivel(var, valor):
    if var in CODIGOS:
        return CODIGOS[var].get(str(valor), str(valor))
    try:
        v = float(valor)
    except ValueError:
        return str(valor)
    txt = f"{v:.0f}"
    return txt + " g" if var == "Peso ao nascer" else (txt + " sem" if var == "Semanas de gestação" else txt)


loc = X_["locais"]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
for ax, (nome, c) in zip(axes, loc["casos"].items()):
    t = c["top"].iloc[::-1]
    rot = [f"{r.variavel} = {legivel(r.variavel, r.valor)}" for r in t.itertuples()]
    ax.barh(rot, t["shap"], color=[COR["XGBoost"] if v > 0 else COR["LR"] for v in t["shap"]], height=0.6)
    ax.axvline(0, color=TINTA2, lw=1)
    desfecho = "óbito" if c["obito"] else "sobreviveu"
    ax.set_title(f"{nome}\nrisco previsto {100 * c['prob_calibrada']:.1f}% · {desfecho}".replace(".", ","),
                 loc="left", fontsize=9)
    ax.set_xlabel("contribuição SHAP (log-odds)"); ax.tick_params(axis="y", labelsize=8); ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(FIG / "xai_local.png")
print(f"Risco de base φ0: {100 * loc['base_prob']:.2f}% | erro de aditividade máx.:",
      max(c["aditividade_erro"] for c in loc["casos"].values()))
pd.concat({k: v["top"].set_index("variavel") for k, v in loc["casos"].items()}).round(3)

# %% [markdown]
# **Estabilidade e atributos sensíveis.**

# %%
est = pd.DataFrame({NOME_CONJ[c]: {k: v for k, v in e.items() if k != "importancia_lgbm"}
                    for c, e in X_["estabilidade"].items()}).T
sens_attr = pd.DataFrame({NOME_CONJ[c]: {preprocess.ROTULOS[k]: v for k, v in s_["fracao_importancia"].items()}
                          for c, s_ in X_["sensiveis"].items()}).T
est.to_csv(FIG / "xai_estabilidade.csv"); sens_attr.to_csv(FIG / "xai_sensiveis.csv")
display(est.round(3)); display((100 * sens_attr).round(1))
X_["sensiveis"]["B_pre_parto"]["RACACORMAE"].rename(index={"1": "Branca", "2": "Preta", "3": "Amarela", "4": "Parda",
                                                            "5": "Indígena", "ausente": "Ignorada"}).round(3)

# %% [markdown]
# ### 5.11 Avaliação das premissas
# - **Colinearidade (regressão logística):** as contagens do histórico obstétrico são quase uma soma umas das
#   outras (gestações ≈ partos + perdas). O VIF mede quanto isso infla a variância dos coeficientes. A
#   colinearidade afeta a interpretação dos coeficientes, mas não a predição. A penalização L2 estabiliza o ajuste,
#   e as árvores não dependem dessa premissa.
# - **Mecanismo dos ausentes:** se a TMN difere muito entre registros com e sem o campo, a ausência não é
#   aleatória (MCAR descartado). Ela é informativa, e por isso usamos indicadores de ausência ou a categoria
#   "ignorado", em vez de só imputar.
# - **Independência:** gêmeos compartilham mãe e gestação. Medimos quanto eles pesam e se a AUROC muda sem eles.

# %%
C["colinearidade"].round(2)

# %%
C["ausencia"].round(2)

# %%
pd.Series(C["gemelares"]).round(4)

# %% [markdown]
# ## 6. Parte 2 (v2): aprofundamento para publicação
# A primeira parte (seções 1–5) é o projeto da disciplina. Esta parte responde às perguntas que um revisor de
# revista faria, e que ficaram em aberto no parecer:
#
# | # | Pergunta | Análise |
# |---|---|---|
# | Q1 | Um escore simples faz o mesmo que o XGBoost? | LR só com peso, idade gestacional e Apgar 5 (e + anomalia) |
# | Q2 | Há vazamento na DN (campos preenchidos depois do óbito)? | AUROC por idade no óbito: dia 0, 1–6 e 7–27 dias |
# | Q3 | Os anos de pandemia distorcem o treino? | Reajuste sem 2020–2021 |
# | Q4 | O resultado depende das variáveis que também são chave do linkage? | Reajuste sem peso, idade da mãe, semanas, gravidez e parto |
# | Q5 | A calibração se degrada com o tempo? | Janelas deslizantes: modelo reajustado × congelado |
# | Q6 | O estimador de falsos pares acerta? | Simulação com padrão-ouro conhecido |
# | Q7 | Uma etapa probabilística recupera os óbitos não pareados? | Fellegi–Sunter sobre o resíduo do determinístico |
#
# Todas as análises usam os hiperparâmetros do experimento principal e avaliam no **mesmo teste**, com Δ por
# bootstrap pareado. O código está em `src/aprofundamento.py` (`python -m src.aprofundamento`, ≈ 1 h), que
# grava `data/resultados_v2.pkl`.

# %%
from src import aprofundamento

caminho_v2 = RAIZ / "data" / "resultados_v2.pkl"
V = aprofundamento.carregar(caminho_v2) if caminho_v2.exists() else aprofundamento.rodar(caminho_v2)
print(f"teste em {V['ano_teste']}; execução completa do aprofundamento: {V['tempo_total_s'] / 60:.0f} min")


def resumo(m):
    """Linha com AUROC, AUPRC, top 5% e O/E de uma tabela `com_ic`."""
    v = m["valor"]
    return {"AUROC": v["AUROC"], "AUPRC": v["AUPRC"], "Top 5%": v["Óbitos no top 5%"], "O/E": v["O/E"]}


def fmt_delta(d):
    return f"{d['delta']:+.4f} ({d['ic_inf']:+.4f} a {d['ic_sup']:+.4f})"


# As figuras v2_*.png e as macros do relatório v2 saem do mesmo gerador, para não divergirem.
import runpy
_ = runpy.run_path(str(RAIZ / "relatorio_v2" / "gerar_numeros_v2.py"))


# %% [markdown]
# ### 6.1 Q1: baseline clínico simples
# Se três variáveis chegam perto do XGBoost com 28, o valor do trabalho está na calibração, na equidade e no
# linkage, e não no algoritmo. Um revisor vai perguntar isso; é melhor responder antes.

# %%
linhas = {"XGBoost (28 variáveis)": resumo(R["metricas"][("A_completo", "XGBoost")]),
          "LR (28 variáveis)": resumo(R["metricas"][("A_completo", "LR")])}
deltas = {"LR (28 variáveis)": fmt_delta(R["deltas"]["XGBoost − LR, A_completo (AUROC)"])}
for nome, b in V["baseline"].items():
    linhas[f"LR ({nome})"] = resumo(b["metricas"])
    deltas[f"LR ({nome})"] = fmt_delta(b["delta_auroc"])
tab_base = pd.DataFrame(linhas).T.assign(**{"ΔAUROC XGBoost − modelo": pd.Series(deltas)})
tab_base.to_csv(FIG / "v2_baseline.csv")
tab_base

# %% [markdown]
# ### 6.2 Q2: desempenho por idade no óbito (vazamento na DN)
# A DN deveria ser preenchida no nascimento. Se campos como a anomalia congênita fossem completados depois do
# óbito, o modelo acertaria desproporcionalmente os óbitos do **dia 0**, que acontecem antes do fechamento da
# declaração. Cada faixa é comparada com todos os sobreviventes do teste.

# %%
V["idade_obito"].round(3)

# %%
from IPython.display import Image
Image(filename=str(FIG / "v2_idade_obito.png")) if (FIG / "v2_idade_obito.png").exists() else None

# %% [markdown]
# **Teste direto pela data da declaração.** A DN registra quando foi preenchida (`DTDECLARAC`) e digitada
# (`DTCADASTRO`). Dentro de cada faixa de idade no óbito, que controla a gravidade, comparamos os óbitos com DN
# declarada **depois** do óbito com os demais. Se a DN "soubesse" do óbito, o primeiro grupo seria previsto
# melhor, e isso deveria aparecer no modelo A (que tem Apgar e anomalia) e **não** no B.

# %%
dec = V["declaracao"]
dec[(dec["Campo"] == "declarada") & (dec["Faixa"] != "7–27 dias")].round(3)

# %% [markdown]
# ### 6.3 Q3 e Q4: pandemia e variáveis de chave
# - **Sem pandemia:** 2020–2021 saem do ajuste; validação e teste não mudam.
# - **Sem chaves:** saem peso, idade da mãe, semanas, gravidez e parto, que também definem o linkage. Uma
#   queda grande mostraria que o modelo depende justamente dos campos que determinam quem ganha rótulo.

# %%
rob = {}
for conj in preprocess.CONJUNTOS:
    rob[(NOME_CONJ[conj], "principal")] = {**resumo(R["metricas"][(conj, "XGBoost")]), "ΔAUROC": "—", "ΔAUPRC": "—"}
    for chave, rot in [("sem_pandemia", "sem 2020–2021"), ("sem_chaves", "sem variáveis de chave")]:
        v = V[chave][conj]
        rob[(NOME_CONJ[conj], rot)] = {**resumo(v["metricas"]), "ΔAUROC": fmt_delta(v["delta_auroc"]),
                                       "ΔAUPRC": fmt_delta(v["delta_auprc"])}
tab_rob = pd.DataFrame(rob).T
tab_rob.to_csv(FIG / "v2_robustez.csv")
tab_rob

# %% [markdown]
# ### 6.4 Q5: janelas deslizantes (drift temporal)
# Para cada ano de teste *t*: ajuste até *t*−2, validação em *t*−1 e teste em *t*. O modelo **reajustado** é
# refeito todo ano; o **congelado** é o da primeira janela aplicado aos anos seguintes, sem reajuste nem
# recalibração. A diferença entre os dois mede o custo de não atualizar o modelo.

# %%
V["janelas"].round(3)

# %%
Image(filename=str(FIG / "v2_janelas.png")) if (FIG / "v2_janelas.png").exists() else None

# %% [markdown]
# ### 6.5 Q6: simulação de padrão-ouro do linkage
# Não temos a base identificada, então fabricamos uma verdade conhecida. As DN dos nascidos efetivamente
# pareados viram "DO sintéticas", com a **ausência observada nas DO reais** e erros de digitação por campo em três
# cenários (baixo, realista, alto). Dois cenários extras retiram 5% e 10% das DN verdadeiras da base
# (sub-registro ou residência diferente na DN e na DO). O linkage roda sobre a base inteira, e os demais nascidos servem de
# distratores. Assim medimos a fração **real** de falsos pares de cada passada e a comparamos com a **estimada**
# pelo modelo de mistura usado na primeira parte.

# %%
S = V["simulacao"]
pd.Series(S["ausencia_do"], name="ausência na DO real").mul(100).round(1)

# %%
comp = []
for cen in [c for c in S if c != "ausencia_do"]:
    real, est = S[cen]["real"], S[cen]["estimado"]
    for p in [i for i in real.index if i != "total"]:
        comp.append({"cenário": cen, "passada": p, "pares": int(real.loc[p, "n_pares"]),
                     "falsos reais (%)": real.loc[p, "falsos_reais"],
                     "falsos estimados (%)": est.loc[p, ["falsos_ESCMAE", "falsos_QTDFILMORT"]].mean()
                     if p in est.index else np.nan})
    comp.append({"cenário": cen, "passada": "total", "pares": int(real.loc["total", "n_pares"]),
                 "falsos reais (%)": real.loc["total", "falsos_reais"],
                 "cobertura verdadeira (%)": real.loc["total", "cobertura_verdadeira_%"],
                 "etapa FS: pares": S[cen]["fs_pares"], "etapa FS: falsos reais (%)": S[cen]["fs_falsos_reais"]})
tab_sim = pd.DataFrame(comp)
tab_sim.to_csv(FIG / "v2_simulacao.csv", index=False)
tab_sim.round(1)

# %%
Image(filename=str(FIG / "v2_simulacao.png")) if (FIG / "v2_simulacao.png").exists() else None

# %% [markdown]
# ### 6.6 Q7: etapa probabilística (Fellegi–Sunter)
# Sobre os óbitos que o determinístico não pareou, comparamos cada óbito com os nascidos da mesma data e sexo
# (bloqueio). Cada campo recebe um nível de concordância (exato, próximo, discorda, ausente). As probabilidades
# *u* vêm de pares aleatórios do bloco; *m*, dos pares determinísticos em cujas chaves o campo não entra; e a
# proporção de pares λ supõe no máximo um par verdadeiro por óbito (Fellegi e Sunter, 1969). Pares com
# probabilidade a posteriori ≥ 0,9 são aceitos 1:1, do mais provável para o menos provável.
#
# O EM, alternativa usual para estimar *m* e λ, é mostrado para comparação: num bloco com ~1 par verdadeiro em
# centenas, ele convergiu para outra partição (λ dez vezes o plausível).

# %%
P = V["probabilistico"]
display(pd.Series({k: P[k] for k in ["n_obitos", "n_deterministico", "n_restantes", "candidatos",
                                     "n_probabilistico", "pct_total", "recuperados_no_teste"]}))
print("EM (comparação):", P["em"], "| λ usado:", round(P["parametros"]["lambda"], 5))

# %%
P["falsos_estimados"]

# %%
pd.DataFrame(P["parametros"]["m"]).T.round(3).add_prefix("m_").join(
    pd.DataFrame(P["parametros"]["u"]).T.round(3).add_prefix("u_"))

# %% [markdown]
# **Por que o resíduo não se recupera?** Quantos campos informativos cada óbito restante tem na DO, e qual a maior
# probabilidade a posteriori entre os seus candidatos:

# %%
display(P["restantes_campos"].rename("óbitos restantes por nº de campos preenchidos (de 7)"))
P["post_max_faixas"].rename("óbitos por faixa da melhor probabilidade a posteriori")

# %% [markdown]
# ### 6.7 Síntese da Parte 2
# | # | Resposta |
# |---|---|
# | Q1 | Peso + IG + Apgar 5 + anomalia chegam quase à AUROC do XGBoost; o ganho do XGBoost está na AUPRC. |
# | Q2 | Sem evidência de vazamento: DN declarada depois do óbito não é prevista melhor na mesma faixa de idade, e as diferenças aparecem também no modelo B. |
# | Q3 | Tirar 2020–2021 não muda o teste. |
# | Q4 | O modelo A resiste sem as variáveis de chave (perde AUPRC); o B depende da idade gestacional. |
# | Q5 | A discriminação é estável no tempo; a calibração deriva e exige recalibração anual. |
# | Q6 | A cascata é precisa quando a DN existe; DN ausente concentra falsos pares na chave mais tolerante; o estimador de mistura é só ordem de grandeza. |
# | Q7 | O resíduo não se recupera com Fellegi–Sunter: a DN provavelmente não está na base do estado. |
#
# Os números citados no relatório v2 (`relatorio_v2/main.pdf`) saem das tabelas acima.

# %% [markdown]
# ## 7. Conclusão
# O resumo dos resultados, a discussão e as limitações estão no `README.md`, em `relatorio/main.pdf` (v1) e em
# `relatorio_v2/main.pdf` (v2, com a Parte 2).
# Todos os números citados nesses documentos saem das tabelas acima.
#
# A execução completa do experimento principal levou:

# %%
print(f"{R['tempo_total_s'] / 60:.0f} min")
