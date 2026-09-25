"""Gera relatorio_v2/numeros_v2.tex (macros e tabelas) e as figuras v2_*.png da parte de aprofundamento.

Lê `data/resultados_v2.pkl`, escrito por `python -m src.aprofundamento`. Os números da primeira
parte continuam vindo de relatorio/numeros.tex (gerar_numeros.py).
Uso: .venv/bin/python relatorio_v2/gerar_numeros_v2.py
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from src import aprofundamento, experimento  # noqa: E402

V = aprofundamento.carregar()
R = experimento.carregar()
FIG = RAIZ / "figuras"
SAIDA = RAIZ / "relatorio_v2" / "numeros_v2.tex"
CONJ = [("A_completo", "A"), ("B_pre_parto", "B")]


def n(v, casas=3):
    return f"{v:.{casas}f}".replace(".", ",").replace("-", "$-$")


def mil(v):
    return f"{int(round(v)):,}".replace(",", ".")


def pc(v, casas=0):
    return f"{100 * v:.{casas}f}\\%".replace(".", ",")


def delta(d, casas=3):
    return f"{n(d['delta'], casas)} ({n(d['ic_inf'], casas)} a {n(d['ic_sup'], casas)})"


macros = {}
M = R["metricas"]

# ------------------------------------------------------------------ baseline clínico
tb = ["\\begin{tabular}{lcccc}", "\\toprule",
      "Modelo (conjunto A) & AUROC & AUPRC & Top 5\\% & $\\Delta$AUROC XGBoost $-$ modelo \\\\", "\\midrule"]
t = M[("A_completo", "XGBoost")]["valor"]
tb.append(f"XGBoost, {len(R['shap']['A_completo']['X'].columns)} variáveis & {n(t['AUROC'])} & {n(t['AUPRC'])} & "
          f"{pc(t['Óbitos no top 5%'])} & -- \\\\")
t = M[("A_completo", "LR")]["valor"]
tb.append(f"Regressão logística, todas & {n(t['AUROC'])} & {n(t['AUPRC'])} & {pc(t['Óbitos no top 5%'])} & "
          f"{delta(R['deltas']['XGBoost − LR, A_completo (AUROC)'])} \\\\")
for i, (nome, b) in enumerate(V["baseline"].items()):
    v = b["metricas"]["valor"]
    tb.append(f"LR, {nome} & {n(v['AUROC'])} & {n(v['AUPRC'])} & {pc(v['Óbitos no top 5%'])} & "
              f"{delta(b['delta_auroc'])} \\\\")
    k = "Tres" if i == 0 else "Quatro"
    macros[f"vBase{k}Auroc"] = n(v["AUROC"])
    macros[f"vBase{k}Auprc"] = n(v["AUPRC"])
    macros[f"vBase{k}Top"] = pc(v["Óbitos no top 5%"])
    macros[f"vBase{k}Dauroc"] = delta(b["delta_auroc"])
    macros[f"vBase{k}Dauprc"] = delta(b["delta_auprc"])
tb += ["\\bottomrule", "\\end{tabular}"]

# ------------------------------------------------------------------ idade no óbito
io = V["idade_obito"]
ti = ["\\begin{tabular}{llrcc}", "\\toprule", "Conj. & Idade no óbito & Óbitos & AUROC (IC 95\\%) & Top 5\\% \\\\",
      "\\midrule"]
for conj, c in CONJ:
    for _, r in io[io["Conjunto"] == conj].iterrows():
        ti.append(f"{c} & {r['Faixa'].replace('–', '--')} & {mil(r['Óbitos'])} & {n(r['AUROC'])} "
                  f"({n(r['AUROC_inf'])}--{n(r['AUROC_sup'])}) & {pc(r['Top 5%'])} \\\\")
    if c == "A":
        ti.append("\\midrule")
ti += ["\\bottomrule", "\\end{tabular}"]
for conj, c in CONJ:
    s = io[io["Conjunto"] == conj].set_index("Faixa")
    for faixa, k in [("dia 0", "Zero"), ("1–6 dias", "Um"), ("7–27 dias", "Sete")]:
        macros[f"vIdade{c}{k}"] = n(s.loc[faixa, "AUROC"])
        macros[f"vIdade{c}{k}Top"] = pc(s.loc[faixa, "Top 5%"])
    macros[f"vIdade{c}ZeroN"] = mil(s.loc["dia 0", "Óbitos"])

fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
for a, (conj, c) in zip(ax, CONJ):
    s = io[io["Conjunto"] == conj]
    a.errorbar(range(len(s)), s["AUROC"], yerr=[s["AUROC"] - s["AUROC_inf"], s["AUROC_sup"] - s["AUROC"]],
               fmt="o", capsize=4, color="#2b6cb0")
    a.set_xticks(range(len(s)), s["Faixa"])
    a.set_title(f"Conjunto {c}")
    a.grid(axis="y", alpha=.3)
ax[0].set_ylabel("AUROC (óbitos da faixa × sobreviventes)")
fig.tight_layout()
fig.savefig(FIG / "v2_idade_obito.png", dpi=200)
plt.close(fig)

# ------------------------------------------------------------------ pandemia e chaves
tr = ["\\begin{tabular}{llccc}", "\\toprule",
      "Conj. & Variante & AUROC & AUPRC & $\\Delta$AUROC variante $-$ principal \\\\", "\\midrule"]
for conj, c in CONJ:
    t = M[(conj, "XGBoost")]["valor"]
    tr.append(f"{c} & principal & {n(t['AUROC'])} & {n(t['AUPRC'])} & -- \\\\")
    for chave, rot, k in [("sem_pandemia", "sem 2020--2021 no ajuste", "Pand"),
                          ("sem_chaves", "sem variáveis de chave", "Chave")]:
        v = V[chave][conj]
        m_ = v["metricas"]["valor"]
        tr.append(f"{c} & {rot} & {n(m_['AUROC'])} & {n(m_['AUPRC'])} & {delta(v['delta_auroc'], 4)} \\\\")
        macros[f"v{k}{c}Auroc"] = n(m_["AUROC"])
        macros[f"v{k}{c}Auprc"] = n(m_["AUPRC"])
        macros[f"v{k}{c}Dauroc"] = delta(v["delta_auroc"], 4)
        macros[f"v{k}{c}Dauprc"] = delta(v["delta_auprc"], 4)
    if c == "A":
        tr.append("\\midrule")
tr += ["\\bottomrule", "\\end{tabular}"]

# ------------------------------------------------------------------ janelas deslizantes
jd = V["janelas"]
fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
cores = {"A_completo": "#2b6cb0", "B_pre_parto": "#c05621"}
for conj, c in CONJ:
    for modelo, estilo in [("reajustado", "-o"), ("congelado", "--s")]:
        s = jd[(jd["Conjunto"] == conj) & (jd["Modelo"] == modelo)]
        rot = f"{c}, {modelo}"
        ax[0].plot(s["Ano teste"], s["AUROC"], estilo, color=cores[conj], label=rot)
        ax[1].plot(s["Ano teste"], s["O/E"], estilo, color=cores[conj], label=rot)
        ax[2].plot(s["Ano teste"], s["Intercepto"], estilo, color=cores[conj], label=rot)
ax[1].axhline(1, color="gray", lw=.8)
ax[2].axhline(0, color="gray", lw=.8)
for a, tit in zip(ax, ["AUROC", "O/E", "Intercepto de calibração"]):
    a.set_title(tit)
    a.set_xlabel("Ano de teste")
    a.grid(alpha=.3)
ax[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(FIG / "v2_janelas.png", dpi=200)
plt.close(fig)
for conj, c in CONJ:
    s = jd[jd["Conjunto"] == conj]
    re_, co = s[s["Modelo"] == "reajustado"], s[s["Modelo"] == "congelado"]
    macros[f"vJan{c}AurocMin"] = n(re_["AUROC"].min())
    macros[f"vJan{c}AurocMax"] = n(re_["AUROC"].max())
    macros[f"vJan{c}OeMin"] = n(re_["O/E"].min(), 2)
    macros[f"vJan{c}OeMax"] = n(re_["O/E"].max(), 2)
    macros[f"vJan{c}CongOeMin"] = n(co["O/E"].min(), 2)
    macros[f"vJan{c}CongOeMax"] = n(co["O/E"].max(), 2)
    macros[f"vJan{c}CongAurocUlt"] = n(co["AUROC"].iloc[-1])
    macros[f"vJan{c}ReAurocUlt"] = n(re_["AUROC"].iloc[-1])
    macros[f"vJan{c}CongIntUlt"] = n(co["Intercepto"].iloc[-1], 2)
    macros[f"vJan{c}CongOeUlt"] = n(co["O/E"].iloc[-1], 2)
    macros[f"vJan{c}ReOeUlt"] = n(re_["O/E"].iloc[-1], 2)
macros["vJanAnoIni"] = str(int(jd["Ano teste"].min()))
macros["vJanAnoFim"] = str(int(jd["Ano teste"].max()))
tj = ["\\begin{tabular}{lllcccc}", "\\toprule",
      "Conj. & Modelo & Teste & TMN teste (‰) & AUROC & O/E & Intercepto \\\\", "\\midrule"]
for conj, c in CONJ:
    for modelo in ["reajustado", "congelado"]:
        for _, r in jd[(jd["Conjunto"] == conj) & (jd["Modelo"] == modelo)].iterrows():
            tj.append(f"{c} & {modelo} & {int(r['Ano teste'])} & {n(r['TMN teste (‰)'], 2)} & {n(r['AUROC'])} & "
                      f"{n(r['O/E'], 2)} & {n(r['Intercepto'], 3)} \\\\")
    if c == "A":
        tj.append("\\midrule")
tj += ["\\bottomrule", "\\end{tabular}"]

# ------------------------------------------------------------------ declaração depois do óbito (vazamento)
D = V["declaracao"]
dd = D[D["Campo"] == "declarada"]
tdec = ["\\begin{tabular}{llrcc}", "\\toprule",
        "Conj. / faixa & DN declarada & Óbitos & AUROC (IC 95\\%) & Top 5\\% \\\\", "\\midrule"]
for conj, c in CONJ:
    for faixa in ["dia 0", "1–6 dias"]:
        for momento, rot in [("depois do óbito", "depois do óbito"), ("antes ou no dia", "até o dia do óbito")]:
            r = dd[(dd["Conjunto"] == conj) & (dd["Faixa"] == faixa) & (dd["DN"] == momento)].iloc[0]
            auc = (f"{n(r['AUROC'])} ({n(r['AUROC_inf'])}--{n(r['AUROC_sup'])})" if pd.notna(r.get("AUROC"))
                   else "--")
            top = pc(r["Top 5%"]) if pd.notna(r.get("Top 5%")) else "--"
            tdec.append(f"{c}, {faixa.replace('–', '--')} & {rot} & {mil(r['Óbitos'])} & {auc} & {top} \\\\")
    if c == "A":
        tdec.append("\\midrule")
tdec += ["\\bottomrule", "\\end{tabular}"]
tot = dd[(dd["Conjunto"] == "A_completo") & (dd["Faixa"] == "todas") & (dd["DN"] == "depois do óbito")].iloc[0]
macros["vDecPct"] = pc(tot["% da faixa"] / 100)
macros["vDecN"] = mil(tot["Óbitos"])
z = dd[(dd["Conjunto"] == "A_completo") & (dd["Faixa"] == "dia 0") & (dd["DN"] == "depois do óbito")].iloc[0]
macros["vDecPctZero"] = pc(z["% da faixa"] / 100)
for conj, c in CONJ:
    for faixa, k in [("dia 0", "Zero"), ("1–6 dias", "Um")]:
        for momento, m in [("depois do óbito", "Dep"), ("antes ou no dia", "Ant")]:
            r = dd[(dd["Conjunto"] == conj) & (dd["Faixa"] == faixa) & (dd["DN"] == momento)].iloc[0]
            macros[f"vDec{c}{k}{m}"] = n(r["AUROC"]) if pd.notna(r.get("AUROC")) else "--"
            macros[f"vDec{c}{k}{m}N"] = mil(r["Óbitos"])
dg = D[(D["Campo"] == "digitada") & (D["Conjunto"] == "A_completo") & (D["Faixa"] == "todas")]
macros["vDigPct"] = pc(dg[dg["DN"] == "depois do óbito"].iloc[0]["% da faixa"] / 100)

# ------------------------------------------------------------------ Fellegi–Sunter
P = V["probabilistico"]
macros["vFsN"] = mil(P["n_probabilistico"])
macros["vFsCand"] = mil(P["candidatos"])
macros["vFsPctTotal"] = pc(P["pct_total"] / 100, 1)
macros["vFsDet"] = mil(P["n_deterministico"])
macros["vFsObitos"] = mil(P["n_obitos"])
macros["vFsRestantes"] = mil(P["n_restantes"])
macros["vFsTeste"] = mil(P["recuperados_no_teste"])
macros["vFsLambda"] = n(P["parametros"]["lambda"], 4)
macros["vFsEmLambda"] = n(P["em"]["lambda"], 3)
macros["vFsEmN"] = mil(P["em"]["n_pares"])
rc = P["restantes_campos"]
macros["vFsCompletos"] = mil(rc.get(len(aprofundamento.CAMPOS_FS), 0))
macros["vFsPoucos"] = mil(rc[rc.index <= 2].sum())
pf = P["post_max_faixas"]
macros["vFsPostBaixo"] = mil(pf.iloc[0])
macros["vFsPostBaixoPct"] = pc(pf.iloc[0] / pf.sum())
fe = P["falsos_estimados"]
ult = len(aprofundamento.linkage.PASSADAS) + 1
macros["vFsFalsos"] = (f"{int(fe.loc[ult, 'falsos_ESCMAE'])}\\% e {int(fe.loc[ult, 'falsos_QTDFILMORT'])}\\%"
                       if ult in fe.index else "--")

# ------------------------------------------------------------------ simulação de padrão-ouro
S = V["simulacao"]
cenarios = [c for c in S if c != "ausencia_do"]
rot_cen = {c: c.replace("realista + ", "+").replace("% sem DN", "\\% sem DN") for c in cenarios}
tsim = ["\\begin{tabular}{l" + "c" * len(cenarios) + "}", "\\toprule",
        "& \\multicolumn{" + str(len(cenarios)) + "}{c}{Falsos pares por cenário: real / estimado (\\%)} \\\\",
        "\\cmidrule(lr){2-" + str(len(cenarios) + 1) + "}",
        "Passada & " + " & ".join(rot_cen[c] for c in cenarios) + " \\\\", "\\midrule"]
passadas = sorted(i for i in S[cenarios[0]]["real"].index if i != "total")
for p_ in passadas:
    cel = []
    for c in cenarios:
        real, est = S[c]["real"], S[c]["estimado"]
        r = real.loc[p_, "falsos_reais"] if p_ in real.index else float("nan")
        e = est.loc[p_, ["falsos_ESCMAE", "falsos_QTDFILMORT"]].mean() if p_ in est.index else float("nan")
        cel.append(f"{n(r, 0)} / {n(e, 0)}")
    tsim.append(f"{p_} & " + " & ".join(cel) + " \\\\")
tsim.append("\\midrule")
tsim.append("Falsos no total (real) & " + " & ".join(f"{n(S[c]['real'].loc['total', 'falsos_reais'], 1)}\\%" for c in cenarios) + " \\\\")
tsim.append("Óbitos com par verdadeiro & " + " & ".join(f"{n(S[c]['real'].loc['total', 'cobertura_verdadeira_%'], 1)}\\%" for c in cenarios) + " \\\\")
tsim.append("Etapa FS: pares (falsos reais) & " + " & ".join(
    f"{mil(S[c]['fs_pares'])} ({n(S[c]['fs_falsos_reais'], 0)}\\%)" for c in cenarios) + " \\\\")
tsim += ["\\bottomrule", "\\end{tabular}"]
chave_cen = {"baixo": "Baixo", "realista": "Real", "alto": "Alto",
             "realista + 5% sem DN": "SemCinco", "realista + 10% sem DN": "SemDez"}
for c in cenarios:
    k = chave_cen.get(c)
    if not k:
        continue
    real = S[c]["real"]
    macros[f"vSim{k}Cob"] = f"{n(real.loc['total', 'cobertura_verdadeira_%'], 1)}\\%"
    macros[f"vSim{k}Falsos"] = f"{n(real.loc['total', 'falsos_reais'], 1)}\\%"
    macros[f"vSim{k}Seis"] = f"{n(real.loc[passadas[-1], 'falsos_reais'], 0)}\\%"
    est = S[c]["estimado"]
    macros[f"vSim{k}SeisEst"] = f"{n(est.loc[passadas[-1], ['falsos_ESCMAE', 'falsos_QTDFILMORT']].mean(), 0)}\\%"
    macros[f"vSim{k}FsN"] = mil(S[c]["fs_pares"])
    macros[f"vSim{k}FsFalsos"] = f"{n(S[c]['fs_falsos_reais'], 0)}\\%"
    macros[f"vSim{k}MaxTol"] = f"{n(real.loc[[p_ for p_ in passadas if p_ > 2], 'falsos_reais'].max(), 0)}\\%"

fig, ax = plt.subplots(1, len(cenarios), figsize=(13, 3.3), sharey=True)
for a, c in zip(ax, cenarios):
    real, est = S[c]["real"], S[c]["estimado"]
    ps = [p_ for p_ in passadas if p_ in est.index]
    a.bar([x - .2 for x in ps], [real.loc[p_, "falsos_reais"] for p_ in ps], .4, label="real", color="#2b6cb0")
    a.bar([x + .2 for x in ps], [est.loc[p_, ["falsos_ESCMAE", "falsos_QTDFILMORT"]].mean() for p_ in ps], .4,
          label="estimado (mistura)", color="#ed8936")
    a.set_title(c, fontsize=9)
    a.set_xlabel("Passada")
    a.grid(axis="y", alpha=.3)
ax[0].set_ylabel("Falsos pares (%)")
ax[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "v2_simulacao.png", dpi=200)
plt.close(fig)

# ------------------------------------------------------------------ saída
linhas = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
conteudo = "\n".join(["% Gerado por relatorio_v2/gerar_numeros_v2.py — não editar à mão.", *linhas,
                      "\\newcommand{\\tabVBaseline}{" + "\n".join(tb) + "}",
                      "\\newcommand{\\tabVIdade}{" + "\n".join(ti) + "}",
                      "\\newcommand{\\tabVRobustez}{" + "\n".join(tr) + "}",
                      "\\newcommand{\\tabVJanelas}{" + "\n".join(tj) + "}",
                      "\\newcommand{\\tabVSimulacao}{" + "\n".join(tsim) + "}",
                      "\\newcommand{\\tabVDeclaracao}{" + "\n".join(tdec) + "}"])
SAIDA.write_text(conteudo + "\n", encoding="utf-8")
(RAIZ / "relatorio_v2" / "macros_v2.json").write_text(json.dumps(macros, ensure_ascii=False, indent=0), encoding="utf-8")
print(f"{len(macros)} macros -> {SAIDA}")
