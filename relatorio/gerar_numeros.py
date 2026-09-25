"""Gera relatorio/numeros.tex: macros e tabelas LaTeX com os números do experimento.

O main.tex só referencia essas macros, então texto e resultados não divergem.
Uso: .venv/bin/python relatorio/gerar_numeros.py
"""
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from src import experimento, linkage, preprocess  # noqa: E402

R = experimento.carregar()
FIG = RAIZ / "figuras"
SAIDA = RAIZ / "relatorio" / "numeros.tex"


def n(v, casas=3):
    return f"{v:.{casas}f}".replace(".", ",")


def mil(v):
    return f"{int(round(v)):,}".replace(",", ".")


def pc(v, casas=1):
    return f"{100 * v:.{casas}f}\\%".replace(".", ",")


def ic(t, m, casas=3):
    r = t.loc[m]
    return f"{n(r.valor, casas)} ({n(r.ic_inf, casas)}--{n(r.ic_sup, casas)})"


macros = {}
rel = R["linkage"]
pa = rel["por_ano"]
macros["nNV"] = mil(pa["nascidos_vivos"].sum())
macros["nObitosSIM"] = mil(rel["n_obitos"])
macros["nPareados"] = mil(rel["n_pareados"])
macros["pctPareado"] = pc(rel["n_pareados"] / rel["n_obitos"])
macros["pctPareadoMin"] = pc(pa["pct_pareado"].min() / 100)
macros["pctPareadoMax"] = pc(pa["pct_pareado"].max() / 100)
macros["tmnSIM"] = n(1000 * pa["obitos_neonatais_sim"].sum() / pa["nascidos_vivos"].sum(), 2)
macros["tmnPareada"] = n(1000 * pa["obitos_pareados"].sum() / pa["nascidos_vivos"].sum(), 2)
fp = rel["falsos_pares"]
macros["nPassadas"] = str(len(linkage.PASSADAS))
macros["pctParesUmDois"] = pc(fp.loc[[1, 2], "n_pares"].sum() / rel["n_pareados"], 0)
macros["falsosUmDoisMax"] = f"{int(fp.loc[[1, 2], ['falsos_ESCMAE', 'falsos_QTDFILMORT']].max().max())}\\%"
macros["falsosMax"] = f"{int(fp[['falsos_ESCMAE', 'falsos_QTDFILMORT']].max().max())}\\%"
f_pond = ((fp["falsos_ESCMAE"] + fp["falsos_QTDFILMORT"]) / 2 * fp["n_pares"]).sum() / fp["n_pares"].sum()
macros["falsosPonderado"] = f"{f_pond:.0f}\\%"

am = R["amostras"]
for chave, nome in [("ajuste 2017–2021", "Ajuste"), ("validação 2022", "Valid"), ("teste 2023", "Teste")]:
    macros[f"n{nome}"] = mil(am[chave]["n"])
    macros[f"ob{nome}"] = mil(am[chave]["óbitos"])
    macros[f"tmn{nome}"] = n(am[chave]["TMN (‰)"], 2)
desc_ = pd.read_csv(FIG / "descritiva_tmn.csv", index_col=[0, 1])["TMN (‰)"]
macros["razaoEscol"] = n(desc_.loc[("Escolaridade da mãe", "Fund. incompleto ou menos")]
                         / desc_.loc[("Escolaridade da mãe", "Superior")], 1)
macros["nVarsA"] = str(len(preprocess.VARS_COMPLETO))
macros["nVarsB"] = str(len(preprocess.VARS_PRE_PARTO))

M = R["metricas"]
for (conj, nome), t in M.items():
    k = ("A" if conj == "A_completo" else "B") + {"LR": "Lr", "XGBoost": "Xgb", "LightGBM": "Lgbm"}[nome]
    for m, sig in [("AUROC", "Auroc"), ("AUPRC", "Auprc"), ("F1", "Fum"), ("Precisão", "Prec"),
                   ("Sensibilidade", "Sens"), ("Especificidade", "Espec"), ("VPN", "Vpn"), ("Brier", "Brier")]:
        macros[f"{k}{sig}"] = n(t.loc[m, "valor"], 4 if m == "Brier" else 3)
        macros[f"{k}{sig}IC"] = ic(t, m, 4 if m == "Brier" else 3)
    macros[f"{k}Top"] = pc(t.loc["Óbitos no top 5%", "valor"], 0)
    macros[f"{k}OE"] = n(t.loc["O/E", "valor"], 2)
for k_, d in R["deltas"].items():
    nome = {"XGBoost A − B (AUROC)": "dAB", "XGBoost A − B (AUPRC)": "dABprc",
            "XGBoost − LR, A_completo (AUROC)": "dXLa", "XGBoost − LR, A_completo (AUPRC)": "dXLaprc",
            "XGBoost − LR, B_pre_parto (AUROC)": "dXLb", "XGBoost − LR, B_pre_parto (AUPRC)": "dXLbprc"}[k_]
    macros[nome] = n(d["delta"])
    macros[nome + "IC"] = f"{n(d['ic_inf'])}--{n(d['ic_sup'])}"
C = R["complementares"]
for conj, k in [("A_completo", "A"), ("B_pre_parto", "B")]:
    macros[f"oeBruto{k}"] = n(R["calibracao"][conj]["OE_bruto"], 2)
    p = R["params"][(conj, "XGBoost")]
    macros[f"arv{k}"] = str(p["n_arvores"])
    macros[f"prof{k}"] = str(p["max_depth"])
    macros[f"lr{k}"] = n(p["learning_rate"], 3)
    macros[f"spw{k}"] = n(p["scale_pos_weight"], 0)
    for cen, sg in [("estrito", "Est"), ("pior caso", "Pior")]:
        v = C["sensibilidade"][(conj, cen)]
        macros[f"sens{k}{sg}Auroc"] = n(v["metricas"].loc["AUROC", "valor"])
        macros[f"sens{k}{sg}Auprc"] = n(v["metricas"].loc["AUPRC", "valor"])
        macros[f"sens{k}{sg}OE"] = n(v["metricas"].loc["O/E", "valor"], 2)
        d = v["delta_auroc"]
        macros[f"sens{k}{sg}Delta"] = f"{n(d['delta'], 4)} ({n(d['ic_inf'], 4)} a {n(d['ic_sup'], 4)})"
    for t, tg in [("bruto", "Bru"), ("calibrado", "Cal")]:
        c = C["calibracao"][conj][t]
        macros[f"cal{k}{tg}Int"] = f"{n(c['intercepto'])} ({n(c['intercepto_inf'])} a {n(c['intercepto_sup'])})"
        macros[f"cal{k}{tg}Inc"] = f"{n(c['inclinacao'])} ({n(c['inclinacao_inf'])} a {n(c['inclinacao_sup'])})"
    dc = C["decisao"][conj].set_index("limiar")
    for pt, sg in [(0.01, "Um"), (0.05, "Cinco"), (0.2, "Vinte")]:
        macros[f"nb{k}{sg}"] = n(1000 * dc.loc[pt, "modelo"], 1)
        macros[f"sinal{k}{sg}"] = pc(dc.loc[pt, "fração sinalizada"], 1)
    macros[f"nbPositivo{k}"] = "sim" if (dc["modelo"] > dc[["tratar todos", "tratar nenhum"]].max(axis=1)).all() else "não"
macros["driftLog"] = n(__import__("math").log(am["teste 2023"]["TMN (‰)"] / am["ajuste 2017–2021"]["TMN (‰)"]))
macros["posTreinoEstrito"] = mil(C["sensibilidade"][("A_completo", "estrito")]["positivos_treino"])

vl = C["vies_linkage"]
pf, ps = vl["perfil"], vl["por_subgrupo"].set_index(["Variável", "Categoria"])
macros["nNaoPareados"] = mil(pf.loc["n", "Não pareados"])
for lin, sg in [("Peso ausente (%)", "PesoAus"), ("Semanas ausentes (%)", "SemAus"),
                ("Idade da mãe ausente (%)", "IdAus"), ("Residente na capital (%)", "Cap"),
                ("Óbito em hospital (%)", "Hosp"), ("Peso mediano (g)", "PesoMed")]:
    casas = 0 if "g)" in lin else 1
    macros[f"par{sg}"] = n(pf.loc[lin, "Pareados"], casas)
    macros[f"npar{sg}"] = n(pf.loc[lin, "Não pareados"], casas)
macros["linkCap"] = pc(ps.loc[("Residência", "Capital"), "Pareados (%)"] / 100)
macros["linkInt"] = pc(ps.loc[("Residência", "Interior"), "Pareados (%)"] / 100)
rr = ps.loc["Raça/cor do recém-nascido (DO)", "Pareados (%)"].drop("Outra/ignorada")
macros["linkRacaMin"] = pc(rr.min() / 100)
macros["linkRacaMax"] = pc(rr.max() / 100)
macros["linkPesoAus"] = pc(ps.loc[("Peso na DO", "ausente"), "Pareados (%)"] / 100)
macros["pctVerdadeiros"] = pc(rel["n_pareados"] * (1 - f_pond / 100) / rel["n_obitos"], 0)

vif = C["colinearidade"]["VIF"]
macros["vifFilhos"], macros["vifGest"] = n(vif["Filhos vivos"], 0), n(vif["Gestações anteriores"], 0)
macros["vifPartos"] = n(vif["Partos vaginais anteriores"], 1)
macros["vifOutrosMax"] = n(vif.drop(["Filhos vivos", "Gestações anteriores", "Partos vaginais anteriores"]).max(), 1)
au = C["ausencia"].set_index("Variável")
for v, sg in [("Consultas de pré-natal", "Cons"), ("Idade do pai", "Pai"), ("Semanas de gestação", "Sem"),
              ("Apgar 5º min", "Apg")]:
    macros[f"aus{sg}"] = n(au.loc[v, "TMN ausente (‰)"], 1)
    macros[f"pre{sg}"] = n(au.loc[v, "TMN preenchido (‰)"], 1)
    macros[f"pctAus{sg}"] = n(au.loc[v, "Ausente (%)"], 1)
g = C["gemelares"]
macros["pctMult"] = n(g["% múltiplos"], 1)
macros["pctObMult"] = n(g["% dos óbitos em múltiplos"], 1)
macros["aurocUnicosA"] = n(g["AUROC A_completo (só únicos)"])
macros["aurocUnicosB"] = n(g["AUROC B_pre_parto (só únicos)"])
macros["repo"] = "https://github.com/CledsonSilvaAraujo/mortalidade-neonatal-rj"
macros["tempoMin"] = f"{R['tempo_total_s'] / 60:.0f}"

sub = pd.read_csv(FIG / "subgrupos.csv").set_index(["Conjunto", "Subgrupo"])
for conj, k in [("A · completo", "A"), ("B · pré-parto", "B")]:
    t = sub.loc[conj]
    macros[f"sub{k}AurocMin"] = n(t["AUROC"].min())
    macros[f"sub{k}AurocMax"] = n(t["AUROC"].max())
    for g, sg in [("Fund. incompleto ou menos", "Fund"), ("Superior", "Sup"), ("Branca", "Branca"),
                  ("Preta", "Preta"), ("Capital", "Cap"), ("Interior", "Int")]:
        r = t.loc[g]
        macros[f"sub{k}{sg}Auroc"] = f"{n(r['AUROC'])} ({n(r['AUROC_inf'])}--{n(r['AUROC_sup'])})"
        macros[f"sub{k}{sg}OE"] = f"{n(r['O/E'], 2)} ({n(r['O/E_inf'], 2)}--{n(r['O/E_sup'], 2)})"

# Ranking SHAP (5 primeiras) por conjunto
for conj, k in [("A_completo", "A"), ("B_pre_parto", "B")]:
    s = R["shap"][conj]
    imp = pd.Series(abs(s["valores"]).mean(0), index=s["X"].columns).sort_values(ascending=False)
    rot = [preprocess.ROTULOS[v] for v in imp.index[:5]]
    macros[f"shapTop{k}"] = ", ".join(r if r.startswith("Apgar") else r[0].lower() + r[1:] for r in rot)

linhas = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]

# Tabela: linkage por passada
tl = ["\\begin{tabular}{clrrr}", "\\toprule",
      "& Chave & Pares & \\multicolumn{2}{c}{Falsos pares estimados} \\\\",
      "\\cmidrule(lr){4-5}", "& & & Escol. mãe & Perdas fetais \\\\", "\\midrule"]
curto = {"DTNASC": "data", "SEXO": "sexo", "CODMUNRES": "mun.", "PESO": "peso", "IDADEMAE": "id. mãe",
         "SEMAGESTAC": "sem. gest.", "GRAVIDEZ": "gravidez", "PARTO": "parto", "CODESTAB": "estab."}
for i, chave in enumerate(linkage.PASSADAS, 1):
    r = fp.loc[i]
    tl.append(f"{i} & {' + '.join(curto[c] for c in chave)} & {mil(r['n_pares'])} & "
              f"{r['falsos_ESCMAE']:.0f}\\% & {r['falsos_QTDFILMORT']:.0f}\\% \\\\")
tl += ["\\bottomrule", "\\end{tabular}"]

# Tabela: métricas no teste
tm = ["\\begin{tabular}{llcccccc}", "\\toprule",
      "Conj. & Modelo & AUROC & AUPRC & F1 & Precisão & Sensib. & Top 5\\% \\\\", "\\midrule"]
for (conj, nome), t in M.items():
    c = "A" if conj == "A_completo" else "B"
    neg = (lambda x: f"\\textbf{{{x}}}") if nome == "XGBoost" else (lambda x: x)
    tm.append(f"{c} & {nome} & {neg(ic(t, 'AUROC'))} & {neg(ic(t, 'AUPRC'))} & {n(t.loc['F1', 'valor'])} & "
              f"{n(t.loc['Precisão', 'valor'])} & {n(t.loc['Sensibilidade', 'valor'])} & "
              f"{pc(t.loc['Óbitos no top 5%', 'valor'], 0)} \\\\")
    if conj == "A_completo" and nome == "LightGBM":
        tm.append("\\midrule")
tm += ["\\midrule", "\\multicolumn{2}{l}{Batista et al. (SP)} & 0,971 & 0,586 & 0,548 & 0,729 & 0,440 & $>$90\\% \\\\",
       "\\bottomrule", "\\end{tabular}"]

# Tabela: descritiva
desc = pd.read_csv(FIG / "descritiva_tmn.csv")
desc.columns = ["Variável", "Categoria", "NV", "Óbitos", "TMN"]
td = ["\\begin{tabular}{llrrr}", "\\toprule", "Variável & Categoria & NV & Óbitos & TMN (‰) \\\\", "\\midrule"]
anterior = None
for _, r in desc.iterrows():
    if r["Categoria"] in ("ausente", "Ignorada", "ignorado") or r["Variável"] not in (
            "Peso ao nascer", "Idade gestacional", "Escolaridade da mãe"):
        continue
    v = r["Variável"] if r["Variável"] != anterior else ""
    if anterior is not None and v:
        td.append("\\addlinespace")
    cat = str(r["Categoria"]).replace("≥", "$\\geq$").replace("<", "$<$")
    td.append(f"{v} & {cat} & {mil(r['NV'])} & {mil(r['Óbitos'])} & {n(r['TMN'], 1)} \\\\")
    anterior = r["Variável"]
td += ["\\bottomrule", "\\end{tabular}"]

# Tabela: hiperparâmetros XGBoost
th = ["\\begin{tabular}{lrr}", "\\toprule", "Hiperparâmetro & A · completo & B · pré-parto \\\\", "\\midrule"]
pa_, pb_ = R["params"][("A_completo", "XGBoost")], R["params"][("B_pre_parto", "XGBoost")]
for k_ in ["learning_rate", "max_depth", "min_child_weight", "subsample", "colsample_bytree", "reg_lambda",
           "reg_alpha", "gamma", "scale_pos_weight", "n_arvores"]:
    fmt = (lambda v: str(int(v))) if k_ in ("max_depth", "n_arvores") else (lambda v: n(v, 3))
    th.append(f"\\texttt{{{k_.replace('_', chr(92) + '_')}}} & {fmt(pa_[k_])} & {fmt(pb_[k_])} \\\\")
th += ["\\bottomrule", "\\end{tabular}"]

tv = ["\\begin{tabular}{lrr}", "\\toprule", "Característica (na DO) & Pareados & Não pareados \\\\", "\\midrule"]
for lin in ["n", "Peso ausente (%)", "Semanas ausentes (%)", "Idade da mãe ausente (%)", "Peso mediano (g)",
            "Peso < 1.500 g (%)", "Óbito no 1º dia (%)", "Residente na capital (%)", "Óbito em hospital (%)"]:
    fmt = mil if lin in ("n", "Peso mediano (g)") else (lambda v: n(v, 1))
    tv.append(f"{lin.replace('%', chr(92) + '%').replace('<', '$<$')} & {fmt(pf.loc[lin, 'Pareados'])} & "
              f"{fmt(pf.loc[lin, 'Não pareados'])} \\\\")
tv += ["\\bottomrule", "\\end{tabular}"]

ts = ["\\begin{tabular}{llcccc}", "\\toprule",
      "Conj. & Rótulos no treino & AUROC & AUPRC & O/E & $\\Delta$AUROC (IC 95\\%) \\\\", "\\midrule"]
for conj, c in [("A_completo", "A"), ("B_pre_parto", "B")]:
    t = M[(conj, "XGBoost")]
    ts.append(f"{c} & principal & {n(t.loc['AUROC', 'valor'])} & {n(t.loc['AUPRC', 'valor'])} & "
              f"{n(t.loc['O/E', 'valor'], 2)} & -- \\\\")
    for cen, rot in [("estrito", "estrito (sem passadas 3--6)"), ("pior caso", "pior caso (passadas 3--6 = 0)")]:
        v = C["sensibilidade"][(conj, cen)]
        m_, d = v["metricas"]["valor"], v["delta_auroc"]
        ts.append(f"{c} & {rot} & {n(m_['AUROC'])} & {n(m_['AUPRC'])} & {n(m_['O/E'], 2)} & "
                  f"{n(d['delta'], 4)} ({n(d['ic_inf'], 4)} a {n(d['ic_sup'], 4)}) \\\\")
    if c == "A":
        ts.append("\\midrule")
ts += ["\\bottomrule", "\\end{tabular}"]

conteudo = "\n".join(["% Gerado por relatorio/gerar_numeros.py — não editar à mão.", *linhas,
                      "\\newcommand{\\tabLinkage}{" + "\n".join(tl) + "}",
                      "\\newcommand{\\tabMetricas}{" + "\n".join(tm) + "}",
                      "\\newcommand{\\tabDescritiva}{" + "\n".join(td) + "}",
                      "\\newcommand{\\tabHiper}{" + "\n".join(th) + "}",
                      "\\newcommand{\\tabVies}{" + "\n".join(tv) + "}",
                      "\\newcommand{\\tabSens}{" + "\n".join(ts) + "}"])
SAIDA.write_text(conteudo + "\n", encoding="utf-8")
print(f"{len(macros)} macros -> {SAIDA}")
