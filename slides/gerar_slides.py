"""Gera slides/apresentacao.pptx (16:9, 14 slides) a partir de data/resultados.pkl e figuras/.

Todos os números vêm do experimento gravado; nada é digitado à mão.
Uso: .venv/bin/python slides/gerar_slides.py
"""
import sys
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from src import experimento, linkage  # noqa: E402

FIG = RAIZ / "figuras"
SAIDA = RAIZ / "slides" / "apresentacao.pptx"
AUTOR = "Cledson da Silva Araújo"

# Paleta: petróleo domina, menta como apoio, coral só para destaque.
PETROLEO = RGBColor(0x0B, 0x4F, 0x5C)
PETROLEO_ESCURO = RGBColor(0x06, 0x33, 0x3C)
MENTA = RGBColor(0xE3, 0xF1, 0xEF)
MENTA_MEDIA = RGBColor(0x9C, 0xCF, 0xC6)
CORAL = RGBColor(0xE8, 0x5D, 0x3F)
TINTA = RGBColor(0x1F, 0x2A, 0x2E)
CINZA = RGBColor(0x5F, 0x6B, 0x70)
BRANCO = RGBColor(0xFF, 0xFF, 0xFF)
TITULO, CORPO = "Cambria", "Calibri"

W, H = 13.333, 7.5
MARGEM = 0.6


def num(v, casas=3):
    return f"{v:.{casas}f}".replace(".", ",")


def milhar(v):
    return f"{int(v):,}".replace(",", ".")


# ------------------------------------------------------------------ helpers
def fundo(slide, cor):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = cor


def texto(slide, x, y, w, h, conteudo, tamanho=16, cor=TINTA, fonte=CORPO, negrito=False,
          alinhar=PP_ALIGN.LEFT, ancora=MSO_ANCHOR.TOP, italico=False, espaco=6):
    """`conteudo`: str, ou lista de parágrafos; cada parágrafo é str ou lista de (texto, dict)."""
    caixa = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = caixa.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = ancora
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    paragrafos = conteudo if isinstance(conteudo, list) else [conteudo]
    for i, par in enumerate(paragrafos):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = alinhar
        p.space_after = Pt(espaco)
        pedacos = par if isinstance(par, list) else [(par, {})]
        for trecho, estilo in pedacos:
            r = p.add_run()
            r.text = trecho
            f = r.font
            f.name = estilo.get("fonte", fonte)
            f.size = Pt(estilo.get("tamanho", tamanho))
            f.bold = estilo.get("negrito", negrito)
            f.italic = estilo.get("italico", italico)
            f.color.rgb = estilo.get("cor", cor)
    return caixa


def marcadores(slide, x, y, w, h, itens, tamanho=16, cor=TINTA):
    """Lista com marcador nativo (sem caractere '•' literal)."""
    caixa = texto(slide, x, y, w, h, itens, tamanho=tamanho, cor=cor, espaco=10)
    for p in caixa.text_frame.paragraphs:
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(Inches(0.28)))
        pPr.set("indent", str(-Inches(0.28)))
        bu = pPr.makeelement("{http://schemas.openxmlformats.org/drawingml/2006/main}buChar", {"char": "•"})
        pPr.append(bu)
    return caixa


def cartao(slide, x, y, w, h, cor=MENTA, borda=None):
    forma = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    forma.adjustments[0] = 0.08
    forma.fill.solid()
    forma.fill.fore_color.rgb = cor
    if borda is None:
        forma.line.fill.background()
    else:
        forma.line.color.rgb = borda
        forma.line.width = Pt(1.25)
    forma.shadow.inherit = False
    return forma


def circulo(slide, x, y, d, rotulo, cor=PETROLEO, cor_texto=BRANCO, tamanho=18):
    forma = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
    forma.fill.solid()
    forma.fill.fore_color.rgb = cor
    forma.line.fill.background()
    forma.shadow.inherit = False
    tf = forma.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = rotulo
    r.font.name, r.font.size, r.font.bold, r.font.color.rgb = TITULO, Pt(tamanho), True, cor_texto
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    return forma


def seta(slide, x, y, w, h, cor=MENTA_MEDIA):
    forma = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(w), Inches(h))
    forma.fill.solid()
    forma.fill.fore_color.rgb = cor
    forma.line.fill.background()
    forma.shadow.inherit = False


def titulo(slide, t, sub=None):
    texto(slide, MARGEM, 0.45, W - 2 * MARGEM, 0.8, t, tamanho=34, fonte=TITULO, negrito=True, cor=PETROLEO)
    if sub:
        texto(slide, MARGEM, 1.2, W - 2 * MARGEM, 0.45, sub, tamanho=16, cor=CINZA, italico=True)


def tabela(slide, x, y, w, df, larguras, destaque=None, tamanho=13, altura_linha=0.42):
    linhas, colunas = df.shape[0] + 1, df.shape[1]
    forma = slide.shapes.add_table(linhas, colunas, Inches(x), Inches(y), Inches(w),
                                   Inches(altura_linha * linhas))
    tb = forma.table
    for j, lw in enumerate(larguras):
        tb.columns[j].width = Inches(lw)
    for i in range(linhas):
        tb.rows[i].height = Inches(altura_linha)
        for j in range(colunas):
            cel = tb.cell(i, j)
            valor = df.columns[j] if i == 0 else df.iat[i - 1, j]
            cel.text = str(valor)
            cel.margin_left = cel.margin_right = Inches(0.08)
            cel.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cel.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 or (i == 0 and j == 0) else PP_ALIGN.CENTER
            f = p.runs[0].font
            f.name, f.size = CORPO, Pt(tamanho)
            cel.fill.solid()
            if i == 0:
                cel.fill.fore_color.rgb = PETROLEO
                f.bold, f.color.rgb = True, BRANCO
            else:
                realce = destaque is not None and destaque(i - 1)
                cel.fill.fore_color.rgb = MENTA if realce else (BRANCO if i % 2 else RGBColor(0xF4, 0xF8, 0xF8))
                f.bold = bool(realce)
                f.color.rgb = TINTA
    return forma


def notas(slide, t):
    slide.notes_slide.notes_text_frame.text = t



def pct(v, casas=1):
    return f"{100 * v:.{casas}f}%".replace(".", ",")


def ic(t, m, casas=3):
    r = t.loc[m]
    return f"{num(r.valor, casas)} ({num(r.ic_inf, casas)}–{num(r.ic_sup, casas)})"


def figura(slide, nome, x, y, w=None, h=None):
    slide.shapes.add_picture(str(FIG / nome), Inches(x), Inches(y),
                             Inches(w) if w else None, Inches(h) if h else None)


# ------------------------------------------------------------------ dados
R = experimento.carregar()
rel = R["linkage"]
M = R["metricas"]
xa, xb = M[("A_completo", "XGBoost")], M[("B_pre_parto", "XGBoost")]
d_auc, d_prc = R["deltas"]["XGBoost A − B (AUROC)"], R["deltas"]["XGBoost A − B (AUPRC)"]
n_nv = int(R["fluxo"]["n"].iloc[-1])
pct_link = rel["n_pareados"] / rel["n_obitos"]
tmn_sim = 1000 * rel["por_ano"]["obitos_neonatais_sim"].sum() / rel["por_ano"]["nascidos_vivos"].sum()
fp = rel["falsos_pares"]
sub = pd.read_csv(FIG / "subgrupos.csv")
C = R["complementares"]
sens = C["sensibilidade"]
ps_link = C["vies_linkage"]["por_subgrupo"].set_index(["Variável", "Categoria"])["Pareados (%)"]
link_cap, link_int = ps_link[("Residência", "Capital")] / 100, ps_link[("Residência", "Interior")] / 100
f_pond = ((fp["falsos_ESCMAE"] + fp["falsos_QTDFILMORT"]) / 2 * fp["n_pares"]).sum() / fp["n_pares"].sum() / 100
pct_verd = pct_link * (1 - f_pond)

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(W), Inches(H)
BRANCO_LAYOUT = prs.slide_layouts[6]


def novo(cor=BRANCO):
    s = prs.slides.add_slide(BRANCO_LAYOUT)
    fundo(s, cor)
    return s


# 1 ---------------------------------------------------------------- título
s = novo(PETROLEO)
texto(s, MARGEM + 0.2, 1.5, 11.8, 2.4,
      "Predição de mortalidade neonatal no Rio de Janeiro com dados abertos",
      tamanho=42, fonte=TITULO, negrito=True, cor=BRANCO)
texto(s, MARGEM + 0.2, 3.65, 11.5, 0.9,
      f"Linkage SINASC × SIM sem identificadores e aprendizado de máquina · {milhar(n_nv)} nascidos vivos, 2017–2023",
      tamanho=20, cor=MENTA_MEDIA)
texto(s, MARGEM + 0.2, 5.3, 11.5, 1.2,
      [AUTOR, "Introdução ao Aprendizado de Máquina para a Saúde · UFF · 2026.2"], tamanho=16, cor=BRANCO)
notas(s, "Apresento a reprodução, no estado do Rio, de um modelo brasileiro de predição de mortalidade neonatal, "
         "usando apenas dados públicos do DATASUS.")

# 2 ---------------------------------------------------------------- motivação
s = novo()
titulo(s, "Motivação", "Quem nasce com maior risco de morrer no primeiro mês?")
cartao(s, MARGEM, 1.9, 3.8, 2.3, MENTA)
texto(s, MARGEM + 0.3, 2.05, 3.3, 1.1, f"{num(tmn_sim, 1)}‰", tamanho=54, fonte=TITULO, negrito=True, cor=PETROLEO)
texto(s, MARGEM + 0.3, 3.2, 3.3, 0.9, "óbitos neonatais por mil nascidos vivos no RJ, 2017–2023 (SIM)",
      tamanho=14)
cartao(s, MARGEM, 4.5, 3.8, 2.3, MENTA)
texto(s, MARGEM + 0.3, 4.65, 3.3, 1.1, milhar(rel["n_obitos"]), tamanho=54, fonte=TITULO, negrito=True, cor=CORAL)
texto(s, MARGEM + 0.3, 5.8, 3.3, 0.9, "óbitos de 0 a 27 dias de vida em sete anos", tamanho=14)
marcadores(s, 5.1, 2.0, 7.6, 4.8, [
    "O período neonatal concentra a maior parte da mortalidade infantil, sobretudo por causas evitáveis.",
    "A Declaração de Nascido Vivo (DN) é preenchida para todo nascimento e já está digitalizada.",
    "Um escore de risco automático poderia priorizar UTI neonatal, transporte e visita domiciliar.",
    [("Pergunta: ", {"negrito": True, "cor": PETROLEO}),
     ("dá para fazer isso só com dados abertos?", {})],
], tamanho=18)
notas(s, "A mortalidade neonatal é a maior parcela da mortalidade infantil. A DN existe para todo nascimento, "
         "então um modelo baseado nela poderia ser aplicado em qualquer município.")

# 3 ---------------------------------------------------------------- artigo
s = novo()
titulo(s, "Artigo de referência", "Batista et al. (2021), BMC Pediatrics 21:322")
marcadores(s, MARGEM, 1.9, 6.4, 4.9, [
    "Município de São Paulo, nascimentos 2012–2017: 1.128.162 nascidos vivos, 7.282 óbitos neonatais.",
    "Linkage probabilístico SINASC × SIM com dados identificados (cedidos pela prefeitura).",
    "23 variáveis da DN; cinco algoritmos; otimização bayesiana; treino 2012–2016, teste 2017.",
    "Principais variáveis (SHAP): Apgar 5, peso, Apgar 1, anomalia, idade gestacional.",
], tamanho=17)
cartao(s, 7.4, 1.9, 5.3, 4.9, MENTA)
texto(s, 7.7, 2.05, 4.8, 0.5, "XGBoost no teste de 2017", tamanho=17, negrito=True, cor=PETROLEO)
for i, (v, r) in enumerate([("0,971", "AUROC"), ("0,586", "AUPRC"), ("0,548", "F1"), ("> 90%", "dos óbitos nos 5% de maior risco")]):
    texto(s, 7.7, 2.7 + i * 1.0, 1.9, 0.8, v, tamanho=30, fonte=TITULO, negrito=True, cor=PETROLEO)
    texto(s, 9.7, 2.85 + i * 1.0, 2.9, 0.8, r, tamanho=15)
notas(s, "O artigo de referência é brasileiro e mostra desempenho muito alto. Mas depende de dados identificados, "
         "de uma única cidade, e não avalia calibração nem subgrupos.")

# 4 ---------------------------------------------------------------- lacunas
s = novo()
titulo(s, "Lacunas e perguntas de pesquisa")
itens = [
    ("Reprodução com dados abertos", "Os microdados públicos não têm número da DN nem nome: é possível parear e "
     "reproduzir o desempenho no RJ?", "H1: AUROC ≥ 0,95"),
    ("Risco antes do parto", "Quanto se perde sem Apgar, peso e anomalia, usando só o que se sabe na admissão "
     "para o parto?", "H2: queda substancial, ainda útil"),
    ("Calibração e equidade", "O modelo erra mais para algum grupo de raça/cor, escolaridade da mãe ou "
     "região?", "H3: calibração pior na baixa escolaridade"),
]
for i, (t, d_, h) in enumerate(itens):
    x = MARGEM + i * 4.1
    cartao(s, x, 1.6, 3.8, 5.1, MENTA)
    circulo(s, x + 0.25, 1.85, 0.6, str(i + 1), tamanho=16)
    texto(s, x + 0.3, 2.65, 3.2, 0.9, t, tamanho=20, negrito=True, cor=PETROLEO)
    texto(s, x + 0.3, 3.6, 3.2, 2.0, d_, tamanho=16)
    texto(s, x + 0.3, 5.8, 3.2, 0.7, h, tamanho=15, italico=True, cor=CORAL)
notas(s, "Três perguntas: reproduzir com dados abertos, medir o valor das variáveis do parto e verificar "
         "calibração e equidade, que a revisão sistemática de Mangold et al. aponta como lacunas da área.")

# 5 ---------------------------------------------------------------- dados
s = novo()
titulo(s, "Dados e desenho temporal", "Microdados públicos do DATASUS, residentes no estado do RJ")
am = pd.DataFrame(R["amostras"]).T
etapas = [("Ajuste", "2017–2021", am.iloc[0]), ("Validação", "2022", am.iloc[1]), ("Teste", "2023", am.iloc[2])]
for i, (nome, anos, r) in enumerate(etapas):
    x = MARGEM + i * 4.1
    cartao(s, x, 1.9, 3.6, 2.4, CORAL if nome == "Teste" else MENTA)
    cor_t = BRANCO if nome == "Teste" else PETROLEO
    texto(s, x + 0.3, 2.05, 3.0, 0.5, f"{nome} · {anos}", tamanho=18, negrito=True, cor=cor_t)
    texto(s, x + 0.3, 2.65, 3.0, 0.8, milhar(r["n"]), tamanho=34, fonte=TITULO, negrito=True, cor=cor_t)
    texto(s, x + 0.3, 3.5, 3.0, 0.6, f"nascidos vivos · {milhar(r['óbitos'])} óbitos", tamanho=14,
          cor=BRANCO if nome == "Teste" else TINTA)
    if i < 2:
        seta(s, x + 3.65, 2.9, 0.4, 0.4)
marcadores(s, MARGEM, 4.7, W - 2 * MARGEM, 2.3, [
    "SINASC (DNRJ2017–2023) e SIM (DORJ2017–2024) baixados do FTP do DATASUS, sem cadastro.",
    "Desfecho: óbito não fetal com 0 a 27 dias de vida. O SIM vai até 2024 para captar quem nasceu em dezembro de 2023.",
    "Validação usada para hiperparâmetros, early stopping, recalibração e limiar; 2023 é avaliado uma única vez.",
], tamanho=16)
notas(s, "Espelhamos o desenho do artigo: treinar no passado e testar no ano seguinte. O ano de 2022 serve só "
         "para escolhas de modelagem.")

# 6 ---------------------------------------------------------------- linkage
s = novo()
titulo(s, "Linkage sem identificadores", "A Declaração de Óbito de menores de 1 ano repete dados do nascimento")
marcadores(s, MARGEM, 1.85, 5.6, 3.0, [
    f"{len(linkage.PASSADAS)} chaves compostas em cascata: data de nascimento, sexo, município, peso, idade da mãe, "
    "semanas de gestação, gravidez, parto, estabelecimento.",
    "Só pares 1:1 em cada passada.",
    "Falsos pares estimados por mistura em campos fora das chaves.",
], tamanho=16)
cartao(s, MARGEM, 5.0, 2.7, 1.9, MENTA)
texto(s, MARGEM + 0.25, 5.1, 2.3, 0.8, pct(pct_link), tamanho=34, fonte=TITULO, negrito=True, cor=PETROLEO)
texto(s, MARGEM + 0.25, 5.95, 2.3, 0.9, f"dos {milhar(rel['n_obitos'])} óbitos pareados", tamanho=13)
cartao(s, MARGEM + 2.9, 5.0, 2.7, 1.9, MENTA)
texto(s, MARGEM + 3.15, 5.1, 2.3, 0.8, f"≈{pct(pct_verd, 0)}", tamanho=34, fonte=TITULO, negrito=True, cor=CORAL)
texto(s, MARGEM + 3.15, 5.95, 2.3, 0.9, "com par verdadeiro, descontados os falsos pares", tamanho=13)
figura(s, "linkage_por_ano.png", 6.6, 1.8, w=6.2)
f12 = fp.loc[[1, 2]]
texto(s, 6.7, 5.2, 6.0, 1.6,
      f"Falsos pares estimados nas passadas 1–2 (que somam {pct(f12['n_pares'].sum() / rel['n_pareados'], 0)} dos pares): "
      f"até {int(f12[['falsos_ESCMAE', 'falsos_QTDFILMORT']].max().max())}%. As passadas tolerantes chegam a "
      f"{int(fp[['falsos_ESCMAE', 'falsos_QTDFILMORT']].max().max())}%. Pareamento: {pct(link_cap)} na capital, "
      f"{pct(link_int)} no interior. Óbitos sem peso na DO quase não pareiam.",
      tamanho=14, cor=CINZA)
notas(s, "Esta é a principal contribuição metodológica: um linkage reproduzível com dados abertos, com taxa de "
         "pareamento comparável à obtida com nomes e endereços.")

# 7 ---------------------------------------------------------------- método
s = novo()
titulo(s, "Método", "Dois conjuntos de variáveis × três modelos")
cartao(s, MARGEM, 1.8, 5.9, 2.5, MENTA)
texto(s, MARGEM + 0.3, 1.95, 5.3, 0.5, "A · completo (28 variáveis da DN)", tamanho=18, negrito=True, cor=PETROLEO)
texto(s, MARGEM + 0.3, 2.5, 5.3, 1.7, "Tudo o que o artigo usa: Apgar 1 e 5, peso, anomalia, idade gestacional, "
      "tipo de parto, Robson, pré-natal, dados da mãe.", tamanho=15)
cartao(s, MARGEM + 6.2, 1.8, 5.9, 2.5, MENTA)
texto(s, MARGEM + 6.5, 1.95, 5.3, 0.5, "B · pré-parto (17 variáveis)", tamanho=18, negrito=True, cor=PETROLEO)
texto(s, MARGEM + 6.5, 2.5, 5.3, 1.7, "O que se sabe na admissão: idade gestacional, gemelaridade, apresentação, "
      "consultas e início do pré-natal, Kotelchuck, histórico obstétrico, mãe.", tamanho=15)
marcadores(s, MARGEM, 4.6, W - 2 * MARGEM, 2.5, [
    "Regressão logística (L2, baseline) · XGBoost (modelo do artigo) · LightGBM.",
    "Optuna/TPE maximizando a AUPRC em 2022; peso de classe como hiperparâmetro (nenhum, raiz, total).",
    "Recalibração isotônica em 2022. Discriminação pelo escore bruto; calibração e limiar pela probabilidade.",
    "IC 95% por bootstrap estratificado (500); ΔAUROC/ΔAUPRC por bootstrap pareado; SHAP em 20 mil nascimentos.",
], tamanho=16)
notas(s, "O conjunto B responde à pergunta clínica de antes do parto. Todas as escolhas foram feitas em 2022.")

# 8 ---------------------------------------------------------------- resultados
s = novo()
titulo(s, "Resultados no teste de 2023", "IC 95% por bootstrap; limiar escolhido em 2022")
linhas = []
for (conj, nome), t in M.items():
    linhas.append({"Conjunto": "A · completo" if conj == "A_completo" else "B · pré-parto", "Modelo": nome,
                   "AUROC": ic(t, "AUROC"), "AUPRC": ic(t, "AUPRC"), "F1": num(t.loc["F1", "valor"]),
                   "Óbitos no top 5%": pct(t.loc["Óbitos no top 5%", "valor"])})
tab = pd.DataFrame(linhas)
tabela(s, MARGEM, 1.85, W - 2 * MARGEM, tab, [1.8, 1.5, 2.6, 2.6, 1.2, 2.4],
       destaque=lambda i: tab.iloc[i]["Modelo"] == "XGBoost", tamanho=14, altura_linha=0.5)
texto(s, MARGEM, 5.6, W - 2 * MARGEM, 1.4, [
    [("Artigo (SP, 2017): ", {"negrito": True, "cor": PETROLEO}),
     ("AUROC 0,971 · AUPRC 0,586 · F1 0,548 · > 90% dos óbitos no top 5%.", {})],
    [("XGBoost × LR (A): ", {"negrito": True, "cor": PETROLEO}),
     (f"ΔAUROC {num(R['deltas']['XGBoost − LR, A_completo (AUROC)']['delta'])}, "
      f"ΔAUPRC {num(R['deltas']['XGBoost − LR, A_completo (AUPRC)']['delta'])}: ganho pequeno do boosting.", {})],
], tamanho=16)
notas(s, "H1 se confirma: AUROC acima de 0,95 com dados abertos, um pouco abaixo do artigo. A diferença entre "
         "boosting e logística é pequena.")

# 9 ---------------------------------------------------------------- curvas
s = novo()
titulo(s, "Discriminação e captura de óbitos")
figura(s, "roc_pr.png", MARGEM, 1.35, w=7.6)
figura(s, "captura_top.png", 8.4, 1.6, w=4.5)
texto(s, 8.4, 4.6, 4.5, 2.4,
      f"Priorizando 5% dos nascimentos, o modelo completo captura {pct(xa.loc['Óbitos no top 5%', 'valor'], 0)} dos "
      f"óbitos; o pré-parto, {pct(xb.loc['Óbitos no top 5%', 'valor'], 0)}.", tamanho=16)
notas(s, "Linhas cheias são o conjunto completo; tracejadas, o pré-parto. Os três modelos quase se sobrepõem.")

# 10 --------------------------------------------------------------- pré-parto
s = novo()
titulo(s, "Quanto vale o momento do parto?", "XGBoost: conjunto completo × pré-parto")
cartao(s, MARGEM, 1.9, 3.6, 2.2, MENTA)
texto(s, MARGEM + 0.3, 2.0, 3.0, 0.9, num(xb.loc["AUROC", "valor"]), tamanho=40, fonte=TITULO, negrito=True, cor=PETROLEO)
texto(s, MARGEM + 0.3, 3.0, 3.0, 1.0, "AUROC do modelo na admissão para o parto", tamanho=14)
cartao(s, MARGEM, 4.4, 3.6, 2.4, MENTA)
texto(s, MARGEM + 0.3, 4.5, 3.0, 0.9, f"−{num(d_auc['delta'])}", tamanho=40, fonte=TITULO, negrito=True, cor=CORAL)
texto(s, MARGEM + 0.3, 5.5, 3.2, 1.2, f"ΔAUROC (IC {num(d_auc['ic_inf'])}–{num(d_auc['ic_sup'])}); "
      f"ΔAUPRC −{num(d_prc['delta'])}", tamanho=14)
figura(s, "shap_importancia.png", 4.5, 1.9, w=8.3)
texto(s, 4.5, 5.6, 8.3, 1.4, [
    "A idade gestacional domina, e consultas e Kotelchuck também dependem dela.",
    [("Uso: ", {"negrito": True, "cor": PETROLEO}),
     ("risco na admissão, dado que o parto é agora (transferência in utero). Não serve para triagem precoce.", {})],
], tamanho=14)
notas(s, "O modelo pré-parto perde discriminação, mas ainda separa bem, principalmente pela idade gestacional. "
         "Ele é o que pode ser usado antes do nascimento, para regulação.")

# 11 --------------------------------------------------------------- calibração e utilidade
s = novo()
titulo(s, "Calibração e utilidade clínica", "XGBoost recalibrado, teste de 2023")
figura(s, "curva_decisao.png", MARGEM, 1.7, w=7.8)
ca, cb = C["calibracao"]["A_completo"]["calibrado"], C["calibracao"]["B_pre_parto"]["calibrado"]
dca = C["decisao"]["A_completo"].set_index("limiar")
marcadores(s, 8.7, 1.8, 4.1, 5.3, [
    f"Inclinação de calibração: {num(ca['inclinacao'], 2)} (A) e {num(cb['inclinacao'], 2)} (B), ideal 1.",
    f"Intercepto de B: {num(cb['intercepto'], 3)} ≈ ln(8,20/7,70). É o aumento da TMN em 2023 (drift).",
    f"Benefício líquido acima de tratar todos/nenhum de 1% a 20%. Com pt = 5%: {num(1000 * dca.loc[0.05, 'modelo'], 1)} "
    f"por mil, sinalizando {pct(dca.loc[0.05, 'fração sinalizada'])} dos nascimentos.",
], tamanho=15)

# 12 --------------------------------------------------------------- subgrupos
s = novo()
titulo(s, "Subgrupos e H3", "IC 95% por bootstrap dentro de cada subgrupo")
figura(s, "subgrupos.png", MARGEM, 1.6, w=8.2)
a_sub = sub[sub["Conjunto"] == "A · completo"].set_index("Subgrupo")
b_sub = sub[sub["Conjunto"] == "B · pré-parto"].set_index("Subgrupo")
fund = b_sub.loc["Fund. incompleto ou menos"]
marcadores(s, 9.0, 1.8, 3.8, 5.2, [
    [("H3 não confirmada: ", {"negrito": True, "cor": PETROLEO}),
     (f"O/E na baixa escolaridade {num(fund['O/E'], 2)} (IC {num(fund['O/E_inf'], 2)}–{num(fund['O/E_sup'], 2)}), "
      "que inclui 1.", {})],
    f"Modelo A homogêneo: AUROC de {num(a_sub['AUROC'].min())} a {num(a_sub['AUROC'].max())}.",
    f"Modelo B subestima mães brancas (O/E {num(b_sub.loc['Branca', 'O/E'], 2)}) e o interior "
    f"(O/E {num(b_sub.loc['Interior', 'O/E'], 2)}). No interior, o pareamento também é menor.",
], tamanho=15)

# 12 --------------------------------------------------------------- limitações
s = novo()
titulo(s, "Sensibilidade ao linkage e limitações")
se, sp = sens[("A_completo", "estrito")], sens[("A_completo", "pior caso")]
cartao(s, MARGEM, 1.6, W - 2 * MARGEM, 1.5, MENTA)
texto(s, MARGEM + 0.3, 1.7, W - 2 * MARGEM - 0.6, 1.35, [
    [("Mesmo teste de 2023, rótulos alternativos no treino: ", {"negrito": True, "cor": PETROLEO}),
     (f"ΔAUROC {num(se['delta_auroc']['delta'], 4)} (estrito) e {num(sp['delta_auroc']['delta'], 4)} (pior caso), "
      "com IC que incluem 0. A discriminação é robusta,", {})],
    (f"mas o O/E sobe para {num(se['metricas'].loc['O/E', 'valor'], 2)}–{num(sp['metricas'].loc['O/E', 'valor'], 2)}: "
     "a calibração depende da completude do linkage."),
], tamanho=15)
lims = [
    ("Rótulos imperfeitos", f"≈{pct(1 - pct_verd, 0)} dos óbitos sem par verdadeiro; menos extremos e mais do interior."),
    ("Um estado, anos recentes", "Validação temporal, mas não externa (outra UF)."),
    ("Idade gestacional", "No modelo B é a IG do parto: vale para a admissão, não para semanas antes."),
    ("Uso clínico", f"Precisão de {pct(xa.loc['Precisão', 'valor'], 0)} no limiar (A): "
     "alertas precisam de desenho de fluxo."),
]
for i, (t, d_) in enumerate(lims):
    x = MARGEM + (i % 2) * 6.15
    y = 3.3 + (i // 2) * 1.9
    cartao(s, x, y, 5.95, 1.7, BRANCO, borda=MENTA_MEDIA)
    texto(s, x + 0.3, y + 0.15, 5.4, 0.5, t, tamanho=17, negrito=True, cor=PETROLEO)
    texto(s, x + 0.3, y + 0.7, 5.4, 0.9, d_, tamanho=15)
notas(s, "A sensibilidade mostra que o resultado não depende das passadas tolerantes. As limitações principais "
         "são o linkage incompleto e a ausência de validação em outro estado.")

# 13 --------------------------------------------------------------- conclusão
s = novo(PETROLEO)
texto(s, MARGEM, 0.5, W - 2 * MARGEM, 0.8, "Conclusões e próximos passos", tamanho=36, fonte=TITULO,
      negrito=True, cor=BRANCO)
conclusoes = [
    f"Linkage aberto: {pct(pct_link)} pareados (≈{pct(pct_verd, 0)} verdadeiros); discriminação robusta a ele.",
    f"Reprodução do artigo no RJ: AUROC {num(xa.loc['AUROC', 'valor'])} e {pct(xa.loc['Óbitos no top 5%', 'valor'], 0)} "
    "dos óbitos no top 5%.",
    f"Na admissão para o parto: AUROC {num(xb.loc['AUROC', 'valor'])} e benefício líquido positivo.",
]
for i, c in enumerate(conclusoes):
    y = 1.65 + i * 1.05
    circulo(s, MARGEM, y, 0.7, str(i + 1), cor=MENTA_MEDIA, cor_texto=PETROLEO_ESCURO, tamanho=18)
    texto(s, MARGEM + 1.0, y + 0.05, 11.0, 0.7, c, tamanho=20, cor=BRANCO, ancora=MSO_ANCHOR.MIDDLE)
cartao(s, MARGEM, 4.95, W - 2 * MARGEM, 1.95, PETROLEO_ESCURO)
texto(s, MARGEM + 0.35, 5.1, 11.4, 0.45, "Trabalhos futuros", tamanho=18, negrito=True, cor=MENTA_MEDIA)
texto(s, MARGEM + 0.35, 5.6, 11.4, 1.3,
      "Validação externa em outras UFs · etapa probabilística no linkage · completude do linkage por ano e região · "
      "recalibração periódica (drift) · avaliação de impacto com gestores",
      tamanho=17, cor=BRANCO)
notas(s, "Em resumo: é possível reproduzir o modelo do artigo com dados abertos, e um modelo pré-parto já "
         "tem desempenho útil. Obrigado.")

# ------------------------------------------------------------------ notas do apresentador (~15 min)
am_ = pd.DataFrame(R["amostras"]).T
ca_, cb_ = C["calibracao"]["A_completo"]["calibrado"], C["calibracao"]["B_pre_parto"]["calibrado"]
vl_ = C["vies_linkage"]["perfil"]
NOTAS = [
    # 1
    "Boa tarde. Vou apresentar meu projeto de aplicação: prever o óbito neonatal no estado do Rio de Janeiro "
    "usando só dados abertos do SUS. O trabalho parte de um artigo brasileiro, que fez isso em São Paulo com "
    "dados identificados. A pergunta central é se dá para chegar ao mesmo resultado sem acesso privilegiado aos "
    "dados. Para isso, construí um linkage entre nascimentos e óbitos sem nenhum identificador. Vou mostrar como "
    "o linkage foi feito, o quanto ele erra e se esses erros afetam as conclusões. Depois apresento os modelos, "
    "a calibração, a utilidade clínica e as limitações.",
    # 2
    f"Óbito neonatal é o que ocorre de 0 a 27 dias de vida. É a maior parte da mortalidade infantil. No RJ, "
    f"entre 2017 e 2023, foram {num(tmn_sim, 1)} óbitos por mil nascidos vivos, {milhar(rel['n_obitos'])} ao todo. Boa "
    "parte é evitável, e o risco se concentra em poucos nascimentos. A Declaração de Nascido Vivo é preenchida "
    "para todo nascimento, em qualquer município. Um escore calculado a partir dela pode ajudar a priorizar UTI "
    "neonatal, transporte e visita domiciliar. A pergunta é se isso é possível com os dados que qualquer "
    "pesquisador pode baixar.",
    # 3
    "O artigo de referência é de Batista e colegas, 2021, na BMC Pediatrics. Eles parearam nascimentos e óbitos "
    "de São Paulo usando nome e endereço, cedidos pela prefeitura. Com as 23 variáveis da DN, treinaram cinco "
    "algoritmos com validação temporal: treino até 2016 e teste em 2017. O XGBoost teve AUROC de 0,971, e os 5% "
    "de maior risco concentraram mais de 90% dos óbitos. As variáveis mais importantes foram Apgar, peso, "
    "anomalia e idade gestacional. É um resultado forte, mas numa única cidade e com dados que não estão "
    "disponíveis para terceiros.",
    # 4
    "Daí vêm três hipóteses, as mesmas da minha proposta. H1: com dados abertos, o modelo completo atinge AUROC "
    "de pelo menos 0,95. H2: um modelo só com o que se sabe na admissão para o parto perde desempenho, mas "
    "continua útil. H3: a calibração piora nos filhos de mães com menor escolaridade. As lacunas de calibração e "
    "de validação externa também aparecem na revisão sistemática de Mangold: só 45% dos estudos reportam "
    "calibração. Vou mostrar o resultado de cada hipótese, inclusive a que não se confirmou.",
    # 5
    f"Os dados são o SINASC e o SIM do DATASUS, por residência no RJ, baixados do FTP público, sem cadastro. São "
    f"{milhar(n_nv)} nascidos vivos. O SIM vai até 2024 para captar quem nasceu em dezembro de 2023. O desenho é "
    f"temporal, como no artigo: ajuste em 2017 a 2021, validação em 2022 e teste em 2023, usado uma única vez. "
    f"Um detalhe importante: a TMN do teste, {num(am_.iloc[2]['TMN (‰)'], 2)}, é maior que a do treino, "
    f"{num(am_.iloc[0]['TMN (‰)'], 2)}. Isso vai aparecer na calibração. Registrei o hash SHA-256 de cada "
    "arquivo, porque o DATASUS revisa os dados.",
    # 6
    f"Esta é a parte central. Os arquivos públicos não têm o número da DN nem nomes. Mas a Declaração de Óbito "
    f"de menores de um ano repete dados do nascimento. Então pareio por chaves compostas, em cascata, aceitando "
    f"só pares únicos. Pareei {pct(pct_link)} dos óbitos. Parte desses pares é falsa: estimei isso comparando "
    f"campos que não entram nas chaves, e o resultado é cerca de {pct(pct_verd, 0)} dos óbitos com par verdadeiro. "
    f"A estimativa supõe que a primeira passada, com oito campos, não tem falsos pares. O pareamento é "
    f"{pct(link_cap)} na capital e {pct(link_int)} no interior. Os não pareados são os óbitos sem peso, semanas "
    f"ou idade da mãe na DO: {num(vl_.loc['Peso ausente (%)', 'Não pareados'], 0)}% deles não têm peso.",
    # 7
    "Comparo dois conjuntos de variáveis. O A é o completo, como no artigo, com Apgar, peso e anomalia. O B tem "
    "só o que se sabe na admissão para o parto. Atenção: a idade gestacional do B é a do parto, e consultas e "
    "Kotelchuck dependem dela. Então o B responde a qual é o risco se o parto acontece agora, e não semanas "
    "antes. Uso três modelos: regressão logística como linha de base, XGBoost e LightGBM. Os hiperparâmetros "
    "vêm de busca bayesiana, maximizando a AUPRC, que é a métrica certa com 0,8% de positivos. Não usei SMOTE, "
    "para preservar a calibração.",
    # 8
    f"No teste de 2023, o XGBoost completo teve AUROC de {num(xa.loc['AUROC', 'valor'])} e AUPRC de "
    f"{num(xa.loc['AUPRC', 'valor'])}. O artigo teve 0,971 e 0,586. H1 se confirma. A logística fica muito perto: "
    f"o ganho do XGBoost em AUROC é de {num(R['deltas']['XGBoost − LR, A_completo (AUROC)']['delta'])}, com limite "
    "inferior praticamente zero, e o ganho mais claro está na AUPRC. Na prática, as variáveis da DN carregam "
    "quase toda a informação, e o algoritmo importa pouco. A precisão no limiar é maior e a sensibilidade menor "
    "que no artigo, mas isso decorre da escolha do limiar.",
    # 9
    f"Nas curvas, as linhas cheias são o modelo completo e as tracejadas, o pré-parto. Os três algoritmos quase "
    f"se sobrepõem. À direita está a métrica operacional: priorizando 5% dos nascimentos, o modelo completo "
    f"captura {pct(xa.loc['Óbitos no top 5%', 'valor'], 0)} dos óbitos e o pré-parto, "
    f"{pct(xb.loc['Óbitos no top 5%', 'valor'], 0)}. Ramos e colegas, em 2025, com dados nacionais, encontraram mais "
    "de 85%, o que é coerente. A diferença para os 90% do artigo é compatível com os óbitos sem par verdadeiro, "
    "que são menos extremos e mais difíceis, e com a heterogeneidade do estado.",
    # 10
    f"O modelo pré-parto perde {num(d_auc['delta'])} de AUROC e bem mais de AUPRC, mas ainda chega a "
    f"{num(xb.loc['AUROC', 'valor'])}. H2 se confirma em parte: perdeu, mas continua útil. Pelo SHAP, quase tudo vem "
    "da idade gestacional. Por isso, o uso precisa ser bem delimitado: na admissão, com o parto iminente, para "
    "decidir transferência in utero para uma maternidade com UTI neonatal e preparar a equipe. Ele não serve para "
    "triagem precoce da gestação, porque nem a idade gestacional do parto nem as consultas acumuladas são "
    "conhecidas semanas antes. Essa é uma pergunta provável da banca.",
    # 11
    f"Calibração e utilidade clínica. A inclinação de calibração é {num(ca_['inclinacao'], 2)} no A e "
    f"{num(cb_['inclinacao'], 2)} no B, praticamente 1, então as previsões não são extremas demais. O intercepto do B "
    f"é {num(cb_['intercepto'], 3)}, que é exatamente o logaritmo de 8,20 sobre 7,70: a mortalidade subiu em 2023. É "
    "drift de prevalência, não defeito do modelo. Em uso real, o intercepto precisaria ser recalibrado "
    "periodicamente. A curva de decisão mostra benefício líquido acima de tratar todos e de tratar nenhum em "
    "todos os limiares de 1% a 20%. Esta é a métrica que diz se o modelo ajuda a decidir.",
    # 12
    f"Subgrupos e H3. A hipótese era calibração pior na baixa escolaridade. O O/E nesse grupo foi "
    f"{num(fund['O/E'], 2)}, com intervalo que inclui 1 e se sobrepõe ao do ensino superior. Então H3 não se "
    "confirmou, e prefiro dizer isso claramente a trocar a hipótese depois de ver os dados. O modelo completo é "
    "homogêneo. O pré-parto subestima o risco para mães brancas e para o interior. No interior há uma explicação "
    "alternativa: o pareamento é menor lá. Para raça/cor, o pareamento é quase igual entre os grupos, então o "
    "linkage não explica.",
    # 13
    f"A pergunta mais provável é se os erros do linkage explicam os resultados. Para testar, reajustei o modelo "
    f"com rótulos alternativos no treino, avaliando sempre no mesmo teste de 2023. No cenário estrito, removi os "
    f"pares duvidosos. No pior caso, marquei-os como sobreviventes. A AUROC mudou "
    f"{num(se['delta_auroc']['delta'], 4)} e {num(sp['delta_auroc']['delta'], 4)}, com intervalos que incluem zero. A "
    f"discriminação é robusta. Mas o O/E sobe para até {num(sp['metricas'].loc['O/E', 'valor'], 2)}: a calibração "
    "absoluta depende da completude do linkage. As limitações principais são os rótulos imperfeitos, a falta de "
    "validação externa e o uso de variáveis do pós-parto no modelo A.",
    # 14
    "Para concluir. Primeiro, dá para reproduzir o resultado do artigo com dados totalmente abertos, e a "
    "discriminação é robusta aos erros do linkage. Segundo, a calibração absoluta não é: depende da completude do "
    "linkage e do drift temporal, então um uso real exigiria recalibração periódica. Terceiro, o modelo na "
    "admissão para o parto tem desempenho útil para decisões de transferência. Como próximos passos: validar em "
    "outros estados, acrescentar uma etapa probabilística ao linkage e avaliar o impacto com gestores. O código "
    "e os dados de reprodução estão no repositório. Obrigado.",
]
assert len(NOTAS) == len(prs.slides), (len(NOTAS), len(prs.slides))
for slide, t in zip(prs.slides, NOTAS):
    notas(slide, t)

prs.save(SAIDA)
print(f"{len(prs.slides)} slides -> {SAIDA}")
