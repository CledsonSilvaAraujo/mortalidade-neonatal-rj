"""Preenche o modelo .odp da disciplina ("Apresentação do tema do projeto de aplicação").

Mantém layout, fontes e estilos do modelo e troca apenas o texto das caixas; os
números saem de data/resultados.pkl. Uso:
    .venv/bin/python slides/gerar_apresentacao_tema.py "<modelo.odp>"
Saída: slides/apresentacao_tema.odp
"""
import re
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

from lxml import etree

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from src import download, experimento, preprocess  # noqa: E402

SAIDA = RAIZ / "slides" / "apresentacao_tema.odp"
AUTOR = "Cledson da Silva Araújo"

A0, AF = download.ANOS_NASCIMENTO[0], download.ANOS_NASCIMENTO[-1]
AV, AT = preprocess.ANO_VALIDACAO, preprocess.ANO_TESTE
R = experimento.carregar()
rel = R["linkage"]
pa = rel["por_ano"]
n_nv = pa["nascidos_vivos"].sum()
tmn = 1000 * rel["n_obitos"] / n_nv
prev_sim = rel["n_obitos"] / n_nv       # todos os óbitos neonatais do SIM
prev = rel["n_pareados"] / n_nv         # prevalência da classe positiva (óbitos pareados)
pct_link = rel["n_pareados"] / rel["n_obitos"]


def br(v, casas=1):
    return f"{v:.{casas}f}".replace(".", ",")


# ------------------------------------------------------------------ conteúdo
# Cada item: ("q", texto) = pergunta/tópico (estilo T3, 18 pt); ("a", texto) = resposta (T4, 12 pt).
SLIDES = {
    "Google Shape;61;p14": [  # Motivação
        ("q", "Qual o problema de saúde?"),
        ("a", f"Óbito neonatal (0–27 dias), maior parte da mortalidade infantil: {br(tmn)} por mil nascidos vivos no RJ."),
        ("q", "Por que ele é relevante?"),
        ("a", "É em boa parte evitável, e o risco se concentra em poucos nascimentos, que podem ser priorizados."),
        ("q", "Quem seria beneficiado pelo resultado?"),
        ("a", "Recém-nascidos e gestantes de alto risco, maternidades, regulação de leitos e atenção básica."),
        ("q", "Qual é a pergunta clínica ou de pesquisa?"),
        ("a", "Dá para prever o óbito neonatal só com a DN e dados abertos? E só com o que se sabe antes do parto?"),
        ("q", "Qual seria o uso pretendido do modelo?"),
        ("a", "Priorizar UTI neonatal e visita domiciliar e, na admissão para o parto, decidir transferência in utero."),
    ],
    "Google Shape;68;p15": [  # Formulação do problema de ML
        ("q", "Tarefa de aprendizado"),
        ("a", f"Classificação binária supervisionada, com classes muito desbalanceadas (≈{br(100 * prev)}% de positivos)."),
        ("q", "Unidade de análise"),
        ("a", "Nascido vivo: uma Declaração de Nascido Vivo (DN) por nascimento."),
        ("q", "Entrada → Saída"),
        ("a", "X: variáveis da DN (mãe, gestação, pré-natal, parto, Apgar, peso, anomalia congênita): 28 no modelo "
              "completo e 17 no modelo pré-parto."),
        ("a", "y: óbito com 0 a 27 dias de vida (1) ou sobrevivência (0), obtido pelo pareamento da DN com a "
              "Declaração de Óbito (SIM)."),
    ],
    "Google Shape;74;p16": [  # Dados
        ("q", "Qual dataset será utilizado?"),
        ("a", "SINASC (nascidos vivos) + SIM (óbitos), microdados do DATASUS por residência no estado do RJ, "
              f"nascimentos de {A0} a {AF}: ftp.datasus.gov.br/dissemin/publicos"),
        ("a", f"≈{br(n_nv / 1e6)} milhão de nascidos vivos e ≈{br(rel['n_obitos'] / 1e3, 0)} mil óbitos neonatais "
              f"({br(100 * prev_sim)}%). Acesso público, sem cadastro, dados anonimizados."),
        ("a", "Artigo de referência: Batista et al. (2021), BMC Pediatrics, município de São Paulo, XGBoost "
              "com AUROC de 0,971."),
        ("q", "O dataset realmente permite responder à pergunta proposta?"),
        ("a", "Sim, com uma ressalva: os arquivos públicos não têm identificador, então nascimento e óbito serão "
              "pareados por chaves compostas (data, sexo, peso, idade da mãe…). Teste preliminar: "
              f"{br(100 * pct_link)}% dos óbitos pareados."),
    ],
    "Google Shape;80;p17": [  # Viabilidade e riscos metodológicos
        ("q", "Principais riscos e como tratá-los"),
        ("a", "1. Rótulo imperfeito: óbitos não pareados viram falsos sobreviventes, e pode haver pares errados → "
              "medir o pareamento, estimar falsos pares e fazer análise de sensibilidade."),
        ("a", f"2. Classes desbalanceadas ({br(100 * prev)}% de óbitos) → AUPRC como métrica principal, peso de "
              "classe como hiperparâmetro, sem sobreamostragem para não distorcer a calibração."),
        ("a", "3. Variáveis registradas só depois do nascimento (Apgar, peso) → comparar com um modelo que usa "
              "apenas o que se sabe na admissão para o parto (risco dado que o parto é agora)."),
        ("a", "4. Campos ausentes ou \"ignorados\" na DN e mudanças no registro ao longo dos anos → ausência como "
              "categoria própria e validação temporal."),
        ("a", "5. Representatividade: um único estado → avaliar desempenho por raça/cor, escolaridade da mãe e "
              "capital × interior."),
    ],
    "Google Shape;86;p18": [  # Estratégia inicial de avaliação
        ("q", "Como saberemos se o modelo funciona?"),
        ("a", "Baseline: regressão logística. Modelos: XGBoost (o do artigo de referência) e LightGBM."),
        ("a", "Métrica principal: AUPRC. Também AUROC, F1, % dos óbitos entre os 5% de maior risco (comparável "
              "ao artigo) e calibração (observado/esperado)."),
        ("a", f"Separação temporal: treino {A0}–{AV - 1}, validação {AV} (hiperparâmetros, calibração e limiar) e "
              f"teste {AT}, usado uma única vez."),
        ("a", "Comparações: modelo completo × pré-parto; subgrupos com IC 95% por bootstrap."),
    ],
}

REFERENCIAS = [
    "Batista AFM, Diniz CSG, Bonilha EA, Kawachi I, Chiavegatto Filho ADP. Neonatal mortality prediction with "
    "routinely collected data: a machine learning approach. BMC Pediatrics 21:322, 2021. (artigo de referência)",
    "Mangold C, Zoretic S, Thallapureddy K, Moreira A, Chorath K, Moreira A. Machine learning models for "
    "predicting neonatal mortality: a systematic review. Neonatology 118:394–405, 2021.",
    "Maia LTS, Souza WV, Mendes ACG, Silva AGS. Use of linkage to improve the completeness of the SIM and "
    "SINASC in the Brazilian capitals. Revista de Saúde Pública 51:112, 2017.",
    "Chen T, Guestrin C. XGBoost: a scalable tree boosting system. KDD, 2016.",
    "Ministério da Saúde/DATASUS. Microdados do SINASC e do SIM. ftp.datasus.gov.br/dissemin/publicos",
]

TITULO = {
    "Google Shape;54;p13": ("T1", "Predição de mortalidade neonatal no RJ"),
    "Google Shape;55;p13": ("T2", "Apresentação do tema do projeto de aplicação"),
    "Google Shape;56;p13": ("T3", AUTOR),
}

NOTAS = [
    "Proponho prever o óbito neonatal no estado do Rio usando apenas dados abertos do SUS.",
    "O óbito neonatal é a maior parcela da mortalidade infantil e se concentra em poucos nascimentos. "
    "A pergunta é se a DN basta para identificar esses nascimentos, inclusive antes do parto.",
    "É uma classificação binária por nascido vivo. O rótulo vem do pareamento com o SIM.",
    "Os dados são públicos. O desafio é que não há identificador; o teste preliminar do linkage pareou "
    "cerca de 90% dos óbitos, mas parte desses pares pode ser falsa, e isso será estimado.",
    "Os riscos principais são o rótulo imperfeito do linkage e o desbalanceamento; cada um tem um tratamento previsto.",
    "A avaliação espelha o artigo: treinar no passado e testar no ano seguinte, com AUPRC como métrica principal.",
    "O artigo de referência é brasileiro; a revisão de Mangold aponta a falta de calibração e validação externa.",
]


# ------------------------------------------------------------------ XML
def paragrafos(itens):
    xs = []
    for i, (tipo, txt) in enumerate(itens):
        estilo_p = "P5" if i == 0 else ("P7" if i == len(itens) - 1 else "P6")
        estilo_t = "T3" if tipo == "q" else "T4"
        xs.append(f'<text:p text:style-name="{estilo_p}"><text:span text:style-name="{estilo_t}">'
                  f"{escape(txt)}</text:span></text:p>")
    return "".join(xs)


def trocar_caixa(xml: str, nome: str, conteudo: str) -> str:
    padrao = re.compile(r'(<draw:frame draw:name="%s".*?)(<draw:text-box>.*?</draw:text-box>|<draw:text-box/>)'
                        % re.escape(nome), re.S)
    novo, n = padrao.subn(lambda m: m.group(1) + f"<draw:text-box>{conteudo}</draw:text-box>", xml, count=1)
    assert n == 1, f"caixa não encontrada: {nome}"
    return novo


def gerar(modelo: Path) -> Path:
    with zipfile.ZipFile(modelo) as z:
        arquivos = {i: z.read(i) for i in z.namelist()}
    xml = arquivos["content.xml"].decode("utf-8")

    for nome, (estilo, txt) in TITULO.items():
        xml = trocar_caixa(xml, nome, f'<text:p text:style-name="P1"><text:span text:style-name="{estilo}">'
                                      f"{escape(txt)}</text:span></text:p>")
    for nome, itens in SLIDES.items():
        xml = trocar_caixa(xml, nome, paragrafos(itens))
    refs = "".join(f'<text:list-item><text:p text:style-name="P11"><text:span text:style-name="T4">{escape(r)}'
                   "</text:span></text:p></text:list-item>" for r in REFERENCIAS)
    xml = trocar_caixa(xml, "Google Shape;92;p19", f'<text:list text:style-name="L7">{refs}</text:list>')

    # Notas do apresentador: uma caixa de notas por slide, na ordem das páginas.
    notas = iter(NOTAS)
    xml = re.sub(r'(presentation:class="notes"[^>]*>)<draw:text-box/>',
                 lambda m: m.group(1) + f"<draw:text-box><text:p>{escape(next(notas))}</text:p></draw:text-box>",
                 xml)
    etree.fromstring(xml.encode("utf-8"))  # garante XML bem formado
    arquivos["content.xml"] = xml.encode("utf-8")

    # O mimetype deve ser o primeiro item e sem compressão (exigência do ODF).
    with zipfile.ZipFile(SAIDA, "w") as z:
        z.writestr(zipfile.ZipInfo("mimetype"), arquivos.pop("mimetype"), compress_type=zipfile.ZIP_STORED)
        for nome, dados in arquivos.items():
            z.writestr(nome, dados, compress_type=zipfile.ZIP_DEFLATED)
    return SAIDA


if __name__ == "__main__":
    print(gerar(Path(sys.argv[1])))
