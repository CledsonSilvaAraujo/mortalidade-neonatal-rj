# Predição de mortalidade neonatal no RJ com dados abertos (SINASC × SIM, 2017–2023)

Projeto de Aplicação — **Introdução ao Aprendizado de Máquina para a Saúde** (UFF, 2026.2, Prof. Flávio Luiz Seixas).
Aluno: Cledson da Silva Araújo.

- **Proposta:** [`proposta/proposta.pdf`](proposta/proposta.pdf).
- **Documentação (Parte 1):** este arquivo.
- **Demonstração (Parte 2):** [`notebook/mortalidade_neonatal_rj.ipynb`](notebook/mortalidade_neonatal_rj.ipynb).
- **Relatório (formato SBC, 12 páginas):** [`relatorio/main.pdf`](relatorio/main.pdf).
- **Slides (13, com notas do apresentador):** [`slides/apresentacao.pptx`](slides/apresentacao.pptx).

---

## 1. Motivação e objetivo

A mortalidade neonatal (óbito de 0 a 27 dias de vida) é a maior parte da mortalidade infantil no Brasil. No estado do RJ, a taxa foi de **8,62 óbitos por mil nascidos vivos** entre 2017 e 2023. A Declaração de Nascido Vivo (DN) é preenchida para todo nascimento, então um escore de risco calculado a partir dela pode apoiar a priorização de UTI neonatal, o transporte e a visita domiciliar.

O **artigo de referência** é Batista AFM, Diniz CSG, Bonilha EA, Kawachi I, Chiavegatto Filho ADP. *Neonatal mortality prediction with routinely collected data: a machine learning approach.* BMC Pediatrics 21:322, 2021 ([doi:10.1186/s12887-021-02788-9](https://doi.org/10.1186/s12887-021-02788-9)).
- Dados do município de São Paulo, com nascimentos de 2012–2017 pareados aos óbitos com dados identificados.
- Validação temporal: treino em 2012–2016 e teste em 2017.
- O XGBoost teve **AUROC 0,971**, AUPRC 0,586 e F1 0,548, com mais de 90% dos óbitos entre os 5% de maior risco.

Lacunas do artigo:
- ele depende de dados identificados;
- usa uma única cidade;
- depende de variáveis do pós-parto;
- não avalia calibração nem subgrupos, lacunas também apontadas pela revisão de Mangold et al. (2021).

**Perguntas de pesquisa**
1. O desempenho se reproduz no estado do RJ **só com microdados públicos**, que não têm identificadores? (H1: AUROC ≥ 0,95)
2. Quanto se perde com um modelo **pré-parto**, que usa apenas o que se sabe na admissão? (H2: queda substancial, com o modelo ainda útil)
3. O modelo é **calibrado e equitativo** por raça/cor e escolaridade da mãe e entre capital e interior? (H3: pior na baixa escolaridade)

## 2. Dataset

### 2.1 Descrição

| | SINASC (nascidos vivos) | SIM (óbitos) |
|---|---|---|
| Responsável | Ministério da Saúde / DATASUS | Ministério da Saúde / DATASUS |
| Arquivos | `DNRJ2017.dbc` … `DNRJ2023.dbc` (7) | `DORJ2017.dbc` … `DORJ2024.dbc` (8) |
| Recorte | residentes no estado do RJ | residentes no estado do RJ |
| Registros | 1.397.216 nascidos vivos | 12.039 óbitos neonatais de nascidos em 2017–2023 |
| Link | <ftp://ftp.datasus.gov.br/dissemin/publicos/SINASC/1996_/Dados/DNRES/> | <ftp://ftp.datasus.gov.br/dissemin/publicos/SIM/CID10/DORES/> |

- **Download automático:** `src/download.py` baixa tudo e guarda cache em `data/raw/`, sem cadastro nem passo manual.
- **Por que o SIM vai até 2024:** para captar os óbitos de quem nasceu no fim de 2023.
- **Desfecho:** óbito não fetal (`TIPOBITO = 2`) com 0 a 27 dias completos de vida, calculados como `DTOBITO − DTNASC`.

### 2.2 Linkage SINASC × SIM sem identificadores (`src/linkage.py`)

Os arquivos públicos **não trazem o número da DN** (`NUMERODN`), o nome da mãe nem o endereço. A Declaração de Óbito de menores de 1 ano, porém, repete dados do nascimento. Por isso, cada óbito é pareado a um nascido vivo por **concordância exata em chaves compostas, aplicadas em cascata**. Em cada passada, só são aceitos pares **1:1**: a chave tem de ser única entre os nascidos ainda não pareados e entre os óbitos ainda não pareados.

| Passada | Chave | Pares | Falsos pares estimados* |
|---|---|---|---|
| 1 | data + sexo + município + peso + idade da mãe + semanas + gravidez + parto | 5.338 | 0% (referência) |
| 2 | data + sexo + município + peso + idade da mãe | 3.521 | 6–8% |
| 3 | data + sexo + município + peso + estabelecimento | 771 | 17–32% |
| 4 | data + sexo + peso + idade da mãe | 439 | 11% |
| 5 | data + sexo + município + idade da mãe + semanas | 492 | 13–16% |
| 6 | data + sexo + município + estabelecimento + idade da mãe | 288 | 22% |

\*Estimativa por **mistura** em campos que nunca entram nas chaves (escolaridade da mãe e perdas fetais anteriores): f = (c_verd − c_k)/(c_verd − c_acaso).
- c_verd é a concordância na passada 1.
- c_acaso é a concordância entre pares sorteados com a mesma data e o mesmo sexo.

Duas chaves mais tolerantes foram testadas e **descartadas** por excesso de falsos pares: data + sexo + município + peso, e data + sexo + município + estabelecimento. A última pareia neonatos transferidos com outro recém-nascido do hospital onde morreram.

**Resultado do linkage**
- **10.849 de 12.039 óbitos pareados (90,1%)**, entre 88,5% e 91,6% conforme o ano.
- Maia et al. (2017), com dados **identificados** das capitais, obtiveram 90,8%.
- A taxa de mortalidade neonatal na base pareada é de 7,76‰, contra 8,62‰ no SIM.

### 2.3 Preparação dos dados (`src/preprocess.py`)

- **Numéricas (13):**
  - idade da mãe e do pai;
  - filhos vivos, perdas fetais, gestações, partos vaginais e cesáreas anteriores;
  - semanas de gestação;
  - consultas e mês de início do pré-natal;
  - Apgar 1 e 5;
  - peso.

  O código de "ignorado" (99, ou 9999 no peso) vira ausente.
- **Categóricas (14):**
  - local de nascimento;
  - estado civil, escolaridade (2010) e raça/cor da mãe;
  - tipo de gravidez e de parto;
  - sexo;
  - anomalia congênita;
  - apresentação;
  - indução;
  - cesárea antes do trabalho de parto;
  - quem assistiu o parto;
  - grupo de Robson;
  - índice de Kotelchuck.

  O "9" (ignorado) e o campo vazio viram **categorias próprias**, porque o não preenchimento também é informativo.
- **Derivada:** residência na capital.
- **Exclusão do artigo:** idade gestacional fora de 15–45 semanas. Nenhum registro foi excluído, porque o SINASC já valida o campo.
- **Imputação e encoding:**
  - no XGBoost e no LightGBM, ausentes e categorias são tratados de forma nativa;
  - na regressão logística, há imputação pela mediana com indicador de ausência, padronização e one-hot, **ajustados só no treino** (dentro do `Pipeline`).

| Fluxo | n |
|---|---|
| Nascidos vivos residentes no RJ, 2017–2023 | 1.397.216 |
| Com data de nascimento válida | 1.397.216 |
| Idade gestacional entre 15 e 45 semanas (ou ausente) | 1.397.216 |
| Óbitos neonatais pareados (positivos) | 10.849 (0,78%) |

## 3. Modelo de aprendizado de máquina

### 3.1 Conjuntos de variáveis

- **(A) Completo, 28 variáveis:** as da DN, como no artigo.
- **(B) Pré-parto, 17 variáveis:** o que se sabe **na admissão para o parto**.
  - Entram: mãe, pai, histórico obstétrico, idade gestacional, gravidez, apresentação, pré-natal, Kotelchuck e capital.
  - Saem: Apgar, peso, anomalia, sexo, local e tipo de parto, indução, cesárea antes do trabalho de parto, quem assistiu e grupo de Robson.

### 3.2 Formalização

- **Regressão logística** (linha de base):
  - P(y = 1 | **x**) = σ(β₀ + **β**ᵀ**x**), com σ(z) = 1/(1 + e⁻ᶻ);
  - penalização L2, com C ∈ {0,01; 0,1; 1} escolhido na validação;
  - `class_weight="balanced"`.
- **XGBoost** (modelo do artigo) e **LightGBM**:
  - F(**x**) = Σₖ fₖ(**x**), com K árvores de regressão ajustadas em sequência (gradient boosting);
  - P(y = 1 | **x**) = σ(F(**x**));
  - cada árvore minimiza a expansão de 2ª ordem da log-loss + Ω(f) = γT + ½λ‖w‖² + α‖w‖₁.
- **Desbalanceamento** (prevalência de ~0,8%):
  - `scale_pos_weight` é tratado como **hiperparâmetro**, com as opções nenhum, √(n₀/n₁) ou n₀/n₁;
  - a busca escolheu **nenhum** nos dois conjuntos;
  - não se usa sobreamostragem, para preservar a calibração.
- **Recalibração:**
  - uma regressão isotônica ajustada em 2022 corrige as probabilidades;
  - como ela é uma função em degraus e cria empates, as métricas de **ordenação** (AUROC, AUPRC e top 5%) usam o escore bruto;
  - as métricas de **valor** (Brier, O/E e as métricas no limiar) usam a probabilidade recalibrada.
- **Complexidade:**
  - XGBoost com histogramas: O(K·n·d) por ajuste e O(K·profundidade) por predição;
  - regressão logística: O(n·d') por iteração do L-BFGS.

### 3.3 Método de validação

A **validação é temporal**, espelhando o artigo (treino no passado, teste no ano seguinte):

| Amostra | Anos | Nascidos vivos | Óbitos | TMN (‰) | Uso |
|---|---|---|---|---|---|
| Ajuste | 2017–2021 | 1.040.702 | 8.017 | 7,70 | treino |
| Validação | 2022 | 180.369 | 1.388 | 7,70 | Optuna, early stopping, isotônica, limiar |
| Teste | 2023 | 176.145 | 1.444 | 8,20 | avaliação final, **usada uma única vez** |

**Hiperparâmetros:**
- **Busca:** otimização bayesiana (Optuna/TPE, semente 42) maximizando a **AUPRC** na validação.
- **Tentativas:** 15 para o XGBoost e 10 para o LightGBM, com early stopping de 50 árvores.

XGBoost final:

| | A · completo | B · pré-parto |
|---|---|---|
| `learning_rate` | 0,059 | 0,041 |
| `max_depth` | 5 | 4 |
| `min_child_weight` | 5,95 | 18,08 |
| `subsample` / `colsample_bytree` | 0,89 / 0,52 | 0,73 / 0,63 |
| `reg_lambda` / `reg_alpha` / `gamma` | 0,114 / 0,234 / 0,001 | 0,666 / 2,248 / 0,162 |
| `scale_pos_weight` | 1 | 1 |
| árvores (early stopping) | 207 | 409 |

### 3.4 Medidas de desempenho

As métricas são calculadas no teste de 2023, com **IC 95% por 500 reamostras bootstrap estratificadas**:
- AUROC e **AUPRC** (métrica principal, dada a prevalência);
- F1, precisão, sensibilidade, especificidade e VPN, no limiar que maximiza o F1 **em 2022**;
- Brier e razão observado/esperado (O/E);
- **% dos óbitos entre os 5% de maior risco**, métrica comparável ao artigo.

Análises complementares:
- ΔAUROC e ΔAUPRC por **bootstrap pareado**;
- SHAP (`TreeExplainer`) em 20.000 nascimentos de 2023;
- subgrupos com IC por bootstrap;
- **sensibilidade ao linkage**, refazendo o modelo só com as passadas 1–2.

## 4. Avaliação

### 4.1 Resultados no teste de 2023

| Conj. | Modelo | AUROC | AUPRC | F1 | Precisão | Sensib. | Top 5% | O/E |
|---|---|---|---|---|---|---|---|---|
| A | LR | 0,953 (0,946–0,961) | 0,533 (0,509–0,563) | 0,546 | 0,524 | 0,570 | 86% | — |
| A | **XGBoost** | **0,956 (0,948–0,962)** | **0,576 (0,553–0,605)** | 0,595 | 0,633 | 0,562 | 87% | 1,02 |
| A | LightGBM | 0,956 (0,949–0,963) | 0,573 (0,551–0,600) | 0,580 | 0,565 | 0,596 | 87% | — |
| B | LR | 0,887 (0,875–0,898) | 0,382 (0,357–0,413) | 0,476 | 0,475 | 0,477 | 71% | — |
| B | **XGBoost** | **0,890 (0,879–0,901)** | **0,406 (0,380–0,437)** | 0,478 | 0,487 | 0,470 | 72% | 1,05 |
| B | LightGBM | 0,891 (0,880–0,902) | 0,399 (0,372–0,429) | 0,469 | 0,464 | 0,474 | 71% | — |
| — | *Batista et al. (SP, 2017)* | *0,971* | *0,586* | *0,548* | *0,729* | *0,440* | *> 90%* | — |

A tabela completa, com especificidade, VPN, Brier e todos os IC, está em [`figuras/metricas_teste.csv`](figuras/metricas_teste.csv).

**Diferenças (bootstrap pareado):**
- **XGBoost A − B:** ΔAUROC = 0,066 (IC 0,056–0,076) e ΔAUPRC = 0,169 (IC 0,145–0,192).
- **XGBoost − LR, conjunto A:** ΔAUROC = 0,003 (IC 0,000–0,005) e ΔAUPRC = 0,042 (IC 0,027–0,059).

**SHAP:**
- No modelo **A**, as cinco principais variáveis são Apgar 5, peso, Apgar 1, semanas de gestação e anomalia congênita. São as mesmas cinco do artigo.
- No modelo **B**, são semanas de gestação, consultas de pré-natal, escolaridade da mãe, apresentação e raça/cor da mãe.

**Calibração:**
- O O/E é de 1,02 (A) e 1,05 (B) após a isotônica, contra 0,99 e 1,02 sem ela.
- No modelo A, a AUROC por subgrupo varia de 0,950 a 0,962, com IC sobrepostos.
- No modelo B, o risco é subestimado para filhos de **mães brancas** (O/E 1,15; IC 1,05–1,24) e para o **interior** (O/E 1,08; IC 1,02–1,15).
- Ainda no modelo B, a AUROC é numericamente menor para mães com fundamental incompleto: 0,868 (0,841–0,895) contra 0,892 (0,865–0,918) no ensino superior.

**Sensibilidade ao linkage:**
- Só com as passadas 1–2, a AUROC é de 0,959 (A) e 0,891 (B), contra 0,956 e 0,890 com todas as passadas.
- Os falsos pares das passadas tolerantes não explicam os resultados.

### 4.2 Figuras (geradas pelo notebook em `figuras/`)

| | |
|---|---|
| ![](figuras/roc_pr.png) | ![](figuras/captura_top.png) |
| ![](figuras/shap_importancia.png) | ![](figuras/subgrupos.png) |

## 5. Conclusão

- **H1 confirmada.** Com dados totalmente abertos e linkage sem identificadores, o XGBoost teve AUROC de **0,956** e concentrou **87%** dos óbitos nos 5% de maior risco, próximo do artigo (0,971; > 90%). As principais variáveis foram as mesmas. A diferença residual é compatível com três fatores:
  - os ~10% de óbitos não pareados;
  - a população estadual, mais heterogênea do que a de um único município;
  - a qualidade de preenchimento da DN no interior.
- **H2 confirmada em parte.** O modelo pré-parto perde discriminação (ΔAUROC = 0,066) e muita AUPRC, mas mantém **AUROC 0,890** e captaria **72%** dos óbitos priorizando 5% das gestantes na admissão. Isso é útil para regular gestantes de alto risco para maternidades com UTI neonatal.
- **H3 com apoio fraco.** O modelo completo é homogêneo entre os subgrupos. No pré-parto, as diferenças de AUROC têm IC sobrepostos, mas o risco é subestimado para mães brancas e para o interior.
- **Limitações:**
  - ~10% dos óbitos não pareados ficam rotulados como sobreviventes;
  - a validação é temporal, mas não externa, porque usa um único estado;
  - no modelo pré-parto, a idade gestacional é a do parto;
  - não havia óbitos suficientes para analisar mães amarelas e indígenas.
- **Trabalhos futuros:**
  - validação em outras UFs;
  - etapa probabilística no linkage;
  - extensão ao óbito pós-neonatal;
  - idade gestacional estimada no pré-natal.

## 6. Reprodução

```bash
uv venv .venv -p 3.12 && uv pip install -p .venv -r requirements.txt
.venv/bin/python -m pytest                      # 16 testes (linkage, preprocess, modelos, avaliação)
.venv/bin/python -m src.experimento             # download + linkage + modelos + avaliação (~36 min, 32 núcleos)
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebook/mortalidade_neonatal_rj.ipynb
.venv/bin/python relatorio/gerar_numeros.py && (cd relatorio && latexmk -pdf main.tex)
.venv/bin/python slides/gerar_slides.py
```

| Caminho | Conteúdo |
|---|---|
| `src/download.py` | FTP do DATASUS → DBC → parquet (cache, `.part`, novas tentativas) |
| `src/linkage.py` | chaves em cascata, pares 1:1, estimativa de falsos pares |
| `src/preprocess.py` | limpeza, conjuntos A/B, divisão temporal |
| `src/models.py` | LR, XGBoost, LightGBM, Optuna, recalibração isotônica |
| `src/evaluate.py` | métricas, bootstrap, ΔAUC pareado, calibração, subgrupos |
| `src/experimento.py` | experimento completo → `data/resultados.pkl` |
| `relatorio/gerar_numeros.py` | macros LaTeX com os números (texto e resultados não divergem) |

Os números deste README, do relatório e dos slides saem todos de `data/resultados.pkl`.
