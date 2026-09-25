# Predição de mortalidade neonatal no RJ com dados abertos (SINASC × SIM, 2017–2024)

Projeto de Aplicação — **Introdução ao Aprendizado de Máquina para a Saúde** (UFF, 2026.2, Prof. Flávio Luiz Seixas).
Aluno: Cledson da Silva Araújo.

> Este README é gerado por `relatorio/gerar_readme.py` a partir de `README.tmpl.md`. Os números vêm das mesmas macros do relatório: edite o template, não este arquivo.

- **Proposta:** [`proposta/proposta.pdf`](proposta/proposta.pdf).
- **Documentação (Parte 1):** este arquivo.
- **Demonstração (Parte 2):** [`notebook/mortalidade_neonatal_rj.ipynb`](notebook/mortalidade_neonatal_rj.ipynb).
- **Relatório (formato SBC, 12 páginas):** [`relatorio/main.pdf`](relatorio/main.pdf).
- **Slides (16, com notas do apresentador, ~14 min de fala):** [`slides/apresentacao.pptx`](slides/apresentacao.pptx).
- **Apresentação do tema (modelo .odp da disciplina):** [`slides/apresentacao_tema.odp`](slides/apresentacao_tema.odp).
- **Roteiro de arguição:** [`slides/arguicao.md`](slides/arguicao.md).
- **Repositório:** <https://github.com/CledsonSilvaAraujo/mortalidade-neonatal-rj>.

---

## 1. Motivação e objetivo

A mortalidade neonatal (óbito de 0 a 27 dias de vida) é a maior parte da mortalidade infantil no Brasil. No estado do RJ, a taxa foi de **8,63 óbitos por mil nascidos vivos** entre 2017 e 2024. A Declaração de Nascido Vivo (DN) é preenchida para todo nascimento, então um escore de risco calculado a partir dela pode apoiar a priorização de UTI neonatal, o transporte e a visita domiciliar.

O **artigo de referência** é Batista AFM, Diniz CSG, Bonilha EA, Kawachi I, Chiavegatto Filho ADP. *Neonatal mortality prediction with routinely collected data: a machine learning approach.* BMC Pediatrics 21:322, 2021 ([doi:10.1186/s12887-021-02788-9](https://doi.org/10.1186/s12887-021-02788-9)).
- Dados do município de São Paulo, com nascimentos de 2012–2017 pareados aos óbitos com dados identificados.
- Validação temporal: treino em 2012–2016 e teste em 2017.
- O XGBoost teve **AUROC 0,971**, AUPRC 0,586 e F1 0,548, com mais de 90% dos óbitos entre os 5% de maior risco.

Lacunas do artigo:
- ele depende de dados identificados;
- usa uma única cidade;
- depende de variáveis do pós-parto;
- não avalia calibração nem subgrupos, lacunas também apontadas pela revisão de Mangold et al. (2021).

**Perguntas de pesquisa** (hipóteses como na proposta)
1. O desempenho se reproduz no estado do RJ **só com microdados públicos**, que não têm identificadores? (H1: AUROC ≥ 0,95)
2. Quanto se perde com um modelo **pré-parto**, que usa apenas o que se sabe na admissão? (H2: queda substancial, com o modelo ainda útil)
3. O modelo é **calibrado e equitativo** por raça/cor e escolaridade da mãe e entre capital e interior? (H3: **a calibração** piora nos subgrupos de menor escolaridade)

## 2. Dataset

### 2.1 Descrição

| | SINASC (nascidos vivos) | SIM (óbitos) |
|---|---|---|
| Responsável | Ministério da Saúde / DATASUS | Ministério da Saúde / DATASUS |
| Arquivos | `DNRJ2017.dbc` … `DNRJ2024.dbc` | `DORJ2017.dbc` … `DORJ2025.dbc` |
| Recorte | residentes no estado do RJ | residentes no estado do RJ |
| Registros | 1.560.938 nascidos vivos | 13.471 óbitos neonatais de nascidos em 2017–2024 |
| Link | <ftp://ftp.datasus.gov.br/dissemin/publicos/SINASC/1996_/Dados/DNRES/> | <ftp://ftp.datasus.gov.br/dissemin/publicos/SIM/CID10/DORES/> (e `SIM/PRELIM/DORES/` para 2025) |

- **Download automático:** `src/download.py` baixa tudo e guarda cache em `data/raw/`, sem cadastro nem passo manual.
- **Por que o SIM vai até 2025:** para captar os óbitos de quem nasceu no fim de 2024.
  - O SIM 2025 ainda é **preliminar** e só contribui com esses óbitos.
- **Desfecho:** óbito não fetal (`TIPOBITO = 2`) com 0 a 27 dias completos de vida, calculados como `DTOBITO − DTNASC`.

### 2.2 Linkage SINASC × SIM sem identificadores (`src/linkage.py`)

Os arquivos públicos **não trazem o número da DN** (`NUMERODN`), o nome da mãe nem o endereço. A Declaração de Óbito de menores de 1 ano, porém, repete dados do nascimento. Por isso, cada óbito é pareado a um nascido vivo por **concordância exata em chaves compostas, aplicadas em cascata**. Em cada passada, só são aceitos pares **1:1**: a chave tem de ser única entre os nascidos ainda não pareados e entre os óbitos ainda não pareados.

| Passada | Chave | Pares | Falsos pares estimados* |
|---|---|---|---|
| 1 | data + sexo + município + peso + idade da mãe + semanas + gravidez + parto | 6.043 | 0% (referência) |
| 2 | data + sexo + município + peso + idade da mãe | 3.917 | 6–8% |
| 3 | data + sexo + município + peso + estabelecimento | 846 | 17–31% |
| 4 | data + sexo + peso + idade da mãe | 479 | 12–16% |
| 5 | data + sexo + município + idade da mãe + semanas | 547 | 12–14% |
| 6 | data + sexo + município + estabelecimento + idade da mãe | 323 | 21–22% |

\*Estimativa por **mistura** em campos que nunca entram nas chaves (escolaridade da mãe e perdas fetais anteriores): f = (c_verd − c_k)/(c_verd − c_acaso).
- c_verd é a concordância na passada 1.
- c_acaso é a concordância entre pares sorteados com a mesma data e o mesmo sexo.

Duas chaves mais tolerantes foram testadas e **descartadas** por excesso de falsos pares: data + sexo + município + peso, e data + sexo + município + estabelecimento. A última pareia neonatos transferidos com outro recém-nascido do hospital onde morreram.

**Resultado do linkage**
- **12.155 de 13.471 óbitos pareados (90,2%)**, entre 88,5% e 91,6% conforme o ano.
- Descontados os falsos pares estimados (média ponderada de 6%), cerca de **85% dos óbitos têm um par verdadeiro**.
- **Premissa da estimativa:** a passada 1 (chave completa, 8 campos) não tem falsos pares. Se tiver, as demais estimativas estão subestimadas.
- Pareamento por região: **94,3% na capital** e **88,2% no interior**.
  - Por raça/cor do **recém-nascido** (a única que a DO informa), vai de 89,4% a 90,6%.
  - A raça/cor da **mãe**, usada nos subgrupos, não está na DO.
- **Completude por ano e região:** é estável entre os anos.

  | Região | Pareamento 2017–2023 | Pareamento 2024 | O/E esperado só pelo linkage | O/E A / B (recalibrado) |
  |---|---|---|---|---|
  | Capital | 94,3% | 94,6% | 1,003 | 0,91 / 0,94 |
  | Interior | 88,0% | 89,6% | 1,018 | 0,98 / 0,97 |

  Só pelo linkage, o O/E se desloca menos de 2% entre as regiões, bem menos que a diferença observada.
- A comparação com Maia et al. (2017) exige cautela: eles tiveram 90,8% nas capitais (93,1% nas capitais do Sudeste), mas com óbitos **infantis** e dados **identificados**.
- A taxa de mortalidade neonatal na base pareada é de 7,79‰, contra 8,63‰ no SIM.

**Quem fica sem par** (`src/complementares.py`, comparação pelas informações da própria DO):

| Na Declaração de Óbito | Pareados | Não pareados |
|---|---|---|
| n | 12.155 | 1.316 |
| Peso ausente | 0,8% | 26,8% |
| Semanas de gestação ausentes | 4,8% | 31,5% |
| Idade da mãe ausente | 2,1% | 32,2% |
| Peso mediano | 1100 g | 1400 g |
| Residente na capital | 35,0% | 19,6% |
| Óbito em hospital | 98,2% | 90,2% |

- Os não pareados são óbitos com campos das chaves ausentes na DO: só 20,7% dos óbitos sem peso foram pareados.
- O perfil deles é **diferente**, e não necessariamente menos grave: mais campos ausentes, mais do interior e mais óbitos no 1º dia e fora do hospital (óbito precoce fora do hospital ou em trânsito).
- O peso mediano maior dos não pareados é uma comparação com seleção: ele só é calculado entre quem tem peso preenchido.
- Isso **não** tornou a ausência na DN um sinal de sobrevivência: na base pareada, a TMN é **maior** quando o campo falta (seção 3.5).

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

Base analítica: 1.560.938 nascidos vivos, com 12.155 óbitos neonatais pareados (positivos, 0,78%).

### 2.4 Versão dos dados e aspectos éticos

- **Versão:** [`manifesto_dados.csv`](manifesto_dados.csv) registra, para cada arquivo, URL, situação (definitivo/preliminar), tamanho, **SHA-256** e data do download. É gerado por `download.manifesto()`.
- **Situação dos dados:** todos os arquivos são **definitivos**, exceto o SIM 2025, ainda **preliminar**.
- **Ética:** os dados são públicos e sem identificação nominal, o que dispensa CEP/Conep (Res. CNS 510/2016, art. 1º, parágrafo único).
- **Risco de reidentificação:** o próprio linkage cria esse risco, porque data, sexo, peso e município tornam o registro quase único. Por isso, o repositório publica o código, mas **não a base pareada** (`data/` está no `.gitignore`).

## 3. Modelo de aprendizado de máquina

### 3.1 Conjuntos de variáveis

- **(A) Completo, 28 variáveis:** as da DN, como no artigo.
- **(B) Pré-parto, 17 variáveis:** o que se sabe **na admissão para o parto**.
  - A idade gestacional é a do parto, e consultas e Kotelchuck dependem dela. Por isso, o modelo estima o risco *dado que o parto ocorre agora*; ele **não** serve para triagem semanas antes.
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
  - no XGBoost, a busca escolheu **√(n₀/n₁)** ≈ 11,35;
  - não se usa sobreamostragem.
- **Recalibração:**
  - uma regressão isotônica ajustada na validação (2023) corrige as probabilidades;
  - como ela é uma função em degraus e cria empates, as métricas de **ordenação** (AUROC, AUPRC e top 5%) usam o escore bruto;
  - as métricas de **valor** (Brier, O/E e as métricas no limiar) usam a probabilidade recalibrada.
- **Complexidade:**
  - XGBoost com histogramas: O(K·n·d) por ajuste e O(K·profundidade) por predição;
  - regressão logística: O(n·d') por iteração do L-BFGS.

### 3.3 Método de validação

A **validação é temporal**, espelhando o artigo (treino no passado, teste no ano seguinte):

| Amostra | Anos | Nascidos vivos | Óbitos | TMN (‰) | Uso |
|---|---|---|---|---|---|
| Ajuste | 2017–2022 | 1.221.071 | 9.405 | 7,70 | treino |
| Validação | 2023 | 176.145 | 1.444 | 8,20 | Optuna, early stopping, isotônica, limiar |
| Teste | 2024 | 163.722 | 1.306 | 7,98 | avaliação final, **usada uma única vez** |

**Hiperparâmetros:**
- **Busca:** otimização bayesiana (Optuna/TPE, semente 42) maximizando a **AUPRC** na validação.
- **Tentativas:** 15 para o XGBoost e 10 para o LightGBM, com early stopping de 50 árvores.

**Por que essas escolhas:**
- **Busca curta:** cada ajuste leva ~20 s com 10⁶ registros, e no XGBoost as cinco melhores tentativas ficaram a no máximo 0,005 de AUPRC da melhor, o que indica um platô (figura `optuna.png`).
- **Holdout temporal em vez de validação cruzada:** imita o uso real, de treinar no passado e aplicar no ano seguinte.
- **Validação reutilizada:** 2023 serve para early stopping, Optuna, isotônica e limiar, então as métricas **de validação** são otimistas. O teste de 2024 continua intocado.
- **Sem reajuste final com 2017–2023:** 2023 é necessário para a recalibração e o limiar.

XGBoost final:

| | A · completo | B · pré-parto |
|---|---|---|
| `learning_rate` | 0,036 | 0,036 |
| `max_depth` | 5 | 5 |
| `min_child_weight` | 1,57 | 1,57 |
| `subsample` / `colsample_bytree` | 0,93 / 0,77 | 0,93 / 0,77 |
| `reg_lambda` / `reg_alpha` / `gamma` | 0,021 / 0,002 / 0,014 | 0,021 / 0,002 / 0,014 |
| `scale_pos_weight` | 11,35 | 11,35 |
| árvores (early stopping) | 247 | 407 |

### 3.4 Medidas de desempenho

As métricas são calculadas no teste de 2024, com **IC 95% por 500 reamostras bootstrap estratificadas**:
- AUROC e **AUPRC** (métrica principal, dada a prevalência);
- F1, precisão, sensibilidade, especificidade e VPN, no limiar que maximiza o F1 **na validação**;
- Brier e razão observado/esperado (O/E);
- **% dos óbitos entre os 5% de maior risco**, métrica comparável ao artigo.

Análises complementares (`src/complementares.py`):
- ΔAUROC e ΔAUPRC por **bootstrap pareado**;
- subgrupos com IC por bootstrap;
- **curva, intercepto e inclinação de calibração** (Van Calster et al., 2019);
- **curva de decisão** e benefício líquido (Vickers e Elkin, 2006);
- **sensibilidade ao linkage**: rótulos alternativos no treino, avaliados no **mesmo** teste;
- XAI (`src/explicabilidade.py`, seção 4.2).

### 3.5 Avaliação das premissas

- **Colinearidade (regressão logística):** filhos vivos e gestações anteriores têm VIF de 20 e 22 (gestações ≈ partos + perdas); as demais variáveis têm VIF ≤ 4,6.
  - Isso afeta a interpretação dos coeficientes, não a predição.
  - A L2 estabiliza o ajuste, e as árvores não dependem dessa premissa. Os coeficientes não são interpretados.
- **Mecanismo dos ausentes:** a ausência não é aleatória. A TMN é maior quando o campo falta:
  - consultas de pré-natal: 19,7‰ contra 7,5‰;
  - Apgar 5: 27,6‰ contra 7,6‰;
  - idade do pai: 8,8‰ contra 6,6‰.

  Isso é compatível com MAR condicional à assistência ou com MNAR, e é o motivo de usar indicadores e categorias de ausência.
- **Independência:**
  - Gêmeos são 2,3% dos nascimentos e 10,2% dos óbitos.
  - O bootstrap os trata como independentes, o que estreita um pouco os IC.
  - Só entre únicos, a AUROC fica em 0,963 (A) e 0,882 (B), contra 0,964 e 0,891.

## 4. Avaliação

### 4.1 Resultados no teste de 2024

| Conj. | Modelo | AUROC | AUPRC | F1 | Precisão | Sensib. | Top 5% | O/E |
|---|---|---|---|---|---|---|---|---|
| A | LR | 0,963 (0,956–0,969) | 0,551 (0,526–0,579) | 0,554 | 0,534 | 0,577 | 88% | 0,93 |
| A | **XGBoost** | **0,964 (0,958–0,970)** | **0,603 (0,577–0,628)** | 0,604 | 0,597 | 0,610 | 88% | 0,95 |
| A | LightGBM | 0,964 (0,958–0,970) | 0,580 (0,552–0,608) | 0,596 | 0,610 | 0,583 | 88% | 0,94 |
| B | LR | 0,886 (0,874–0,896) | 0,366 (0,338–0,394) | 0,450 | 0,434 | 0,467 | 71% | 0,95 |
| B | **XGBoost** | **0,891 (0,879–0,902)** | **0,390 (0,360–0,418)** | 0,454 | 0,470 | 0,440 | 71% | 0,96 |
| B | LightGBM | 0,890 (0,879–0,901) | 0,396 (0,367–0,424) | 0,457 | 0,466 | 0,448 | 71% | 0,97 |
| — | *Batista et al. (SP, 2017)* | *0,971* | *0,586* | *0,548* | *0,729* | *0,440* | *> 90%* | — |

A tabela completa, com especificidade, VPN, Brier e todos os IC, está em [`figuras/metricas_teste.csv`](figuras/metricas_teste.csv).

**Diferenças (bootstrap pareado):**
- **XGBoost A − B:** ΔAUROC = 0,074 (IC 0,064–0,084) e ΔAUPRC = 0,214 (IC 0,191–0,241).
- **XGBoost − LR, conjunto A:**
  - em AUROC, ΔAUROC = 0,001 (IC −0,001–0,003), com IC que inclui zero: não há diferença;
  - em AUPRC, ΔAUPRC = 0,052 (IC 0,040–0,066): o XGBoost é mais preciso entre os de maior risco.

**Calibração** (curva em 20 faixas em `figuras/calibracao_decisao.png`):
- **Sem recalibração, o escore é muito mal calibrado.** Com o peso de classe escolhido na busca, o O/E bruto é de 0,23 (A) e 0,17 (B), com intercepto de -2,456 (−2,526 a −2,387).
- **A isotônica é indispensável:** leva o O/E a 0,95 e 0,96, e a curva recalibrada acompanha a diagonal.
- **Intercepto / inclinação (recalibrado):** −0,082 (−0,154 a −0,011) / 1,009 (0,981 a 1,037) no A; −0,060 (−0,124 a 0,004) / 0,979 (0,954 a 1,005) no B.
- A leve superestimação que resta é compatível com a variação anual da TMN (8,20‰ na validação e 7,98‰ no teste). É uma explicação plausível, não provada.

**Utilidade clínica (curva de decisão):**
- Com prevalência de 0,8%, "tratar todos" é negativo acima de p_t ≈ 0,8%. Superá-la é trivial, então a comparação relevante é com **não intervir**.
- **Faixa clínica de 2% a 10%:** para uma transferência *in utero* ou a priorização de UTI neonatal, é plausível aceitar de 9 a 49 falsos alarmes por óbito identificado.
- Nessa faixa, o benefício líquido vai de 6,2 a 4,9 por mil nascidos vivos (A) e de 4,7 a 3,3 por mil (B), sempre acima de zero. O máximo possível é a TMN, 7,98 por mil.

**H3, como na proposta:**
- No modelo B, o O/E dos filhos de mães com fundamental incompleto é 0,99 (0,90–1,09); no A, 0,99 (0,90–1,08). Os IC incluem 1, então **H3 não se confirma**.
- **Achado de equidade que se sustenta:** no modelo A, a *discriminação* é pior na baixa escolaridade. A AUROC é 0,950 (0,932–0,965), contra 0,980 (0,970–0,989) no ensino superior, e os IC não se sobrepõem.
- **Calibração por subgrupo é instável entre anos:**
  - no teste de 2023, o modelo B subestimava mães brancas (O/E 1,15) e o interior (1,08);
  - no teste de 2024, isso não se repetiu (1,02 (0,94–1,11) e 0,97 (0,91–1,02));
  - em 2024, o modelo A superestima levemente o risco em quase todos os subgrupos, mais na capital (0,91 (0,85–0,97)).

**Sensibilidade ao linkage** (mesmo teste de 2024; Δ por bootstrap pareado):

| Conj. | Rótulos no treino | AUROC | AUPRC | O/E | ΔAUROC vs principal |
|---|---|---|---|---|---|
| A | principal | 0,964 | 0,603 | 0,95 | — |
| A | estrito (sem passadas 3–6) | 0,963 | 0,601 | 1,07 | −0,0009 (−0,0020 a 0,0001) |
| A | pior caso (passadas 3–6 = 0) | 0,963 | 0,599 | 1,16 | −0,0013 (−0,0029 a 0,0005) |
| B | principal | 0,891 | 0,390 | 0,96 | — |
| B | estrito | 0,891 | 0,392 | 1,11 | 0,0006 (−0,0015 a 0,0029) |
| B | pior caso | 0,892 | 0,391 | 1,16 | 0,0013 (−0,0009 a 0,0034) |

- O modelo é **robusto a ruído de rótulo no treino**.
- A **calibração absoluta não é**: com menos positivos no treino, o modelo aprende uma prevalência menor.
- **O teste também tem rótulos imperfeitos.**
  - Sem os pares tolerantes do teste, a AUROC do modelo principal é 0,965 (A) e 0,890 (B).
  - Com esses pares marcados como sobreviventes, é 0,964 e 0,890.
  - Os pares tolerantes **não inflam** a AUROC reportada.
- Os ~10% de óbitos sem nenhum par não podem ser avaliados: é um limite da validação.

### 4.2 Explicabilidade (XAI)

O projeto usa o SHAP exato (TreeSHAP), com a decomposição logit p̂ = φ₀ + Σφⱼ, em 20.000 nascimentos de 2024 (`src/explicabilidade.py`).

1. **Verificação de aditividade:** φ₀ + Σφ reproduz o log-odds do modelo com erro de 7×10⁻6 (A).
   - A verificação expôs uma armadilha: com *early stopping*, a API do XGBoost usa por padrão **todas** as árvores, inclusive as 50 posteriores à melhor iteração.
   - É preciso passar `iteration_range`; do contrário, as explicações se referem a outro modelo. Há um teste de regressão para isso.
2. **Global:** no modelo A, o top 5 é peso ao nascer, Apgar 5º min, Apgar 1º min, semanas de gestação, anomalia congênita, as mesmas variáveis do artigo. No B, é semanas de gestação, consultas de pré-natal, escolaridade da mãe, apresentação, raça/cor da mãe.
3. **Dependência** (`figuras/xai_dependencia.png`): os efeitos são clinicamente plausíveis.
   - O risco cai de forma monotônica com Apgar, peso e idade gestacional.
   - O risco sobe com a idade da mãe acima de ~37 anos.
   - As **consultas de pré-natal têm forma de U**: acima de ~17 consultas, o risco sobe. É confusão por indicação, então o SHAP mostra associação, não causa.
4. **Local** (`figuras/xai_local.png`): três casos.
   - **Óbito bem previsto:** 670 g, 22 semanas, Apgar 1/1.
   - **Óbito não sinalizado (4,4%):** bebê a termo (39 semanas, 3.450 g) com anomalia. O indicador binário de anomalia não distingue malformações letais; o tipo (`CODANOMAL`) existe na DN e fica como trabalho futuro.
   - **"Sobrevivente" com risco de 100,0%:** 385 g e 21 semanas. É quase certamente um óbito não pareado: a explicação local expõe o ruído de rótulo do linkage.
5. **Estabilidade:**
   - Spearman do ranking XGBoost × LightGBM de 0,90 (A) e 0,98 (B);
   - entre metades da amostra, 0,999.
6. **Atributos sensíveis:**
   - Raça/cor e escolaridade da mãe pesam 3,0% da importância no A e 7,9% no B.
   - No B, mãe branca tem SHAP médio de −0,11 e mãe preta, 0,06; a escolaridade vai de 0,33 (fundamental incompleto) a −0,27 (superior).
   - O modelo pré-parto usa o gradiente socioeconômico como substituto de parte do risco.

### 4.3 Figuras (geradas pelo notebook em `figuras/`)

| | |
|---|---|
| ![](figuras/roc_pr.png) | ![](figuras/captura_top.png) |
| ![](figuras/shap_importancia.png) | ![](figuras/subgrupos.png) |
| ![](figuras/xai_dependencia.png) | ![](figuras/xai_local.png) |
| ![](figuras/calibracao_decisao.png) | |

## 5. Conclusão

- **H1 confirmada.** Com dados totalmente abertos e linkage sem identificadores, o XGBoost teve AUROC de **0,964** e concentrou **88%** dos óbitos nos 5% de maior risco, contra 0,971 e mais de 90% no artigo.
  - As principais variáveis foram as mesmas.
  - O modelo é **robusto a ruído de rótulo no treino**, e os pares duvidosos do teste não inflam a AUROC. Não é possível descartar um viés vindo dos óbitos sem par.
- **H2 confirmada em parte.** O modelo pré-parto perde discriminação (ΔAUROC = 0,074), mas mantém **AUROC 0,891** e benefício líquido positivo.
  - **Uso adequado:** na admissão, com o parto iminente, para decidir transferência *in utero*.
  - Não serve para triagem precoce da gestação.
- **H3 não confirmada.** A calibração na baixa escolaridade não é pior.
  - O que se sustenta é uma **discriminação menor na baixa escolaridade** (modelo A).
  - As diferenças de calibração por subgrupo mudam de um ano para outro.
  - O linkage diferencial por região desloca o O/E em menos de 2%.
- **XAI:** as explicações são aditivas (verificadas), estáveis entre algoritmos e amostras e clinicamente plausíveis. Elas expõem erro de rótulo e a falta do tipo de anomalia.
- **Calibração:** depende inteiramente da recalibração (o escore bruto tem O/E de 0,23) e da estabilidade da TMN; um uso real exigiria recalibração periódica.
- **Limitações:**
  - ~15% dos óbitos sem par verdadeiro;
  - validação temporal, mas não externa;
  - o modelo A usa variáveis posteriores ao nascimento;
  - o bootstrap ignora a dependência entre gêmeos;
  - a calibração por subgrupo é instável;
  - o SIM 2025 é preliminar;
  - não havia óbitos suficientes para analisar mães amarelas e indígenas.
- **Trabalhos futuros:**
  - validação em outras UFs;
  - etapa probabilística no linkage;
  - uso do tipo de anomalia (`CODANOMAL`);
  - estudo das diferenças regionais e anuais de calibração;
  - avaliação de impacto com gestores.

## 6. Reprodução

```bash
uv venv .venv -p 3.12 && uv pip install -p .venv -r requirements.txt
.venv/bin/python -m pytest                      # testes (download, linkage, preprocess, modelos, avaliação, complementares, XAI)
.venv/bin/python -m src.experimento             # download + linkage + modelos + avaliação + complementares + XAI (~60 min, 32 núcleos)
.venv/bin/python -m src.experimento --complementares   # refaz só as análises complementares (~20 min)
.venv/bin/python -m src.experimento --atualizar        # só as partes baratas + XAI (~5 min; modelos em cache)
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebook/mortalidade_neonatal_rj.ipynb
.venv/bin/python relatorio/gerar_numeros.py && (cd relatorio && latexmk -pdf main.tex)
.venv/bin/python relatorio/gerar_readme.py      # README.md a partir de README.tmpl.md
.venv/bin/python slides/gerar_slides.py
```

O período é definido em `src/download.py` (`ANOS_NASCIMENTO`, `ANOS_OBITO`, `PRELIMINARES`) e em `src/preprocess.py` (`ANO_VALIDACAO`, `ANO_TESTE`). Ao mudá-lo, apague `data/base_analitica.parquet` e `data/modelos/` antes de rodar.

| Caminho | Conteúdo |
|---|---|
| `src/download.py` | FTP do DATASUS → DBC → parquet (cache, `.part`, novas tentativas); manifesto com SHA-256 |
| `src/linkage.py` | chaves em cascata, pares 1:1, estimativa de falsos pares |
| `src/preprocess.py` | limpeza, conjuntos A/B, divisão temporal |
| `src/models.py` | LR, XGBoost, LightGBM, Optuna, recalibração isotônica |
| `src/evaluate.py` | métricas, bootstrap, ΔAUC pareado, calibração, subgrupos |
| `src/complementares.py` | viés do linkage, sensibilidade, calibração, curva de decisão, premissas |
| `src/explicabilidade.py` | XAI: SHAP exato, dependência, explicações locais, estabilidade, atributos sensíveis |
| `src/experimento.py` | experimento completo → `data/resultados.pkl` |
| `manifesto_dados.csv` | URL, situação, tamanho, SHA-256 e data de cada arquivo do DATASUS |
| `relatorio/gerar_numeros.py` | macros LaTeX (e `macros.json`) com os números |
| `relatorio/gerar_readme.py` | preenche este README a partir das mesmas macros |
| `slides/gerar_slides.py`, `slides/gerar_apresentacao_tema.py` | slides e apresentação do tema |

Os números deste README, do relatório e dos slides saem todos de `data/resultados.pkl`.
