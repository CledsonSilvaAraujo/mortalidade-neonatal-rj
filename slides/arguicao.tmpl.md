# Roteiro de arguição — perguntas prováveis

> Gerado por `relatorio/gerar_readme.py` a partir de `slides/arguicao.tmpl.md`. Os números são os mesmos do relatório (teste de {{anoTeste}}).

## 1. "Os erros do linkage não explicam o resultado?"

- O modelo foi reajustado com rótulos alternativos **no treino**, e a avaliação foi sempre no **mesmo** teste de {{anoTeste}}:
  - **Estrito:** sai quem foi pareado nas passadas 3–6.
  - **Pior caso:** essas pessoas ficam, mas marcadas como sobreviventes.
- ΔAUROC = {{sensAEstDelta}} no estrito e {{sensAPiorDelta}} no pior caso, por bootstrap pareado.
- O modelo é **robusto a ruído de rótulo no treino**. Não diga "insensível aos erros do linkage": o teste também tem rótulos imperfeitos.
- **E o teste?** Sem os pares tolerantes do teste, a AUROC do modelo principal é {{testeEstrAAuroc}}; com esses pares zerados, é {{testeZeroAAuroc}} (reportado: {{AXgbAuroc}}). Eles não inflam o resultado.
- Os ~10% de óbitos sem nenhum par não podem ser avaliados, e isso é um limite admitido.
- A **calibração absoluta não é robusta**: o O/E sobe para {{sensAEstOE}}–{{sensAPiorOE}}. Com menos positivos no treino, o modelo aprende uma prevalência menor.
- **Falsos pares:** estimativa de {{falsosPonderado}} em média, supondo 0% na passada 1 (chave com 8 campos). Isso deixa ≈{{pctVerdadeiros}} dos óbitos com par verdadeiro.

## 2. "E os ~10% de óbitos não pareados, rotulados como sobreviventes?"

- Pela própria DO, os não pareados têm campos das chaves ausentes:
  - peso: {{nparPesoAus}}% contra {{parPesoAus}}%;
  - semanas: {{nparSemAus}}% contra {{parSemAus}}%;
  - idade da mãe: {{nparIdAus}}% contra {{parIdAus}}%.
- O perfil é **diferente**: mais do interior, mais óbitos no 1º dia e menos em hospital (óbito precoce fora do hospital ou em trânsito).
  - **Não diga** que são "menos graves": o peso mediano maior deles só é calculado entre quem tem peso, então é uma comparação com seleção.
- O pareamento é de {{linkCap}} na capital e {{linkInt}} no interior. Por raça/cor do recém-nascido, é quase uniforme ({{linkRacaMin}}–{{linkRacaMax}}).
- A ausência na DN **não** virou sinal de sobrevivência: na base pareada, a TMN é **maior** quando o campo falta (consultas: {{ausCons}}‰ contra {{preCons}}‰).
- **O linkage explica diferenças de calibração entre regiões? Não.**
  - Em {{anoTeste}}, a completude subiu {{compDifCap}} ponto na capital e {{compDifInt}} no interior.
  - Só pelo linkage, o O/E se deslocaria {{oeLinkCap}} e {{oeLinkInt}}: menos de 2% de diferença.
- **E a raça da mãe?** A DO só informa a raça do recém-nascido, então a evidência sobre o linkage por raça materna é indireta.

## 3. "Para que serve o modelo pré-parto, se a idade gestacional é a do parto?"

- O modelo estima o risco **dado que o parto ocorre agora**, com essa idade gestacional. Consultas e Kotelchuck também dependem da idade gestacional.
- **Uso:** na admissão, para decidir transferência *in utero* para uma maternidade com UTI neonatal e para preparar a equipe.
- **Não** serve para triagem precoce da gestação.
- AUROC de {{BXgbAuroc}}. Benefício líquido positivo na faixa clínica: {{nbBDois}} a {{nbBDez}} por mil.

## 4. Outras prováveis

- **Por que não SMOTE?** Distorce a calibração. O peso de classe foi tratado como hiperparâmetro; a busca escolheu {{pesoEscolhido}} ≈ {{hpASpw}} no XGBoost.
- **Então o modelo é mal calibrado?**
  - O escore **bruto** é, com O/E de {{oeBrutoA}}.
  - Por isso a recalibração isotônica é parte do modelo: ela leva o O/E a {{AXgbOE}}, com inclinação de {{calACalInc}}.
  - A leve superestimação que resta é compatível com a variação da TMN ({{tmnValid}}‰ na validação, {{tmnTeste}}‰ no teste). É uma explicação plausível, não uma prova.
- **XGBoost vale a pena contra a logística?**
  - Em AUROC, não: Δ = {{dXLa}} (IC {{dXLaIC}}).
  - Em AUPRC, sim: Δ = {{dXLaprc}} (IC {{dXLaprcIC}}). O ganho está entre os casos de maior risco.
- **Por que só 15/10 tentativas?**
  - Cada ajuste leva ~20 s com 10⁶ registros, e as 5 melhores tentativas do XGBoost ficaram a 0,005 de AUPRC da melhor.
  - O holdout é temporal, e não validação cruzada, porque imita o uso real.
- **Por que a faixa de 2–10% na curva de decisão?**
  - Ela corresponde a aceitar de 9 a 49 falsos alarmes por óbito identificado, plausível para transferência *in utero*.
  - Tratar todos é negativo acima de 0,8%, então a comparação que vale é com não intervir: {{nbADois}} a {{nbADez}} por mil no modelo A.
- **H3?**
  - Não se confirmou: o O/E na baixa escolaridade é {{subBFundOE}} no modelo B.
  - O achado que se sustenta é a **discriminação menor** na baixa escolaridade no modelo A: {{subAFundAuroc}} contra {{subASupAuroc}}.
- **Por que você mudou de conclusão sobre subgrupos entre versões?**
  - No teste de {{anoTesteAnt}}, o modelo B subestimava mães brancas (O/E {{antBBrancaOE}}); em {{anoTeste}}, isso não se repetiu ({{subBBrancaOE}}).
  - A conclusão é que diferenças de calibração por subgrupo são **instáveis** entre anos, e não viés estrutural.
- **Ética?** Dados públicos sem identificação dispensam CEP (Res. CNS 510/2016). O linkage cria risco de reidentificação, então a base pareada não é publicada.

## 5. XAI

- **Como você sabe que as explicações estão certas?** Verifiquei a aditividade: φ₀ + Σφ reproduz o log-odds com erro de {{xaiAditA}}.
  - A verificação pegou uma armadilha: com *early stopping*, a API do XGBoost usa por padrão todas as árvores, inclusive as 50 posteriores à melhor iteração.
  - Sem `iteration_range`, o SHAP explicaria outro modelo. Há um teste de regressão para isso.
- **As explicações são estáveis?** O Spearman do ranking XGBoost × LightGBM é {{xaiSpearA}} (A) e {{xaiSpearB}} (B); entre metades da amostra, é {{xaiMetadesA}}.
- **Consultas em U não é absurdo?** É confusão por indicação: gestações de alto risco são mais acompanhadas. O SHAP mostra associação, não causa.
- **Por que o modelo errou nesses casos?**
  - O óbito não sinalizado ({{xaiFnProb}}) era um bebê a termo, de {{xaiFnPeso}}, {{xaiFnAnom}}. O indicador binário não diz se a malformação é letal; o tipo (`CODANOMAL`) fica como trabalho futuro.
  - O "sobrevivente" com risco de {{xaiFpProb}} tinha {{xaiFpPeso}} e {{xaiFpSem}} semanas: quase certamente é um óbito não pareado. A explicação local expôs o ruído de rótulo.
- **O modelo usa raça?**
  - No modelo B, raça/cor e escolaridade pesam {{xaiSensB}} da importância. Mãe branca tem SHAP médio de {{xaiBrancaB}}; mãe preta, {{xaiPretaB}}.
  - É o gradiente socioeconômico substituindo o estado do recém-nascido, e precisa ser discutido antes de qualquer uso.
