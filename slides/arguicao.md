# Roteiro de arguição — perguntas prováveis

> Gerado por `relatorio/gerar_readme.py` a partir de `slides/arguicao.tmpl.md`. Os números são os mesmos do relatório (teste de 2024).

## 1. "Os erros do linkage não explicam o resultado?"

- O modelo foi reajustado com rótulos alternativos **no treino**, e a avaliação foi sempre no **mesmo** teste de 2024:
  - **Estrito:** sai quem foi pareado nas passadas 3–6.
  - **Pior caso:** essas pessoas ficam, mas marcadas como sobreviventes.
- ΔAUROC = −0,0009 (−0,0020 a 0,0001) no estrito e −0,0013 (−0,0029 a 0,0005) no pior caso, por bootstrap pareado.
- O modelo é **robusto a ruído de rótulo no treino**. Não diga "insensível aos erros do linkage": o teste também tem rótulos imperfeitos.
- **E o teste?** Sem os pares tolerantes do teste, a AUROC do modelo principal é 0,965; com esses pares zerados, é 0,964 (reportado: 0,964). Eles não inflam o resultado.
- Os ~10% de óbitos sem nenhum par não podem ser avaliados, e isso é um limite admitido.
- A **calibração absoluta não é robusta**: o O/E sobe para 1,07–1,16. Com menos positivos no treino, o modelo aprende uma prevalência menor.
- **Falsos pares:** estimativa de 6% em média, supondo 0% na passada 1 (chave com 8 campos). Isso deixa ≈85% dos óbitos com par verdadeiro.

## 2. "E os ~10% de óbitos não pareados, rotulados como sobreviventes?"

- Pela própria DO, os não pareados têm campos das chaves ausentes:
  - peso: 26,8% contra 0,8%;
  - semanas: 31,5% contra 4,8%;
  - idade da mãe: 32,2% contra 2,1%.
- O perfil é **diferente**: mais do interior, mais óbitos no 1º dia e menos em hospital (óbito precoce fora do hospital ou em trânsito).
  - **Não diga** que são "menos graves": o peso mediano maior deles só é calculado entre quem tem peso, então é uma comparação com seleção.
- O pareamento é de 94,3% na capital e 88,2% no interior. Por raça/cor do recém-nascido, é quase uniforme (89,4%–90,6%).
- A ausência na DN **não** virou sinal de sobrevivência: na base pareada, a TMN é **maior** quando o campo falta (consultas: 19,7‰ contra 7,5‰).
- **O linkage explica diferenças de calibração entre regiões? Não.**
  - Em 2024, a completude subiu 0,3 ponto na capital e 1,6 no interior.
  - Só pelo linkage, o O/E se deslocaria 1,003 e 1,018: menos de 2% de diferença.
- **E a raça da mãe?** A DO só informa a raça do recém-nascido, então a evidência sobre o linkage por raça materna é indireta.

## 3. "Para que serve o modelo pré-parto, se a idade gestacional é a do parto?"

- O modelo estima o risco **dado que o parto ocorre agora**, com essa idade gestacional. Consultas e Kotelchuck também dependem da idade gestacional.
- **Uso:** na admissão, para decidir transferência *in utero* para uma maternidade com UTI neonatal e para preparar a equipe.
- **Não** serve para triagem precoce da gestação.
- AUROC de 0,891. Benefício líquido positivo na faixa clínica: 4,7 a 3,3 por mil.

## 4. Outras prováveis

- **Por que não SMOTE?** Distorce a calibração. O peso de classe foi tratado como hiperparâmetro; a busca escolheu √(n₀/n₁) ≈ 11,35 no XGBoost.
- **Então o modelo é mal calibrado?**
  - O escore **bruto** é, com O/E de 0,23.
  - Por isso a recalibração isotônica é parte do modelo: ela leva o O/E a 0,95, com inclinação de 1,009 (0,981 a 1,037).
  - A leve superestimação que resta é compatível com a variação da TMN (8,20‰ na validação, 7,98‰ no teste). É uma explicação plausível, não uma prova.
- **XGBoost vale a pena contra a logística?**
  - Em AUROC, não: Δ = 0,001 (IC −0,001–0,003).
  - Em AUPRC, sim: Δ = 0,052 (IC 0,040–0,066). O ganho está entre os casos de maior risco.
- **Por que só 15/10 tentativas?**
  - Cada ajuste leva ~20 s com 10⁶ registros, e as 5 melhores tentativas do XGBoost ficaram a 0,005 de AUPRC da melhor.
  - O holdout é temporal, e não validação cruzada, porque imita o uso real.
- **Por que a faixa de 2–10% na curva de decisão?**
  - Ela corresponde a aceitar de 9 a 49 falsos alarmes por óbito identificado, plausível para transferência *in utero*.
  - Tratar todos é negativo acima de 0,8%, então a comparação que vale é com não intervir: 6,2 a 4,9 por mil no modelo A.
- **H3?**
  - Não se confirmou: o O/E na baixa escolaridade é 0,99 (0,90–1,09) no modelo B.
  - O achado que se sustenta é a **discriminação menor** na baixa escolaridade no modelo A: 0,950 (0,932–0,965) contra 0,980 (0,970–0,989).
- **Por que você mudou de conclusão sobre subgrupos entre versões?**
  - No teste de 2023, o modelo B subestimava mães brancas (O/E 1,15); em 2024, isso não se repetiu (1,02 (0,94–1,11)).
  - A conclusão é que diferenças de calibração por subgrupo são **instáveis** entre anos, e não viés estrutural.
- **Ética?** Dados públicos sem identificação dispensam CEP (Res. CNS 510/2016). O linkage cria risco de reidentificação, então a base pareada não é publicada.

## 5. XAI

- **Como você sabe que as explicações estão certas?** Verifiquei a aditividade: φ₀ + Σφ reproduz o log-odds com erro de 7×10⁻6.
  - A verificação pegou uma armadilha: com *early stopping*, a API do XGBoost usa por padrão todas as árvores, inclusive as 50 posteriores à melhor iteração.
  - Sem `iteration_range`, o SHAP explicaria outro modelo. Há um teste de regressão para isso.
- **As explicações são estáveis?** O Spearman do ranking XGBoost × LightGBM é 0,90 (A) e 0,98 (B); entre metades da amostra, é 0,999.
- **Consultas em U não é absurdo?** É confusão por indicação: gestações de alto risco são mais acompanhadas. O SHAP mostra associação, não causa.
- **Por que o modelo errou nesses casos?**
  - O óbito não sinalizado (4,4%) era um bebê a termo, de 3.450 g, com anomalia. O indicador binário não diz se a malformação é letal; o tipo (`CODANOMAL`) fica como trabalho futuro.
  - O "sobrevivente" com risco de 100,0% tinha 385 g e 21 semanas: quase certamente é um óbito não pareado. A explicação local expôs o ruído de rótulo.
- **O modelo usa raça?**
  - No modelo B, raça/cor e escolaridade pesam 7,9% da importância. Mãe branca tem SHAP médio de −0,11; mãe preta, 0,06.
  - É o gradiente socioeconômico substituindo o estado do recém-nascido, e precisa ser discutido antes de qualquer uso.
