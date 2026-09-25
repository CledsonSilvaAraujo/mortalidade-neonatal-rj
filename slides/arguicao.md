# Roteiro de arguição — perguntas prováveis

Os números vêm de `data/resultados.pkl`; os mesmos estão no relatório e nos slides.

## 1. "Os erros do linkage não explicam o resultado?"

- O modelo foi reajustado com rótulos alternativos **no treino**, e a avaliação foi sempre no **mesmo** teste de 2023:
  - **Estrito:** sai quem foi pareado nas passadas 3–6.
  - **Pior caso:** essas pessoas ficam, mas marcadas como sobreviventes.
- ΔAUROC = −0,0001 (−0,0008 a 0,0006) no estrito e −0,0005 (−0,0014 a 0,0004) no pior caso, por bootstrap pareado.
- O modelo é **robusto a ruído de rótulo no treino**. Não diga "insensível aos erros do linkage": o teste também tem rótulos imperfeitos.
- **E o teste?** O modelo principal, sem os pares tolerantes do teste, tem AUROC de 0,959; com esses pares zerados, 0,959 (reportado: 0,956). Eles não inflam o resultado.
- Os ~10% de óbitos sem nenhum par não podem ser avaliados, e isso é um limite admitido.
- A **calibração absoluta não é**: o O/E sobe para 1,15–1,27. Com menos positivos no treino, o modelo aprende uma prevalência menor.
- Falsos pares: estimativa de 6% em média, supondo 0% na passada 1 (chave com 8 campos). Isso deixa ≈85% dos óbitos com par verdadeiro.

## 2. "E os ~10% de óbitos não pareados, rotulados como sobreviventes?"

- Pela própria DO, os não pareados têm campos das chaves ausentes:
  - peso: 26,7% contra 0,8%;
  - semanas: 31,6% contra 4,9%;
  - idade da mãe: 32,5% contra 2,1%.
- O perfil é **diferente**: mais do interior, mais óbitos no 1º dia e menos em hospital (óbito precoce fora do hospital ou em trânsito).
- **Não diga** que são "menos graves". O peso mediano maior deles só é calculado entre quem tem peso, e 27% não têm: é uma comparação com seleção.
- O pareamento é de 94,3% na capital e 88,0% no interior; por raça/cor, é quase uniforme (89–91%).
- A ausência na DN **não** virou sinal de sobrevivência: na base pareada, a TMN é **maior** quando o campo falta (consultas: 18,9‰ contra 7,5‰).
- **O linkage explica o O/E alto no interior? Não.** A completude é estável entre os anos e em 2023 subiu ≈2 pontos nas duas regiões (O/E esperado só pelo linkage: 1,02 nas duas). A diferença de O/E entre capital (0,96) e interior (1,05) no modelo B continua.
- **E para mães brancas?** O pareamento por raça/cor é quase uniforme, mas essa raça é a do recém-nascido, porque a DO não informa a da mãe. A evidência é indireta.

## 3. "Para que serve o modelo pré-parto, se a idade gestacional é a do parto?"

- O modelo estima o risco **dado que o parto ocorre agora**, com essa idade gestacional. Consultas e Kotelchuck também dependem da idade gestacional.
- **Uso:** na admissão, para decidir transferência *in utero* para uma maternidade com UTI neonatal e para preparar a equipe.
- **Não** serve para triagem precoce da gestação.
- AUROC de 0,890. Benefício líquido positivo de 1% a 20% de limiar: 4,2 por mil com p_t = 5%.

## 4. Outras prováveis

- **Por que não SMOTE?** Distorce a calibração. O peso de classe foi tratado como hiperparâmetro, e a busca escolheu "nenhum".
- **Por que só 15/10 tentativas?** Cada ajuste leva ~20 s com 10⁶ registros, e a AUPRC estabilizou antes da metade. O artigo usou validação cruzada de 10 folds; aqui o holdout é temporal, que imita o uso real.
- **O O/E > 1 em 2023 é defeito?** A explicação plausível é o *drift*: a TMN subiu de 7,70‰ para 8,20‰, e parte disso (≈2%) vem da completude maior do linkage. Não é uma prova, porque o intercepto de A não acompanha a mesma magnitude.
- **A isotônica ajudou?** No teste, piorou um pouco (O/E de 0,99 → 1,02 em A), o que é coerente com o drift de 2022 para 2023. Admita isso: é um achado honesto.
- **Por que a faixa de 2–10% na curva de decisão?** Ela corresponde a aceitar de 9 a 49 falsos alarmes por óbito identificado, o que é plausível para transferência *in utero* e priorização de UTI. Tratar todos é negativo acima de 0,8%, então a comparação que vale é com não intervir: o benefício é de 6,2 a 4,9 por mil (A).
- **H3?** Não se confirmou. O O/E na baixa escolaridade é 1,06 (0,97–1,16) no B, com IC que inclui 1.
- **Ética?** Dados públicos sem identificação dispensam CEP (Res. CNS 510/2016). O linkage cria risco de reidentificação, então a base pareada não é publicada.

## 5. XAI

- **Como você sabe que as explicações estão certas?** Verifiquei a aditividade: φ₀ + Σφ reproduz o log-odds com erro < 1e-5.
  - A verificação pegou uma armadilha: com *early stopping*, a API do XGBoost usa por padrão todas as árvores, inclusive as 50 posteriores à melhor iteração.
  - Sem `iteration_range`, o SHAP explicaria outro modelo. Há um teste de regressão para isso.
- **As explicações são estáveis?**
  - O Spearman do ranking XGBoost × LightGBM é 0,96 (A) e 0,89 (B), com 4 das 5 principais em comum.
  - Entre metades da amostra, é 0,999.
- **Consultas em U (mais de 14 consultas aumentam o risco) não é absurdo?** É confusão por indicação: gestações de alto risco são mais acompanhadas. O SHAP mostra associação no modelo, não efeito causal.
- **Por que o modelo errou nesses casos?**
  - O óbito não sinalizado (risco de 3,6%) tinha Apgar 5/7 e 2.232 g. A DN não distingue esse caso de sobreviventes parecidos.
  - O falso positivo (96%) tinha Apgar 1/1 e anomalia, e sobreviveu.
  - Os erros vêm de falta de informação, não de efeitos espúrios.
- **O modelo usa raça?**
  - No modelo B, raça/cor e escolaridade pesam 8% da importância. Mãe branca tem SHAP médio de −0,13 e mãe preta, +0,08.
  - É o gradiente socioeconômico substituindo o estado do recém-nascido, coerente com a subestimação para mães brancas.
  - Isso é uma questão de equidade a discutir antes de qualquer uso.
