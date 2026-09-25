# Roteiro de arguição — perguntas prováveis

Os números vêm de `data/resultados.pkl`; os mesmos estão no relatório e nos slides.

## 1. "Os erros do linkage não explicam o resultado?"

- O modelo foi reajustado com rótulos alternativos **no treino**, e a avaliação foi sempre no **mesmo** teste de 2023:
  - **Estrito:** sai quem foi pareado nas passadas 3–6.
  - **Pior caso:** essas pessoas ficam, mas marcadas como sobreviventes.
- ΔAUROC = −0,0001 (−0,0008 a 0,0006) no estrito e −0,0005 (−0,0014 a 0,0004) no pior caso, por bootstrap pareado.
- A **discriminação** é robusta aos erros do linkage.
- A **calibração absoluta não é**: o O/E sobe para 1,15–1,27. Com menos positivos no treino, o modelo aprende uma prevalência menor.
- Falsos pares: estimativa de 6% em média, supondo 0% na passada 1 (chave com 8 campos). Isso deixa ≈85% dos óbitos com par verdadeiro.

## 2. "E os ~10% de óbitos não pareados, rotulados como sobreviventes?"

- Pela própria DO, os não pareados têm campos das chaves ausentes:
  - peso: 26,7% contra 0,8%;
  - semanas: 31,6% contra 4,9%;
  - idade da mãe: 32,5% contra 2,1%.
- Eles também são menos extremos (peso mediano de 1.408 g contra 1.100 g), mais do interior e morrem menos em hospital.
- O pareamento é de 94,3% na capital e 88,0% no interior; por raça/cor, é quase uniforme (89–91%).
- A ausência na DN **não** virou sinal de sobrevivência: na base pareada, a TMN é **maior** quando o campo falta (consultas: 18,9‰ contra 7,5‰).
- A subestimação no interior (modelo B) pode ter relação com o linkage diferencial. Para mães brancas, não, porque o pareamento é uniforme por raça/cor.

## 3. "Para que serve o modelo pré-parto, se a idade gestacional é a do parto?"

- O modelo estima o risco **dado que o parto ocorre agora**, com essa idade gestacional. Consultas e Kotelchuck também dependem da idade gestacional.
- **Uso:** na admissão, para decidir transferência *in utero* para uma maternidade com UTI neonatal e para preparar a equipe.
- **Não** serve para triagem precoce da gestação.
- AUROC de 0,890. Benefício líquido positivo de 1% a 20% de limiar: 4,2 por mil com p_t = 5%.

## 4. Outras prováveis

- **Por que não SMOTE?** Distorce a calibração. O peso de classe foi tratado como hiperparâmetro, e a busca escolheu "nenhum".
- **Por que só 15/10 tentativas?** Cada ajuste leva ~20 s com 10⁶ registros, e a AUPRC estabilizou antes da metade. O artigo usou validação cruzada de 10 folds; aqui o holdout é temporal, que imita o uso real.
- **O O/E > 1 em 2023 é defeito?** Não. A TMN subiu de 7,70‰ para 8,20‰, e o intercepto de B (0,063) é igual a ln(8,20/7,70). É *drift*, e exige recalibração periódica.
- **H3?** Não se confirmou. O O/E na baixa escolaridade é 1,06 (0,97–1,16) no B, com IC que inclui 1.
- **Ética?** Dados públicos sem identificação dispensam CEP (Res. CNS 510/2016). O linkage cria risco de reidentificação, então a base pareada não é publicada.
