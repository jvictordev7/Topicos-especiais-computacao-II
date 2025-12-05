# Análise Crítica - Matriz de Confusão

Resumo das métricas e inspeção das principais confusões entre classes.

## Resumo de Métricas (extraído de resumo_metricas.txt)

```
Acurcia Final: 98.44%
Perda Final: 0.0506

```

## Matriz de Confusão (absoluto)

```
  80    0    0    0    0    0    0    0    0    0    0    0
   0   80    0    0    0    0    0    0    0    0    0    0
   0    0   80    0    0    0    0    0    0    0    0    0
   0    0    0   72    0    0    8    0    0    0    0    0
   0    0    0    0   75    4    1    0    0    0    0    0
   0    0    0    0    0   80    0    0    0    0    0    0
   0    0    0    0    0    0   80    0    0    0    0    0
   0    0    0    0    0    0    0   80    0    0    0    0
   0    0    0    0    0    0    0    0   80    0    0    0
   0    0    0    0    0    0    0    0    0   80    0    0
   0    0    0    0    0    0    0    0    0    0   80    0
   0    0    0    0    0    0    0    0    0    0    0   80

```

## Maiores confusões (classe_real -> classe_prevista : contagem)

- **D** -> **0** : 8 imagens
- **E** -> **F** : 4 imagens
- **E** -> **0** : 1 imagens

## Interpretação e hipóteses

- Verifiquem se as confusões listadas acima correspondem a padrões visuais (ex.: traços abertos, bolinhas que confundem D e 0, etc.).
- Possíveis causas: falta de amostras, traço muito fino, recortes que cortam partes do caractere, ruído de digitalização.

## Recomendações práticas para melhorar o desempenho

- Recolher mais amostras especificamente para os pares mais confundidos (p.ex. se D->0 ocorrer muito, coletar mais D e 0 em variedades).
- Aplicar data augmentation focada: variar espessura (dilate/erode), rotações pequenas, variações de brilho/contraste.
- Experimentar aumentar a capacidade do modelo: adicionar filtros, ou treinar mais epochs com EarlyStopping.
- Tentar técnicas de pre-processamento específicas por classe (ex.: normalização de proporção, remoção de loops).
- Se persistir confusão entre letras e números, considerar treinar um ensemble ou um classificador secundário entre os pares mais confundidos.

## Próximos passos sugeridos para entrega

- Recriar o ambiente virtual com Python 3.10 e instalar `requirements.txt`.
- Rodar `python processo_coleta.py` (opção 3) e checar manualmente `dataset_limpo/`.
- Rodar `python principal.py` para treinar/avaliar com o venv limpo (se quiser, aumentar epochs para 30 e ajustar callbacks).
- Gerar o arquivo ZIP final contendo os itens solicitados pelo professor (eu posso gerar automaticamente).
