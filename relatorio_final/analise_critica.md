# Análise Crítica - Matriz de Confusão

Resumo das métricas e inspeção das principais confusões entre classes.

## Resumo de Métricas (extraído de resumo_metricas.txt)

```
Acurácia Final: 97.40%
Perda Final: 0.0910

```

## Matriz de Confusão (absoluto)

```
  17    0    0    0    0    0    0    0    0    0    0    0
   0   11    0    0    0    0    0    0    0    0    0    0
   0    0   13    0    0    0    0    0    0    0    0    0
   0    0    0   15    0    0    3    0    0    0    0    0
   0    0    0    0   14    0    0    0    0    0    0    0
   0    0    0    0    0   13    0    0    0    0    0    0
   0    0    0    4    0    0   16    0    0    0    0    0
   0    0    0    0    0    0    0   22    0    0    0    0
   0    0    0    0    0    0    0    0   18    0    0    0
   0    0    0    0    0    0    0    0    0   12    0    0
   0    0    0    0    0    0    0    0    0    0   14    0
   0    0    0    0    0    0    0    0    0    0    0   20

```

## Maiores confusões (classe_real -> classe_prevista : contagem)

- **D** -> **0** : 3 imagens
- **0** -> **D** : 4 imagens
- Demais classes praticamente sem confusão (linhas e colunas quase diagonais).

## Interpretação e hipóteses

- As confusões D↔0 indicam que em alguns recortes o “D” ficou mais redondo ou fechado, lembrando um “0”, e em outros o “0” pode ter gancho lateral lembrando “D”.
- Possíveis causas: variação de traço (muito fino ou muito grosso), cortes nas bordas (recorte apertado) e falta de exemplos variados desses dois caracteres.

## Recomendações práticas para melhorar o desempenho

- Coletar mais amostras de D e 0, variando espessura e estilos, e reprocessar com o robô.
- Data augmentation focada nesses pares: dilate/erode, rotações leves, shifts pequenos.
- Revisar recortes de D e 0 para garantir que não haja cortes laterais; ajustar margens se necessário.
- Treinar mais algumas épocas (com EarlyStopping) ou aumentar filtros no último bloco convolucional para maior capacidade.
- Se persistir, usar um classificador auxiliar só para decidir entre D e 0.

## Próximos passos sugeridos para entrega

- Recriar o ambiente virtual com Python 3.10 e instalar `requirements.txt`.
- Rodar `python processo_coleta.py` (opção 3) e checar manualmente `dataset_limpo/`.
- Rodar `python principal.py` para treinar/avaliar com o venv limpo (se quiser, aumentar epochs para 30 e ajustar callbacks).
- Gerar o arquivo ZIP final contendo os itens solicitados pelo professor (eu posso gerar automaticamente).
