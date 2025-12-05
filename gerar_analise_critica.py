import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix

# Config
MODEL_PATH = os.path.join('relatorio_final', 'modelo_ocr_v1.h5')
DATASET_DIR = 'dataset_limpo'
OUTPUT_MD = os.path.join('relatorio_final', 'analise_critica.md')
CLASSES = ['A','B','C','D','E','F','0','1','2','3','4','5']


def carregar_dataset(dirpath):
    X = []
    y = []
    for idx, cls in enumerate(CLASSES):
        pasta = os.path.join(dirpath, cls)
        if not os.path.isdir(pasta):
            continue
        arquivos = [f for f in os.listdir(pasta) if f.lower().endswith(('.png','.jpg','.jpeg'))]
        for a in arquivos:
            p = os.path.join(pasta, a)
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (128,128), interpolation=cv2.INTER_AREA)
            img = 1.0 - (img.astype('float32') / 255.0)
            img = np.expand_dims(img, axis=-1)
            X.append(img)
            y.append(idx)
    X = np.array(X)
    y = np.array(y)
    return X, y


def main():
    if not os.path.exists(MODEL_PATH):
        print('Modelo não encontrado em', MODEL_PATH)
        return
    print('Carregando modelo...')
    model = load_model(MODEL_PATH)
    print('Carregando dataset...')
    X, y = carregar_dataset(DATASET_DIR)
    if len(X) == 0:
        print('Nenhuma imagem encontrada em', DATASET_DIR)
        return

    print('Predizendo...')
    preds = model.predict(X, verbose=0)
    y_pred = np.argmax(preds, axis=1)

    cm = confusion_matrix(y, y_pred)

    # Normaliza linha a linha (por classe real)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Identifica maiores confusoes (fora diagonal)
    confs = []
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if i == j:
                continue
            confs.append((i, j, cm[i, j]))
    confs_sorted = sorted(confs, key=lambda x: x[2], reverse=True)

    # Cria markdown
    with open(OUTPUT_MD, 'w', encoding='utf-8') as f:
        f.write('# Análise Crítica - Matriz de Confusão\n\n')
        f.write('Resumo das métricas e inspeção das principais confusões entre classes.\n\n')

        # insert metrics from resumo_metricas.txt if available
        resumo_path = os.path.join('relatorio_final', 'resumo_metricas.txt')
        if os.path.exists(resumo_path):
            with open(resumo_path, 'r', encoding='utf-8', errors='ignore') as r:
                f.write('## Resumo de Métricas (extraído de resumo_metricas.txt)\n\n')
                f.write("```\n")
                f.write(r.read())
                f.write('\n')
                f.write("```\n\n")

        f.write('## Matriz de Confusão (absoluto)\n\n')
        f.write("```\n")
        for row in cm:
            f.write(' '.join([str(int(x)).rjust(4) for x in row]) + '\n')
        f.write('\n')
        f.write("```\n\n")

        f.write('## Maiores confusões (classe_real -> classe_prevista : contagem)\n\n')
        top_k = 10
        for i, j, cnt in confs_sorted[:top_k]:
            if cnt == 0:
                continue
            f.write(f'- **{CLASSES[i]}** -> **{CLASSES[j]}** : {int(cnt)} imagens\n')

        f.write('\n')
        f.write('## Interpretação e hipóteses\n\n')
        f.write('- Verifiquem se as confusões listadas acima correspondem a padrões visuais (ex.: traços abertos, bolinhas que confundem D e 0, etc.).\n')
        f.write('- Possíveis causas: falta de amostras, traço muito fino, recortes que cortam partes do caractere, ruído de digitalização.\n')

        f.write('\n')
        f.write('## Recomendações práticas para melhorar o desempenho\n\n')
        f.write('- Recolher mais amostras especificamente para os pares mais confundidos (p.ex. se D->0 ocorrer muito, coletar mais D e 0 em variedades).\n')
        f.write('- Aplicar data augmentation focada: variar espessura (dilate/erode), rotações pequenas, variações de brilho/contraste.\n')
        f.write('- Experimentar aumentar a capacidade do modelo: adicionar filtros, ou treinar mais epochs com EarlyStopping.\n')
        f.write('- Tentar técnicas de pre-processamento específicas por classe (ex.: normalização de proporção, remoção de loops).\n')
        f.write('- Se persistir confusão entre letras e números, considerar treinar um ensemble ou um classificador secundário entre os pares mais confundidos.\n')

        f.write('\n')
        f.write('## Próximos passos sugeridos para entrega\n\n')
        f.write('- Recriar o ambiente virtual com Python 3.10 e instalar `requirements.txt`.\n')
        f.write('- Rodar `python processo_coleta.py` (opção 3) e checar manualmente `dataset_limpo/`.\n')
        f.write('- Rodar `python principal.py` para treinar/avaliar com o venv limpo (se quiser, aumentar epochs para 30 e ajustar callbacks).\n')
        f.write('- Gerar o arquivo ZIP final contendo os itens solicitados pelo professor (eu posso gerar automaticamente).\n')

    print('Analise gerada em', OUTPUT_MD)


if __name__ == '__main__':
    main()
