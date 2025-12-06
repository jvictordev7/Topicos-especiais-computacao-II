# 🧠 OCR de Escrita Manual (A–F, 0–5)

Reconhecimento de caracteres manuscritos usando uma pipeline completa: coleta e limpeza automatizada de dados, treinamento de uma CNN em TensorFlow/Keras, geração de relatórios e uma interface Gradio para testes rápidos. 🎯

## 📂 Estrutura do projeto
- `processo_coleta.py` → robotiza o recorte e a limpeza das folhas escaneadas, produzindo `dataset_limpo/`.
- `principal.py` → treina a CNN (128×128 em tons de cinza), gera métricas, gráficos e matriz de confusão em `relatorio_final/`.
- `gerar_analise_critica.py` → lê o modelo treinado e o dataset limpo para criar um diagnóstico em Markdown.
- `app_gradio.py` → interface web para desenhar e testar previsões em tempo real.
- Pastas de dados:
  - `folhas_coleta/` (entrada de scans), `dataset_bruto/` (intermediário), `dataset_limpo/` (dataset final por classe), `debug_visual/` (cortes anotados) e `relatorio_final/` (modelo e relatórios).

## 🚀 Pré-requisitos
- Python 3.10+ recomendado.
- Dependências em `requirements.txt` (TensorFlow 2.16+, OpenCV, NumPy < 2, Matplotlib, scikit-learn, Seaborn, Gradio 6.0.2).
- GPU é opcional, mas acelera o treinamento.

## 🛠️ Instalação rápida
```bash
python -m venv .venv
source .venv/bin/activate  # ou .venv\Scripts\activate no Windows
pip install --upgrade pip
pip install -r requirements.txt
```

## 🖼️ Pipeline de dados (coleta → limpeza)
1) Coloque scans das planilhas em `folhas_coleta/` (formatos .jpg/.png).  
2) Execute:
```bash
python processo_coleta.py
```
O script:
- Corrige rotação, reconstrói a grade e recorta cada célula.
- Elimina recortes com ruído/bordas cortadas e centraliza o caractere.
- Salva recortes aprovados em `dataset_limpo/<classe>/` (A–F, 0–5) e cortes anotados em `debug_visual/`.
👉 Faça uma inspeção manual final em `dataset_limpo/` para garantir qualidade.

## 🧑‍💻 Treinamento da CNN
```bash
python principal.py
```
- Modelo: 3 blocos Conv2D+MaxPooling, Dense(128) + Dropout(0.5), saída softmax para 12 classes.
- Augmentation: rotações, shifts, zoom, shear.
- Callbacks: `EarlyStopping` e `ReduceLROnPlateau`.
- Saídas em `relatorio_final/`:
  - `modelo_ocr_v1.h5` (modelo salvo).
  - `resumo_metricas.txt` (acurácia e loss finais).
  - `grafico_evolucao.png` (acurácia/loss), `matriz_confusao.png` (heatmap) e `exemplos_visuais.png` (amostras corretas/erradas).

## 🔎 Análise crítica pós-treino
```bash
python gerar_analise_critica.py
```
- Gera `relatorio_final/analise_critica.md` com matriz de confusão, top confusões e recomendações de melhoria.

## 🖌️ Testes interativos (Gradio)
```bash
python app_gradio.py
```
- Abre um canvas para desenhar letras/números.
- Retorna as 3 classes mais prováveis com probabilidades.
- Necessita do modelo salvo em `relatorio_final/modelo_ocr_v1.h5`.

## 📏 Boas práticas e dicas
- Garanta equilíbrio de classes em `dataset_limpo/`; colete mais amostras para pares confundidos (ex.: D vs 0).
- Experimente mais épocas ou ajuste `batch_size` se houver GPU disponível.
- Se usar outra versão do TensorFlow, alinhe com a versão do Python e reinstale o venv.

## ✅ Checklist rápido
- [ ] Ativar venv e instalar dependências.  
- [ ] Preencher `folhas_coleta/` e rodar `processo_coleta.py`.  
- [ ] Conferir `dataset_limpo/` manualmente.  
- [ ] Rodar `principal.py` e revisar saídas em `relatorio_final/`.  
- [ ] (Opcional) Rodar `gerar_analise_critica.py` e `app_gradio.py`.  
