"""
=========================================================================================
					   TREINADOR DE REDE NEURAL (CNN) - OCR
=========================================================================================
ARQUIVO DE ATIVIDADE: PREENCHA AS LACUNAS (TODO)
-----------------------------------------------------------------------------------------
Este script já possui:
1. O carregamento das imagens da pasta 'dataset_limpo'.
2. O pré-processamento (normalização, redimensionamento).
3. A geração automática de relatórios e gráficos no final.
 
SUA MISSÃO:
Você é o arquiteto do cérebro da IA. Você deve projetar as camadas (criar_modelo_cnn)
e iniciar o treinamento (.fit). Siga os comentários marcados com 'TODO'.
=========================================================================================
"""

# Para o Codex: adicione (ou ajuste) os imports do Keras abaixo caso necessário.
# Objetivo: usar o padrão `from tensorflow.keras...` para construir uma CNN.
# Exemplo esperado (ou próximo):
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --- CONFIGURAÇÕES GLOBAIS ---
PASTA_DATASET = "dataset_limpo"
PASTA_RELATORIO = "relatorio_final"
TAMANHO_IMAGEM = (128, 128)
NUM_CLASSES = 12 # A-F e 0-5
CLASSES_NOMES = ['A', 'B', 'C', 'D', 'E', 'F', '0', '1', '2', '3', '4', '5']

os.makedirs(PASTA_RELATORIO, exist_ok=True)

# ===============================================================================
# 1. PREPARAÇÃO DOS DADOS (JÁ IMPLEMENTADO - NÃO MEXER)
# ===============================================================================
def processar_imagem(caminho_imagem):
	img = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)
	if img is None: return None
	img = cv2.resize(img, TAMANHO_IMAGEM, interpolation=cv2.INTER_AREA)
	img = img / 255.0 # Normalização
	img = 1.0 - img   # Inversão
	return img

def carregar_dados():
	print("\n--- 1. CARREGANDO IMAGENS ---")
	X_data = []
	Y_data = []
	mapa_rotulos = {name: i for i, name in enumerate(CLASSES_NOMES)}

	for rotulo_nome, rotulo_idx in mapa_rotulos.items():
		pasta_caractere = os.path.join(PASTA_DATASET, rotulo_nome)
		if not os.path.isdir(pasta_caractere): continue
        
		arquivos = [f for f in os.listdir(pasta_caractere) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
		print(f"-> Lendo classe '{rotulo_nome}': {len(arquivos)} imagens.")
        
		for nome_arquivo in arquivos:
			caminho = os.path.join(pasta_caractere, nome_arquivo)
			matriz = processar_imagem(caminho)
			if matriz is not None:
				X_data.append(matriz)
				Y_data.append(rotulo_idx)

	X_data = np.array(X_data)
	Y_data = np.array(Y_data)
	X_data = np.expand_dims(X_data, axis=-1) # Adiciona canal de cor (1)
	Y_data_one_hot = to_categorical(Y_data, num_classes=NUM_CLASSES)
    
	return train_test_split(X_data, Y_data_one_hot, test_size=0.2, random_state=42)

# ===============================================================================
# 2. ARQUITETURA DA REDE (SUA MISSÃO: COMPLETAR O CÓDIGO)
# ===============================================================================
# Para o Codex: Preencha a função `criar_modelo_cnn()` seguindo exatamente
# as instruções dos comentários TODO abaixo.
# Arquitetura esperada:
# - Conv2D(32, (3,3), activation='relu', input_shape=(128,128,1))
# - MaxPooling2D((2,2))
# - Conv2D(64, (3,3), activation='relu') + MaxPooling2D((2,2))
# - Conv2D(128, (3,3), activation='relu') + MaxPooling2D((2,2))
# - Flatten()
# - Dense(128, activation='relu')
# - Dropout(0.5)
# - Dense(12, activation='softmax')
# Depois, compile com optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']

def criar_modelo_cnn():
	# Inicializa um modelo sequencial (uma pilha de camadas)
	model = Sequential()

	print("--- 2. CONSTRUINDO O CÉREBRO DA IA ---")

	# Camada de entrada + primeiro bloco convolucional
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(TAMANHO_IMAGEM[0], TAMANHO_IMAGEM[1], 1)))
	model.add(MaxPooling2D((2, 2)))

	# Segundo bloco convolucional
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D((2, 2)))

	# Terceiro bloco convolucional
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D((2, 2)))

	# Flatten para passar às camadas densas
	model.add(Flatten())

	# Camada densa (raciocínio) + dropout
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))

	# Camada de saída
	model.add(Dense(NUM_CLASSES, activation='softmax'))

	# Compilando o modelo
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
	return model

# ===============================================================================
# 3. RELATÓRIOS E GRÁFICOS (JÁ IMPLEMENTADO)
# ===============================================================================
def salvar_graficos_historico(history):
	acc = history.history['accuracy']
	val_acc = history.history['val_accuracy']
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	epochs_range = range(len(acc))

	plt.figure(figsize=(14, 5))
	plt.subplot(1, 2, 1)
	plt.plot(epochs_range, acc, label='Treino (Estudo)')
	plt.plot(epochs_range, val_acc, label='Validação (Prova)')
	plt.legend(loc='lower right')
	plt.title('Acurácia')
	plt.grid(True)

	plt.subplot(1, 2, 2)
	plt.plot(epochs_range, loss, label='Treino')
	plt.plot(epochs_range, val_loss, label='Validação')
	plt.legend(loc='upper right')
	plt.title('Loss (Erro)')
	plt.grid(True)
    
	caminho = os.path.join(PASTA_RELATORIO, "grafico_evolucao.png")
	plt.savefig(caminho)
	print(f"[Relatório] Gráfico salvo em: {caminho}")

def salvar_matriz_confusao(model, X_test, Y_test):
	Y_pred_probs = model.predict(X_test)
	Y_pred = np.argmax(Y_pred_probs, axis=1)
	Y_true = np.argmax(Y_test, axis=1)
	cm = confusion_matrix(Y_true, Y_pred)
    
	plt.figure(figsize=(10, 8))
	sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES_NOMES, yticklabels=CLASSES_NOMES)
	plt.xlabel('IA Previu')
	plt.ylabel('Real')
	plt.title('Matriz de Confusão')
    
	caminho = os.path.join(PASTA_RELATORIO, "matriz_confusao.png")
	plt.savefig(caminho)
	print(f"[Relatório] Matriz salva em: {caminho}")

def salvar_previsoes_visuais(model, X_test, Y_test):
	indices = np.random.choice(len(X_test), 15, replace=False)
	plt.figure(figsize=(15, 8))
	plt.suptitle("Teste Visual (Verde=Acerto, Vermelho=Erro)", fontsize=16)
    
	for i, idx in enumerate(indices):
		img = X_test[idx]
		label_real = CLASSES_NOMES[np.argmax(Y_test[idx])]
		pred_probs = model.predict(np.expand_dims(img, axis=0), verbose=0)
		pred_label = CLASSES_NOMES[np.argmax(pred_probs)]
		conf = np.max(pred_probs) * 100
		cor = 'green' if label_real == pred_label else 'red'
        
		plt.subplot(3, 5, i + 1)
		plt.imshow(img.squeeze(), cmap='gray')
		plt.axis('off')
		plt.title(f"Real: {label_real}\nIA: {pred_label} ({conf:.1f}%)", color=cor, fontsize=10, fontweight='bold')
        
	plt.tight_layout()
	caminho = os.path.join(PASTA_RELATORIO, "exemplos_visuais.png")
	plt.savefig(caminho)
	print(f"[Relatório] Exemplos salvos em: {caminho}")

def salvar_resumo_texto(score_loss, score_acc):
	caminho = os.path.join(PASTA_RELATORIO, "resumo_metricas.txt")
	with open(caminho, "w") as f:
		f.write(f"Acurácia Final: {score_acc*100:.2f}%\nPerda Final: {score_loss:.4f}\n")
	print(f"[Relatório] Resumo salvo em: {caminho}")

# ===============================================================================
# 4. EXECUÇÃO PRINCIPAL
# ===============================================================================
if __name__ == "__main__":
	# 1. Carregar Dados
	X_train, X_test, Y_train, Y_test = carregar_dados()
	print(f"\n[Status] Dataset carregado. Treino: {X_train.shape}, Teste: {X_test.shape}")

	# 2. Criar Modelo
	modelo = criar_modelo_cnn()
    
	# Validação pedagógica (Checagem de erro)
	if not hasattr(modelo, 'optimizer') or modelo.optimizer is None:
		print("\n[ERRO FATAL] O modelo não foi compilado!")
		print("Vá até a função 'criar_modelo_cnn' e complete o TODO 2 (model.compile).")
		exit()

	try:
		modelo.summary()
	except ValueError:
		print("\n[ERRO FATAL] O modelo não tem camadas!")
		print("Vá até a função 'criar_modelo_cnn' e complete o TODO 1 (model.add).")
		exit()

	# 3. Data Augmentation
	print("\n--- 3. CONFIGURANDO DATA AUGMENTATION ---")
	datagen = ImageDataGenerator(
		rotation_range=15, width_shift_range=0.1, height_shift_range=0.1,
		zoom_range=0.15, shear_range=0.1, fill_mode='nearest'
	)

	callbacks = [
		EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True),
		ReduceLROnPlateau(patience=3, factor=0.5, monitor='val_loss')

	]

	# --- TODO 3: INICIAR O TREINAMENTO (DAR PLAY) ---
	print("\n--- 4. INICIANDO TREINAMENTO ---")
	# Aqui conectamos tudo: imagens (datagen), respostas (Y_train) e regras (callbacks).
    
	# Para o Codex: No local indicado pelo TODO para treinar o modelo, chame o método
	# `fit` usando as variáveis de treino e validação já criadas acima no código.
	# Use, por exemplo:
	# epochs=20
	# batch_size=32
	# validation_data=(X_test, Y_test)
	# Guarde o retorno em uma variável chamada `history`.
    
	history = modelo.fit(
		datagen.flow(X_train, Y_train, batch_size=32),
		epochs=40,
		validation_data=(X_test, Y_test),
		callbacks=callbacks
	)
    
	# --- 5. SALVAMENTO E RELATÓRIO ---
	if 'history' in locals():
		print("\n--- 5. GERANDO RELATÓRIO FINAL ---")
		caminho_modelo = os.path.join(PASTA_RELATORIO, 'modelo_ocr_v1.h5')
		modelo.save(caminho_modelo)
		print(f"[Sucesso] Modelo salvo em: {caminho_modelo}")

		loss, acc = modelo.evaluate(X_test, Y_test, verbose=0)
		salvar_resumo_texto(loss, acc)
		salvar_graficos_historico(history)
		salvar_matriz_confusao(modelo, X_test, Y_test)
		salvar_previsoes_visuais(modelo, X_test, Y_test)
        
		print(f"\n[FIM] Tudo pronto! Verifique a pasta '{PASTA_RELATORIO}'")
	else:
		print("\n[ERRO] A variável 'historico' não existe.")
		print("Você esqueceu de implementar o 'modelo.fit' no TODO 3?")


