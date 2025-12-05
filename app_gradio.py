import numpy as np
import cv2
import gradio as gr
import tensorflow as tf
import os

# Caminho do modelo treinado
MODEL_PATH = os.path.join("relatorio_final", "modelo_ocr_v1.h5")

# Lista de classes na mesma ordem usada no treinamento
CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', '0', '1', '2', '3', '4', '5']

# Tenta carregar o modelo (mostra erro amigável se não encontrar)
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Modelo carregado: {MODEL_PATH}")
except Exception as e:
    model = None
    print(f"Não foi possível carregar o modelo em '{MODEL_PATH}': {e}")


def preprocess_and_predict(image):
    """
    Recebe imagem do Gradio (H, W, 3) ou (H, W, 4).
    Retorna um dicionário classe->probabilidade ou mensagem de erro.
    """
    if image is None:
        return {"Erro": 1.0}

    if model is None:
        return {"Erro": 1.0}

    img = image.astype("uint8")

    # Converte para escala de cinza
    if img.ndim == 3 and img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    elif img.ndim == 3 and img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif img.ndim == 2:
        img = img
    else:
        # formatos estranhos
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Inverte as cores (esperamos traço claro em fundo escuro)
    img = 255 - img

    # Redimensiona para 128x128
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)

    # Engrossa o traço (dilate)
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)

    # Normaliza para [0,1]
    img = img.astype("float32") / 255.0

    # Ajusta shape para (1,128,128,1)
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    # Predição
    preds = model.predict(img, verbose=0)[0]

    # Constrói dicionário classe->probabilidade
    result = {CLASSES[i]: float(preds[i]) for i in range(len(CLASSES))}
    return result


# Interface Gradio
canvas = gr.Image(
    shape=(256, 256),
    image_mode="RGB",
    source="canvas",
    invert_colors=False,
    tool="freedraw"
)

iface = gr.Interface(
    fn=preprocess_and_predict,
    inputs=canvas,
    outputs=gr.Label(num_top_classes=3),
    title="OCR da Minha Letra",
    description="Desenhe uma letra (A-F) ou número (0-5) e veja as 3 classes mais prováveis.",
)

if __name__ == "__main__":
    iface.launch()
