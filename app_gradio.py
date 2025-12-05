import numpy as np
import cv2
import gradio as gr
import tensorflow as tf

# Caminho do modelo treinado
MODEL_PATH = "relatorio_final/modelo_ocr_v1.h5"

# Carrega o modelo
model = tf.keras.models.load_model(MODEL_PATH)

# Classes na mesma ordem da saída da rede
CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', '0', '1', '2', '3', '4', '5']


def preprocess_and_predict(image):
    """
    A imagem vem do Gradio como array numpy (H, W, C).
    Precisamos transformar para (1, 128, 128, 1).
    """
    if image is None:
        return {"Nenhuma imagem": 1.0}

    img = image.astype("uint8")

    # Converte para escala de cinza
    if img.ndim == 3 and img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif img.ndim == 3 and img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

    # Inverte as cores (fundo preto, traço branco)
    img = 255 - img

    # Redimensiona para 128x128
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)

    # Engrossa o traço
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)

    # Normaliza
    img = img.astype("float32") / 255.0

    # Ajusta shape
    img = np.expand_dims(img, axis=-1)  # (128,128) -> (128,128,1)
    img = np.expand_dims(img, axis=0)   # -> (1,128,128,1)

    # Predição
    preds = model.predict(img, verbose=0)[0]

    # Monta dicionário classe -> prob
    return {CLASSES[i]: float(preds[i]) for i in range(len(CLASSES))}



# Componente de entrada: canvas de desenho
try:
    # nova assinatura: use height/width em vez de shape
    input_canvas = gr.Image(
        image_mode="RGB",
        source="canvas",   # <- diz que é pra desenhar
        tool="freedraw",   # <- ferramenta de desenho livre
        type="numpy",      # <- retorna numpy array para a função
        height=256,
        width=256,
    )
except TypeError:
    try:
        # fallback para versões antigas de Gradio
        input_canvas = gr.inputs.Image(shape=(256, 256), image_mode="RGB", source="canvas", tool="freedraw")
    except Exception:
        # último recurso: componente Image simples (pode não suportar canvas)
        input_canvas = gr.Image()

output_label = gr.Label(num_top_classes=3)

demo = gr.Interface(
    fn=preprocess_and_predict,
    inputs=input_canvas,
    outputs=output_label,
    title="OCR da Minha Letra - João",
    description="Desenhe uma letra (A-F) ou número (0-5) e veja as 3 classes mais prováveis."
)

if __name__ == "__main__":
    demo.launch()
