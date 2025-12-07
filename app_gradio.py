import numpy as np
import cv2
import gradio as gr
import tensorflow as tf

MODEL_PATH = "relatorio_final/modelo_ocr_v1.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Ordem que o Keras usa (alfabética dos nomes das pastas)
CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', '0', '1', '2', '3', '4', '5']

def preprocess_and_predict(image):
    if image is None:
        return {"Nenhuma imagem": 1.0}

    # Sketchpad manda um dict: pegamos a imagem final ("composite")
    if isinstance(image, dict):
        image = image.get("composite", None)
        if image is None:
            return {"Imagem inválida": 1.0}

    img = np.array(image).astype("uint8")

    # Verifica se a imagem está vazia ou é só fundo branco
    if img.size == 0 or np.mean(img) > 250:
        return {"Nenhuma imagem desenhada": 1.0}

    # Converte para escala de cinza
    if img.ndim == 3 and img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif img.ndim == 3 and img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

    # NÃO inverter: dataset já é fundo branco com letra preta
    # img = 255 - img

    # Redimensiona pro mesmo tamanho do dataset (128x128)
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)

    # Engrossa um pouco o traço pra ficar mais parecido com o pincel das folhas
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)

    # Normaliza
    img = img.astype("float32") / 255.0

    # Inverte para combinar com o treinamento (fundo preto, letra branca)
    img = 1.0 - img

    # Ajusta shape para (1,128,128,1)
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img, verbose=0)[0]

    return {CLASSES[i]: float(preds[i]) for i in range(len(CLASSES))}

input_canvas = gr.Sketchpad(label="Desenhe aqui")
output_label = gr.Label(num_top_classes=3)

demo = gr.Interface(
    fn=preprocess_and_predict,
    inputs=input_canvas,
    outputs=output_label,
    live=True,
    title="OCR da Minha Letra - João",
    description="Desenhe uma letra (A-F) ou número (0-5) e veja as 3 classes mais prováveis."
)

if __name__ == "__main__":
    demo.launch()
