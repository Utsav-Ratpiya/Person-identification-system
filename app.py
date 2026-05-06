import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

# Load model
model = tf.keras.models.load_model("best_model.keras")

# Class labels
class_names = [
    "Janavi", "Kaushik", "Mehul", "Nabh", "Nishi",
    "Parth", "Raviraj", "Rudren", "Saiyam",
    "Utsav", "Vrajesh", "Yug"
]

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.6

# Preprocess function
def preprocess(image):
    image = image.resize((160, 160))  # match your model input
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Prediction function
def predict(image):
    if image is None:
        return {"No Image Uploaded": 1.0}

    image = preprocess(image)

    # Faster inference
    preds = model(image, training=False)[0]

    max_prob = float(np.max(preds))
    max_index = int(np.argmax(preds))

    # Avoid wrong predictions
    if max_prob < CONFIDENCE_THRESHOLD:
        return {"Unknown Person": 1.0}

    return {
        class_names[i]: float(preds[i])
        for i in range(len(class_names))
    }

# Gradio Interface (UPLOAD ONLY)
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="📤 Upload Image"),
    outputs=gr.Label(num_top_classes=3, label="Prediction"),
    title="Person Identification System",
    description="Upload an image to identify the person"
)

if __name__ == "__main__":
    interface.launch()