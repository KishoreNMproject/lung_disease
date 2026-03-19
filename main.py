from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import numpy as np  # type: ignore
from flask import Flask, redirect, render_template, request, url_for  # type: ignore
from tensorflow.keras.applications import DenseNet201  # type: ignore
from tensorflow.keras.applications.densenet import preprocess_input  # type: ignore
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, GlobalAveragePooling2D, Input  # type: ignore
from tensorflow.keras.models import Model, load_model  # type: ignore
from tensorflow.keras.preprocessing import image  # type: ignore
from werkzeug.utils import secure_filename  # type: ignore
import tensorflow as tf  # type: ignore
import cv2  # type: ignore


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
IMAGE_SIZE = (224, 224)
CLASS_NAMES = [
    "atelectasis",
    "bacterial_pneumonia",
    "covid19",
    "emphysema",
    "normal",
    "tuberculosis",
    "viral_pneumonia",
]
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


def format_label(name):
    return name.replace("_", " ").title()


def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def build_densenet201_classifier(num_classes, input_shape=(224, 224, 3)):
    inputs = Input(shape=input_shape, name="input_image")
    base_model = DenseNet201(weights=None, include_top=False, input_tensor=inputs)

    x = base_model.outputs[0]
    x = GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = Dense(1024, activation="swish", name="dense_1024")(x)
    x = BatchNormalization(name="batch_norm_1024")(x)
    x = Dropout(0.3, name="dropout_1024")(x)
    x = Dense(512, activation="swish", name="dense_512")(x)
    x = BatchNormalization(name="batch_norm_512")(x)
    x = Dropout(0.25, name="dropout_512")(x)
    x = Dense(256, activation="swish", name="dense_256")(x)
    outputs = Dense(num_classes, activation="softmax", name="predictions")(x)

    return Model(inputs=inputs, outputs=outputs, name="densenet201_transfer_classifier")


@lru_cache(maxsize=1)
def get_model():
    keras_path = BASE_DIR / "final_model.keras"
    h5_path = BASE_DIR / "final_model.h5"

    for model_path in (keras_path, h5_path):
        if not model_path.exists():
            continue

        try:
            return load_model(model_path, compile=False)
        except Exception:
            continue

    if not h5_path.exists():
        raise FileNotFoundError("Could not find final_model.h5 for the DenseNet201 fallback loader.")

    model = build_densenet201_classifier(
        num_classes=len(CLASS_NAMES),
        input_shape=IMAGE_SIZE + (3,),
    )
    model.load_weights(h5_path)
    return model


def predict_image(image_path):
    model = get_model()
    loaded_image = image.load_img(image_path, target_size=IMAGE_SIZE, color_mode="rgb")
    image_array = image.img_to_array(loaded_image)
    batch = np.expand_dims(image_array, axis=0)
    batch = preprocess_input(batch)

    probabilities = model.predict(batch, verbose=0)[0]
    sorted_indices = np.argsort(probabilities)[::-1]
    top_index = int(sorted_indices[0])

    predictions = [
        {
            "label": CLASS_NAMES[index],
            "display_label": format_label(CLASS_NAMES[index]),
            "probability": float(probabilities[index]),
            "percentage": float(probabilities[index] * 100),
        }
        for index in sorted_indices
    ]

    return {
        "top_label": CLASS_NAMES[top_index],
        "top_display_label": format_label(CLASS_NAMES[top_index]),
        "top_probability": float(probabilities[top_index]),
        "top_percentage": float(probabilities[top_index] * 100),
        "predictions": predictions,
        "top_index": top_index,
        "batch": batch
    }

def generate_gradcam(image_path, model, batch, top_index, save_path):
    # Find last conv layer
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        try:
            sh = layer.output_shape
            if isinstance(sh, list):
                sh = sh[0]
            if len(sh) == 4:
                last_conv_layer_name = layer.name
                break
        except Exception:
            pass
            
    if not last_conv_layer_name:
        return
        
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(batch)
        class_channel = preds[:, top_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    
    # Save the superimposed image
    img = cv2.imread(str(image_path))
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
    
    superimposed_img = heatmap_colored * 0.4 + img
    cv2.imwrite(str(save_path), superimposed_img)



@app.route("/")
def index():
    return render_template(
        "index.html",
        class_labels=[format_label(name) for name in CLASS_NAMES],
    )


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("filename")
    if file is None or not file.filename:
        return render_template(
            "index.html",
            error="Choose a chest X-ray image before submitting.",
            class_labels=[format_label(name) for name in CLASS_NAMES],
        )

    if not allowed_file(file.filename):
        return render_template(
            "index.html",
            error="Unsupported file type. Upload PNG, JPG, JPEG, BMP, or WEBP.",
            class_labels=[format_label(name) for name in CLASS_NAMES],
        )

    filename = secure_filename(file.filename)
    unique_name = f"{uuid4().hex}{Path(filename).suffix.lower()}"
    saved_path = UPLOAD_FOLDER / unique_name
    file.save(saved_path)

    try:
        result = predict_image(saved_path)
        
        # Generate Grad-CAM Heatmap
        heatmap_name = f"cam_{unique_name}"
        heatmap_path = UPLOAD_FOLDER / heatmap_name
        model = get_model()
        generate_gradcam(saved_path, model, result["batch"], result["top_index"], heatmap_path)

    except Exception as exc:
        return render_template(
            "index.html",
            error=f"Prediction failed: {exc}",
            class_labels=[format_label(name) for name in CLASS_NAMES],
        )

    return render_template(
        "index.html",
        image_url=url_for("static", filename=f"uploads/{unique_name}"),
        heatmap_url=url_for("static", filename=f"uploads/{heatmap_name}"),
        predictions=result["predictions"],
        predicted_label=result["top_display_label"],
        predicted_confidence=result["top_percentage"],
        class_labels=[format_label(name) for name in CLASS_NAMES],
    )


@app.route("/upload", methods=["GET"])
def upload_redirect():
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
