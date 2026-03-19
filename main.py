from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import numpy as np
from flask import Flask, redirect, render_template, request, url_for
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename


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
    }


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
    except Exception as exc:
        return render_template(
            "index.html",
            error=f"Prediction failed: {exc}",
            class_labels=[format_label(name) for name in CLASS_NAMES],
        )

    return render_template(
        "index.html",
        image_url=url_for("static", filename=f"uploads/{unique_name}"),
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
