import json
from pathlib import Path
from textwrap import dedent


NOTEBOOK_PATH = Path("DenseNet201_DirectML_Training.ipynb")


def source_lines(text: str):
    text = dedent(text).strip("\n")
    if not text:
        return []
    return [line + "\n" for line in text.splitlines()]


def markdown_cell(text: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines(text),
    }


def code_cell(text: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines(text),
    }


cells = [
    markdown_cell(
        """
        # DenseNet201 Transfer Learning for Multi-Class Medical Image Classification

        This notebook trains a DenseNet201-based classifier with the same overall philosophy as the earlier experiment:

        - ImageNet-pretrained DenseNet backbone
        - Medical-safe augmentation
        - Class weighting for imbalance
        - Two-stage training: feature extraction, then fine-tuning
        - Early stopping, learning rate reduction, and checkpointing
        - Full evaluation with confusion matrix, classification report, recall-focused metrics, and ROC-AUC

        The default paths below are already configured for this repository:

        - `data/lung_disease/train`
        - `data/lung_disease/val`
        - `data/lung_disease/test`

        Only the dataset paths need to change for a different project.
        """
    ),
    markdown_cell(
        """
        ## 1. Environment Setup

        This section imports the required libraries, fixes the random seed for reproducibility, and asks TensorFlow to use the best available accelerator with this priority:

        1. Dedicated GPU
        2. AMD ROCm
        3. DirectML-exposed iGPU on Windows
        4. CPU

        The notebook stays compatible with plain TensorFlow as well. If `tensorflow-directml-plugin` is installed on Windows, the code will try to use it automatically when a regular TensorFlow GPU device is not already visible.
        """
    ),
    code_cell(
        """
        import os
        import platform
        import random
        import warnings
        from importlib import metadata as importlib_metadata
        from pathlib import Path

        os.environ.setdefault("MPLBACKEND", "Agg")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        from IPython.display import display
        from sklearn.metrics import (
            accuracy_score,
            auc,
            classification_report,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
            roc_curve,
        )
        from sklearn.preprocessing import label_binarize
        from sklearn.utils.class_weight import compute_class_weight

        import tensorflow as tf
        from tensorflow.keras.applications import DenseNet201
        from tensorflow.keras.applications.densenet import preprocess_input
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
        from tensorflow.keras.layers import (
            BatchNormalization,
            Dense,
            Dropout,
            GlobalAveragePooling2D,
            Input,
        )
        from tensorflow.keras.losses import CategoricalCrossentropy
        from tensorflow.keras.metrics import AUC
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        LOCAL_ARTIFACTS_DIR = Path("artifacts").resolve()
        LOCAL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        LOCAL_KERAS_CACHE = LOCAL_ARTIFACTS_DIR / "keras_cache"
        LOCAL_KERAS_CACHE.mkdir(parents=True, exist_ok=True)

        SEED = 42
        os.environ["PYTHONHASHSEED"] = str(SEED)
        random.seed(SEED)
        np.random.seed(SEED)
        tf.random.set_seed(SEED)

        sns.set_theme(style="whitegrid")
        plt.rcParams["figure.figsize"] = (12, 5)
        pd.set_option("display.max_columns", None)
        warnings.filterwarnings("ignore")

        try:
            from keras.utils import data_utils as keras_data_utils

            _original_get_file = keras_data_utils.get_file

            def workspace_get_file(*args, **kwargs):
                kwargs.setdefault("cache_dir", str(LOCAL_KERAS_CACHE))
                return _original_get_file(*args, **kwargs)

            keras_data_utils.get_file = workspace_get_file
            tf.keras.utils.get_file = workspace_get_file
        except Exception as exc:
            print(f"Could not patch the local Keras cache directory: {exc}")


        def render_figure(fig):
            display(fig)
            plt.close(fig)


        def configure_best_available_accelerator():
            directml_loaded = False
            directml_note = None

            def score_device(device_name, rocm_build):
                name = device_name.lower()

                if any(token in name for token in ["directml", "dml"]):
                    return 260, "GPU via DirectML"

                if any(token in name for token in ["nvidia", "geforce", "rtx", "gtx", "quadro", "tesla", "titan", "arc"]):
                    return 400, "Dedicated GPU"

                if rocm_build and any(token in name for token in ["amd", "radeon", "gfx"]):
                    return 350, "AMD ROCm"

                if any(token in name for token in ["radeon rx", "radeon pro", "firepro"]):
                    return 330, "Dedicated GPU"

                if any(token in name for token in ["intel", "iris", "uhd", "vega", "radeon graphics", "integrated"]):
                    tier = "iGPU via DirectML" if directml_loaded else "Integrated GPU"
                    return 220 if directml_loaded else 120, tier

                if "gpu" in name:
                    return 250 if directml_loaded else 180, "GPU"

                return 0, "CPU"

            if platform.system() == "Windows":
                try:
                    importlib_metadata.version("tensorflow-directml-plugin")
                    directml_loaded = True
                    directml_note = "DirectML plugin package detected."
                except importlib_metadata.PackageNotFoundError:
                    directml_note = "DirectML plugin not installed; falling back to the default TensorFlow device search."
                except Exception as exc:
                    directml_note = f"DirectML detection warning: {exc}"

            rocm_build = bool(tf.sysconfig.get_build_info().get("is_rocm_build", False))
            gpu_devices = tf.config.list_physical_devices("GPU")

            if not gpu_devices:
                accelerator_info = {
                    "backend": "CPU",
                    "device_name": "CPU",
                    "directml_loaded": directml_loaded,
                    "rocm_build": rocm_build,
                }
                print(f"TensorFlow version: {tf.__version__}")
                print("Selected accelerator: CPU")
                if directml_note:
                    print(directml_note)
                return accelerator_info

            ranked_devices = []
            for gpu_device in gpu_devices:
                try:
                    details = tf.config.experimental.get_device_details(gpu_device)
                except Exception:
                    details = {}

                device_name = (
                    details.get("device_name")
                    or details.get("name")
                    or getattr(gpu_device, "name", str(gpu_device))
                )
                lowered_name = str(device_name).lower()
                if any(token in lowered_name for token in ["directml", "dml"]):
                    directml_loaded = True
                    if directml_note is None:
                        directml_note = "DirectML-backed TensorFlow GPU device detected."
                score, tier = score_device(str(device_name), rocm_build)
                ranked_devices.append(
                    {
                        "device": gpu_device,
                        "device_name": str(device_name),
                        "score": score,
                        "tier": tier,
                    }
                )

            ranked_devices.sort(key=lambda item: item["score"], reverse=True)
            selected = ranked_devices[0]

            try:
                tf.config.set_visible_devices(selected["device"], "GPU")
            except RuntimeError:
                pass

            for visible_gpu in tf.config.list_physical_devices("GPU"):
                try:
                    tf.config.experimental.set_memory_growth(visible_gpu, True)
                except Exception:
                    pass

            accelerator_info = {
                "backend": selected["tier"],
                "device_name": selected["device_name"],
                "directml_loaded": directml_loaded,
                "rocm_build": rocm_build,
            }
            print(f"TensorFlow version: {tf.__version__}")
            print(f"Selected accelerator: {selected['tier']} -> {selected['device_name']}")
            if directml_note:
                print(directml_note)
            return accelerator_info


        ACCELERATOR_INFO = configure_best_available_accelerator()
        ACCELERATOR_INFO
        """
    ),
    markdown_cell(
        """
        ## 2. Dataset Configuration

        Update the three path variables below if you want to train on another dataset. The notebook will automatically detect the classes from the training directory and verify that the same class mapping is used by the validation and test splits.

        The local repository uses `val/`, but this notebook also supports `validation/` if you point `val_dir` there instead.
        """
    ),
    code_cell(
        """
        DATASET_ROOT = Path("data/lung_disease")

        train_dir = DATASET_ROOT / "train"
        val_dir = DATASET_ROOT / "val"
        test_dir = DATASET_ROOT / "test"

        IMAGE_SIZE = (224, 224)
        BATCH_SIZE = 32
        PHASE1_EPOCHS = 25
        PHASE2_EPOCHS = 12
        FINE_TUNE_LAYERS = 50


        def resolve_split_path(path_like, alternate_paths=None):
            path = Path(path_like)
            if path.exists():
                return path

            if alternate_paths:
                for alternate in alternate_paths:
                    alternate = Path(alternate)
                    if alternate.exists():
                        print(f"Using {alternate} because {path} was not found.")
                        return alternate

            raise FileNotFoundError(f"Dataset directory not found: {path}")


        train_dir = resolve_split_path(train_dir)
        val_dir = resolve_split_path(val_dir, alternate_paths=[DATASET_ROOT / "validation"])
        test_dir = resolve_split_path(test_dir)

        print("Train directory:", train_dir.resolve())
        print("Validation directory:", val_dir.resolve())
        print("Test directory:", test_dir.resolve())


        def index_split(split_dir, split_name):
            rows = []
            for class_dir in sorted(path for path in Path(split_dir).iterdir() if path.is_dir()):
                for file_path in class_dir.rglob("*"):
                    if file_path.is_file():
                        rows.append(
                            {
                                "split": split_name,
                                "class_name": class_dir.name,
                                "filepath": str(file_path),
                            }
                        )

            frame = pd.DataFrame(rows)
            summary = (
                frame.groupby(["split", "class_name"], as_index=False)
                .size()
                .rename(columns={"size": "image_count"})
            )
            return frame, summary


        train_files, train_summary = index_split(train_dir, "train")
        val_files, val_summary = index_split(val_dir, "validation")
        test_files, test_summary = index_split(test_dir, "test")

        dataset_summary = pd.concat(
            [
                train_summary,
                val_summary,
                test_summary,
            ],
            ignore_index=True,
        )
        display(dataset_summary)

        detected_classes = dataset_summary.loc[dataset_summary["split"] == "train", "class_name"].tolist()
        num_classes = len(detected_classes)
        print(f"Detected {num_classes} classes: {detected_classes}")

        fig, ax = plt.subplots(figsize=(14, 6))
        ax = sns.barplot(data=dataset_summary, x="class_name", y="image_count", hue="split")
        ax.set_title("Image Count per Class and Split")
        ax.set_xlabel("Class")
        ax.set_ylabel("Images")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        render_figure(fig)
        """
    ),
    markdown_cell(
        """
        ## 3. Data Preprocessing and Augmentation

        DenseNet preprocessing is applied to all images. The training generator also adds light augmentation chosen to be reasonable for chest imaging:

        - rotation up to 10 degrees
        - small translations
        - slight zoom and shear
        - horizontal flips

        Validation and test sets only receive preprocessing so they stay as clean evaluation benchmarks.
        """
    ),
    code_cell(
        """
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=10,
            width_shift_range=0.05,
            height_shift_range=0.05,
            zoom_range=0.1,
            shear_range=0.05,
            horizontal_flip=True,
            fill_mode="nearest",
        )

        eval_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

        generator_kwargs = {
            "target_size": IMAGE_SIZE,
            "batch_size": BATCH_SIZE,
            "class_mode": "categorical",
            "color_mode": "rgb",
        }

        train_generator = train_datagen.flow_from_dataframe(
            dataframe=train_files,
            x_col="filepath",
            y_col="class_name",
            classes=detected_classes,
            shuffle=True,
            seed=SEED,
            validate_filenames=False,
            **generator_kwargs,
        )

        val_generator = eval_datagen.flow_from_dataframe(
            dataframe=val_files,
            x_col="filepath",
            y_col="class_name",
            classes=detected_classes,
            shuffle=False,
            seed=SEED,
            validate_filenames=False,
            **generator_kwargs,
        )

        test_generator = eval_datagen.flow_from_dataframe(
            dataframe=test_files,
            x_col="filepath",
            y_col="class_name",
            classes=detected_classes,
            shuffle=False,
            seed=SEED,
            validate_filenames=False,
            **generator_kwargs,
        )

        class_names = list(train_generator.class_indices.keys())
        num_classes = len(class_names)

        print("Class indices:", train_generator.class_indices)
        print("Number of classes:", num_classes)
        print("Training samples:", train_generator.samples)
        print("Validation samples:", val_generator.samples)
        print("Test samples:", test_generator.samples)
        """
    ),
    markdown_cell(
        """
        ## 4. Class Weight Calculation

        Class weights are computed from the training labels so the optimizer pays more attention to under-represented classes when the dataset is imbalanced.
        """
    ),
    code_cell(
        """
        class_weight_values = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(train_generator.classes),
            y=train_generator.classes,
        )

        class_weights = {
            int(class_id): float(weight)
            for class_id, weight in zip(np.unique(train_generator.classes), class_weight_values)
        }

        class_weight_frame = pd.DataFrame(
            {
                "class_name": class_names,
                "class_id": list(range(num_classes)),
                "class_weight": [class_weights[class_id] for class_id in range(num_classes)],
            }
        )
        display(class_weight_frame)
        """
    ),
    markdown_cell(
        """
        ## 5. Model Architecture

        The model uses an ImageNet-pretrained DenseNet201 backbone followed by a compact fully connected head:

        - `GlobalAveragePooling2D`
        - `Dense(1024, activation="swish")`
        - `BatchNormalization`
        - `Dropout(0.3)`
        - `Dense(512, activation="swish")`
        - `BatchNormalization`
        - `Dropout(0.25)`
        - `Dense(256, activation="swish")`
        - `Dense(num_classes, activation="softmax")`

        The model is created with the Functional API so it remains easy to extend later.
        """
    ),
    code_cell(
        """
        def build_densenet201_classifier(num_classes, input_shape=(224, 224, 3)):
            inputs = Input(shape=input_shape, name="input_image")
            base_model = DenseNet201(
                weights="imagenet",
                include_top=False,
                input_tensor=inputs,
            )
            base_model.trainable = False

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

            model = Model(inputs=inputs, outputs=outputs, name="densenet201_transfer_classifier")
            return model, base_model


        model, base_model = build_densenet201_classifier(
            num_classes=num_classes,
            input_shape=IMAGE_SIZE + (3,),
        )

        model.summary(line_length=120)
        """
    ),
    markdown_cell(
        """
        ## 6. Phase 1 Training: Feature Extraction

        The DenseNet backbone stays frozen in this stage. Only the custom classification head is trained.

        - Optimizer: `Adam(1e-4)`
        - Loss: `CategoricalCrossentropy(label_smoothing=0.1)`
        - Metrics: `accuracy` and multi-class ROC-AUC
        - Callbacks: early stopping, LR reduction, best-checkpoint saving
        """
    ),
    code_cell(
        """
        CHECKPOINT_DIR = Path("checkpoints")
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

        phase1_checkpoint_path = CHECKPOINT_DIR / "densenet201_phase1_best.h5"
        phase2_checkpoint_path = CHECKPOINT_DIR / "densenet201_phase2_best.h5"


        def compile_model(model, learning_rate, num_classes):
            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss=CategoricalCrossentropy(label_smoothing=0.1),
                metrics=[
                    "accuracy",
                    AUC(name="auc", multi_label=True, num_labels=num_classes, curve="ROC"),
                ],
            )


        def build_callbacks(checkpoint_path):
            return [
                EarlyStopping(
                    monitor="val_auc",
                    mode="max",
                    patience=4,
                    restore_best_weights=True,
                    verbose=1,
                ),
                ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.3,
                    patience=2,
                    min_lr=1e-7,
                    verbose=1,
                ),
                ModelCheckpoint(
                    filepath=str(checkpoint_path),
                    monitor="val_auc",
                    mode="max",
                    save_best_only=True,
                    verbose=1,
                ),
            ]


        compile_model(model, learning_rate=1e-4, num_classes=num_classes)

        phase1_history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=PHASE1_EPOCHS,
            class_weight=class_weights,
            callbacks=build_callbacks(phase1_checkpoint_path),
            verbose=1,
        )
        """
    ),
    markdown_cell(
        """
        ## 7. Phase 2 Training: Fine-Tuning

        After the classification head stabilizes, the last 50 DenseNet layers are unfrozen and the entire model is recompiled with a smaller learning rate for careful fine-tuning.
        """
    ),
    code_cell(
        """
        def unfreeze_last_layers(base_model, n_layers=50):
            base_model.trainable = True
            n_layers = min(n_layers, len(base_model.layers))

            for layer in base_model.layers[:-n_layers]:
                layer.trainable = False

            for layer in base_model.layers[-n_layers:]:
                layer.trainable = True

            return n_layers


        unfrozen_layers = unfreeze_last_layers(base_model, n_layers=FINE_TUNE_LAYERS)
        print(f"Unfroze the last {unfrozen_layers} layers of DenseNet201 for fine-tuning.")

        compile_model(model, learning_rate=1e-5, num_classes=num_classes)

        phase2_history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=PHASE2_EPOCHS,
            class_weight=class_weights,
            callbacks=build_callbacks(phase2_checkpoint_path),
            verbose=1,
        )

        model = tf.keras.models.load_model(phase2_checkpoint_path)
        print(f"Loaded best fine-tuned checkpoint from {phase2_checkpoint_path.resolve()}")
        """
    ),
    markdown_cell(
        """
        ## 8. Training Visualization

        The plots below merge both training stages and mark the transition from frozen-backbone training to fine-tuning.
        """
    ),
    code_cell(
        """
        def merge_histories(*histories):
            merged = {}
            for history in histories:
                for metric_name, values in history.history.items():
                    merged.setdefault(metric_name, []).extend(values)
            return merged


        full_history = merge_histories(phase1_history, phase2_history)
        epoch_index = np.arange(1, len(full_history["loss"]) + 1)
        fine_tune_start_epoch = len(phase1_history.history["loss"])

        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        axes[0].plot(epoch_index, full_history["accuracy"], label="Train Accuracy", linewidth=2)
        axes[0].plot(epoch_index, full_history["val_accuracy"], label="Validation Accuracy", linewidth=2)
        axes[0].axvline(fine_tune_start_epoch + 0.5, color="black", linestyle="--", label="Start Fine-Tuning")
        axes[0].set_title("Training vs Validation Accuracy")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy")
        axes[0].legend()

        axes[1].plot(epoch_index, full_history["loss"], label="Train Loss", linewidth=2)
        axes[1].plot(epoch_index, full_history["val_loss"], label="Validation Loss", linewidth=2)
        axes[1].axvline(fine_tune_start_epoch + 0.5, color="black", linestyle="--", label="Start Fine-Tuning")
        axes[1].set_title("Training vs Validation Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].legend()

        fig.tight_layout()
        render_figure(fig)

        history_frame = pd.DataFrame(full_history)
        history_frame.index = np.arange(1, len(history_frame) + 1)
        history_frame.index.name = "epoch"
        display(history_frame.tail())
        """
    ),
    markdown_cell(
        """
        ## 9. Model Evaluation

        Evaluation is performed on the validation and test splits. The test analysis includes:

        - confusion matrix
        - classification report
        - weighted precision, recall, and F1-score
        - multi-class ROC-AUC with one-vs-rest averaging
        - per-class ROC curves and AUC values
        """
    ),
    code_cell(
        """
        evaluation_rows = []

        for split_name, generator in [("validation", val_generator), ("test", test_generator)]:
            metrics = model.evaluate(generator, verbose=1, return_dict=True)
            metrics["split"] = split_name
            evaluation_rows.append(metrics)

        evaluation_frame = pd.DataFrame(evaluation_rows).set_index("split")
        display(evaluation_frame.style.format("{:.4f}"))
        """
    ),
    code_cell(
        """
        test_generator.reset()
        predicted_probabilities = model.predict(test_generator, verbose=1)
        predicted_classes = np.argmax(predicted_probabilities, axis=1)
        true_classes = test_generator.classes

        true_one_hot = label_binarize(true_classes, classes=np.arange(num_classes))

        summary_metrics = pd.DataFrame(
            [
                {
                    "accuracy": accuracy_score(true_classes, predicted_classes),
                    "precision_weighted": precision_score(
                        true_classes,
                        predicted_classes,
                        average="weighted",
                        zero_division=0,
                    ),
                    "recall_weighted": recall_score(
                        true_classes,
                        predicted_classes,
                        average="weighted",
                        zero_division=0,
                    ),
                    "f1_weighted": f1_score(
                        true_classes,
                        predicted_classes,
                        average="weighted",
                        zero_division=0,
                    ),
                    "roc_auc_macro_ovr": roc_auc_score(
                        true_one_hot,
                        predicted_probabilities,
                        multi_class="ovr",
                        average="macro",
                    ),
                    "roc_auc_weighted_ovr": roc_auc_score(
                        true_one_hot,
                        predicted_probabilities,
                        multi_class="ovr",
                        average="weighted",
                    ),
                }
            ],
            index=["test"],
        )
        display(summary_metrics.style.format("{:.4f}"))

        classification_report_frame = pd.DataFrame(
            classification_report(
                true_classes,
                predicted_classes,
                target_names=class_names,
                output_dict=True,
                zero_division=0,
            )
        ).transpose()
        display(classification_report_frame.style.format("{:.4f}"))
        """
    ),
    code_cell(
        """
        confusion = confusion_matrix(true_classes, predicted_classes)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            confusion,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )
        ax.set_title("Test Confusion Matrix")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        fig.tight_layout()
        render_figure(fig)
        """
    ),
    code_cell(
        """
        roc_rows = []

        fig, ax = plt.subplots(figsize=(10, 8))

        for class_index, class_name in enumerate(class_names):
            fpr, tpr, _ = roc_curve(true_one_hot[:, class_index], predicted_probabilities[:, class_index])
            class_auc = auc(fpr, tpr)
            roc_rows.append({"class_name": class_name, "roc_auc": class_auc})
            ax.plot(fpr, tpr, linewidth=2, label=f"{class_name} (AUC = {class_auc:.3f})")

        ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Chance")
        ax.set_title("One-vs-Rest ROC Curves on the Test Set")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        fig.tight_layout()
        render_figure(fig)

        per_class_auc_frame = pd.DataFrame(roc_rows).sort_values("roc_auc", ascending=False)
        display(per_class_auc_frame.style.format({"roc_auc": "{:.4f}"}))
        """
    ),
    markdown_cell(
        """
        ## 10. Save Final Model

        The best fine-tuned model is saved in both the native Keras format and the legacy HDF5 format for easier downstream integration.
        """
    ),
    code_cell(
        """
        final_keras_path = Path("final_model.keras")
        final_h5_path = Path("final_model.h5")

        try:
            model.save(final_keras_path)
            print(f"Saved native Keras model to: {final_keras_path.resolve()}")
        except (TypeError, ValueError) as exc:
            print("Native `.keras` saving is not available in this TensorFlow build.")
            print(f"Falling back to a SavedModel export at: {final_keras_path.resolve()}")
            print(f"Original save error: {exc}")
            tf.saved_model.save(model, str(final_keras_path))

        model.save(final_h5_path)
        print(f"Saved HDF5 model to: {final_h5_path.resolve()}")
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
print(f"Wrote {NOTEBOOK_PATH.resolve()}")
