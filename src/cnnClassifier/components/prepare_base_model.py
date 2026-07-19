import os
import tensorflow as tf
from pathlib import Path
from src.cnnClassifier.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        """Load VGG16 backbone without the top classification layer."""
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top   # False → we add our own head
        )
        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        """Freeze backbone layers and add a Dense softmax head."""
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        flatten_in = tf.keras.layers.Flatten()(model.output)
        prediction = tf.keras.layers.Dense(
            units=classes,          # ✅ 4 for brain tumor (glioma / meningioma / notumor / pituitary)
            activation="softmax"
        )(flatten_in)

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        # ✅ Adam optimizer — better convergence than SGD for fine-tuning
        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model

    def update_base_model(self, freeze_all: bool = True, freeze_till: int = None):
        """
        Attach 4-class head and save updated model.

        Default behavior (freeze_all=True) is unchanged from before — VGG16
        stays fully frozen, only the Dense(4) head trains.

        To fine-tune later: call update_base_model(freeze_all=False, freeze_till=4)
        to unfreeze VGG16's last 4 layers. If you do this, also lower
        LEARNING_RATE in params.yaml (e.g. to 1e-5) for the fine-tuning run —
        full LR on unfrozen pretrained weights will wreck them.
        """
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,  # ✅ reads from params.yaml → 4
            freeze_all=freeze_all,
            freeze_till=freeze_till,
            learning_rate=self.config.params_learning_rate
        )
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)