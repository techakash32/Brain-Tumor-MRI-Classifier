import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from pathlib import Path
from src.cnnClassifier.entity.config_entity import TrainingConfig


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        """Load the updated VGG16 model and recompile with Adam."""
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
        self.model.compile(
            optimizer=Adam(learning_rate=self.config.params_learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def train_valid_generator(self):
        """
        Build train and validation generators from the brain_tumor folder.
        Flow_from_directory will auto-detect the 4 subfolders as classes
        in alphabetical order:
          glioma      → index 0
          meningioma  → index 1
          notumor     → index 2
          pituitary   → index 3
        """
        datagenerator_kwargs = dict(
            rescale=1. / 255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],  # (224, 224)
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self, callback_list: list, checkpoint_filepath: Path = None):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            callbacks=callback_list
        )

        # With EarlyStopping(restore_best_weights=True) above, self.model already
        # holds the best-val-accuracy epoch's weights. As a safety net, explicitly
        # reload the checkpoint ModelCheckpoint wrote to disk, so the saved model
        # is guaranteed to be the best epoch even if callback order/behavior changes.
        if checkpoint_filepath is not None:
            best_model = tf.keras.models.load_model(checkpoint_filepath)
        else:
            best_model = self.model

        self.save_model(
            path=self.config.trained_model_path,
            model=best_model
        )