import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DIR = os.path.join(ROOT_DIR, "asl_dataset_gray", "train")
VAL_DIR   = os.path.join(ROOT_DIR, "asl_dataset_gray", "val")
TEST_DIR  = os.path.join(ROOT_DIR, "asl_dataset_gray", "test")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "asl_model.h5")

class ASLDetector:
    def __init__(self, train_dir, val_dir, test_dir, img_size=(64, 64)):
        self.train_path = train_dir
        self.val_path = val_dir
        self.test_path = test_dir
        self.img_size = img_size
        self.model = None
        self.history = None
        self.class_names = None

    def prepare_data(self, batch_size=32):
        datagen = ImageDataGenerator(rescale=1./255)

        self.train_generator = datagen.flow_from_directory(
            self.train_path,
            target_size=self.img_size,
            batch_size=batch_size,
            color_mode='grayscale',
            class_mode='categorical',
            shuffle=True
        )

        self.val_generator = datagen.flow_from_directory(
            self.val_path,
            target_size=self.img_size,
            batch_size=batch_size,
            color_mode='grayscale',
            class_mode='categorical',
            shuffle=False
        )

        self.test_generator = datagen.flow_from_directory(
            self.test_path,
            target_size=self.img_size,
            batch_size=batch_size,
            color_mode='grayscale',
            class_mode='categorical',
            shuffle=False
        )

        self.class_names = list(self.train_generator.class_indices.keys())
        print(f"Classes found: {self.class_names}")

    def build_model(self):
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 1)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(len(self.class_names), activation='softmax')
        ])

        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        self.model = model
        print(model.summary())

    def train_model(self, epochs=30, patience=5):
        if self.model is None:
            raise ValueError("Model not built.")

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-5)
        ]

        self.history = self.model.fit(
            self.train_generator,
            validation_data=self.val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

    def evaluate_model(self):
        if self.model is None:
            raise ValueError("Model not trained.")

        loss, acc = self.model.evaluate(self.test_generator, verbose=0)
        print(f"Test Accuracy: {acc:.4f}")
        print(f"Test Loss: {loss:.4f}")

        y_true = self.test_generator.classes
        y_pred = np.argmax(self.model.predict(self.test_generator, verbose=0), axis=1)

        print("Classification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=self.class_names, yticklabels=self.class_names, cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        plt.close()

    def plot_history(self):
        if not self.history:
            raise ValueError("No training history to plot.")

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Train')
        plt.plot(self.history.history['val_accuracy'], label='Validation')
        plt.title("Accuracy")
        plt.legend()
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Train')
        plt.plot(self.history.history['val_loss'], label='Validation')
        plt.title("Loss")
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.savefig("training_plot.png")
        plt.close()

    def save_model(self, filepath="models/asl_model.h5"):
        self.model.save(filepath)
        print(f"------->Model saved to: {filepath}")

def main():
    detector = ASLDetector(TRAIN_DIR, VAL_DIR, TEST_DIR)
    detector.prepare_data()
    detector.build_model()
    detector.train_model()
    detector.plot_history()
    detector.save_model(filepath=MODEL_PATH)
    detector.evaluate_model()
    print("<------ALL DONE------>")


if __name__ == "__main__":
    main()

