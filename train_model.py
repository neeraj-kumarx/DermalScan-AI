import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.optimizers import Adam
from preprocessing import build_generators, IMG_SIZE, NUM_CLASSES

os.makedirs("models", exist_ok=True)
os.makedirs("logs",   exist_ok=True)
MODEL_PATH = "models/dermalscan_efficientnetb0.h5"

def build_model():
    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    base   = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=inputs)
    base.trainable = False
    x   = layers.GlobalAveragePooling2D()(base.output)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.5)(x)
    x   = layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return Model(inputs, out), base

def train():
    train_gen, val_gen, test_gen = build_generators()
    model, base = build_model()
    model.compile(optimizer=Adam(1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    callbacks = [
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, verbose=1),
        CSVLogger("logs/training_log.csv", append=True)
    ]
    print("\n[Phase 1] Training head...")
    h1 = model.fit(train_gen, epochs=10, validation_data=val_gen, callbacks=callbacks)

    print("\n[Phase 2] Fine-tuning...")
    for layer in base.layers[-100:]:
        layer.trainable = True
    model.compile(optimizer=Adam(1e-5),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    h2 = model.fit(train_gen, epochs=10, validation_data=val_gen, callbacks=callbacks)

    loss, acc = model.evaluate(test_gen)
    print(f"\n✅ Test Accuracy: {acc*100:.2f}%")

    train_acc = h1.history["accuracy"] + h2.history["accuracy"]
    val_acc   = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    train_loss = h1.history["loss"] + h2.history["loss"]
    val_loss   = h1.history["val_loss"] + h2.history["val_loss"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(train_acc, label="Train")
    axes[0].plot(val_acc, label="Val")
    axes[0].set_title("Accuracy"); axes[0].legend()
    axes[1].plot(train_loss, label="Train")
    axes[1].plot(val_loss, label="Val")
    axes[1].set_title("Loss"); axes[1].legend()
    plt.savefig("logs/training_curves.png", dpi=150)
    plt.close()
    print(f"[✓] Model saved → {MODEL_PATH}")

if __name__ == "__main__":
    train()
    