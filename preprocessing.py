import os
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

IMG_SIZE    = (224, 224)
BATCH_SIZE  = 16
DATASET_DIR = "dataset_split"
IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASSES     = ["wrinkles", "dark spots", "puffy eyes", "clear skin"]
NUM_CLASSES = len(CLASSES)
CLASS_MAP   = {cls: idx for idx, cls in enumerate(CLASSES)}

def build_generators():
    train_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        horizontal_flip=True,
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.10,
        height_shift_range=0.10,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest"
    ).flow_from_directory(
        os.path.join(DATASET_DIR, "train"),
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", classes=CLASSES,
        shuffle=True, seed=42)

    val_gen = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
        os.path.join(DATASET_DIR, "val"),
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", classes=CLASSES, shuffle=False)

    test_gen = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
        os.path.join(DATASET_DIR, "test"),
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", classes=CLASSES, shuffle=False)

    print(f"[✓] Train:{train_gen.samples} | Val:{val_gen.samples} | Test:{test_gen.samples}")
    return train_gen, val_gen, test_gen

def preprocess_array(arr: np.ndarray) -> np.ndarray:
    rgb     = arr[:, :, ::-1]
    resized = tf.image.resize(rgb, IMG_SIZE).numpy()
    return np.expand_dims(preprocess_input(resized), axis=0)

def visualize_augmentation(source_dir="DATASET/clear skin", output_path="logs/augmentation_samples.png", num_images=6):
    os.makedirs("logs", exist_ok=True)
    source_path = Path(source_dir)
    samples = [p for p in source_path.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file()]
    if not samples:
        print(f"[!] No images found in {source_dir}")
        return
    img = Image.open(samples[0]).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_arr = np.array(img)

    datagen = ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.10,
        height_shift_range=0.10,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest"
    )
    generator = datagen.flow(np.expand_dims(img_arr, 0), batch_size=1)

    cols = min(3, num_images)
    rows = (num_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.array(axes).reshape(-1)
    for i in range(num_images):
        aug = next(generator)[0].astype(np.uint8)
        axes[i].imshow(aug)
        axes[i].axis("off")
    for ax in axes[num_images:]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[✓] Augmentation visualization saved → {output_path}")


def one_hot_label(class_name: str) -> np.ndarray:
    return to_categorical(CLASS_MAP[class_name], num_classes=NUM_CLASSES)


if __name__ == "__main__":
    visualize_augmentation()