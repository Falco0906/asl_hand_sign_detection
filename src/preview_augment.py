import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set up augmentation (same as training)
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False,
    fill_mode='nearest'
)

# Update this path if needed
train_path = "../asl_dataset/asl_alphabet_train"

# Create generator
generator = datagen.flow_from_directory(
    train_path,
    target_size=(64, 64),
    batch_size=9,
    class_mode='categorical',
    shuffle=True
)

# Fetch a batch
images, labels = next(generator)

# Plot
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(images[i])
    plt.axis("off")
    plt.title("Augmented")
plt.tight_layout()
plt.show()
