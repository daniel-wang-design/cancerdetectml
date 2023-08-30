import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import math
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
layers = tf.keras.layers
from tensorflow import keras
from sklearn.metrics import roc_curve
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

train_examples = 20225
test_examples = 2551
validation_examples = 2555
img_height = img_width = 224
batch_size = 32

print("Building model..")
model = keras.Sequential([
   hub.KerasLayer("https://tfhub.dev/google/imagenet/nasnet_mobile/feature_vector/4",
                  trainable=True),
   layers.Dense(1, activation="sigmoid"),
])

print("Model built successfully")
print("Generating image data for - training")
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    zoom_range=(0.95, 0.95),
    horizontal_flip=True,
    vertical_flip=True,
    data_format="channels_last",
    dtype=np.float32,
)
print("Generating image data for - validation")
validation_datagen = ImageDataGenerator(rescale=1.0 / 255, dtype=np.float32)
print("Generating image data for - test")
test_datagen = ImageDataGenerator(rescale=1.0 / 255, dtype=np.float32)

print("Fetching data - training")
train_gen = train_datagen.flow_from_directory(
    "data/train/",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode="rgb",
    class_mode="binary",
    shuffle=True,
    seed=123,
)
print("Fetching data - validation")
validation_gen = validation_datagen.flow_from_directory(
    "data/validation/",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode="rgb",
    class_mode="binary",
    shuffle=True,
    seed=123,
)
print("Fetching data - test")
test_gen = test_datagen.flow_from_directory(
    "data/test/",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode="rgb",
    class_mode="binary",
    shuffle=True,
    seed=123,
)
print("Setting custom metrics...")
METRICS = [
    keras.metrics.BinaryAccuracy(name="accuracy"),
    keras.metrics.Precision(name="precision"),
    keras.metrics.Recall(name="recall"),
    keras.metrics.AUC(name="auc"),
]
print("Custom metrics set")
print("Compiling model...")
model.compile(
    optimizer=keras.optimizers.legacy.Adam(learning_rate=3e-4),
    loss=[keras.losses.BinaryCrossentropy(from_logits=False)],
    metrics=METRICS,
)
print("Model compiled successfully")
print("Training model...")
model.fit(
    train_gen,
    epochs=1,
    verbose=2,
    steps_per_epoch=train_examples // batch_size,
    validation_data=validation_gen,
    validation_steps=validation_examples // batch_size,
)
print("Model training complete")
def plot_roc(labels, data):
    predictions = model.predict(data)
    fp, tp, _ = roc_curve(labels, predictions)

    plt.plot(100 * fp, 100 * tp)
    plt.xlabel("False positives [%]")
    plt.ylabel("True positives [%]")
    plt.show()


test_labels = np.array([])
num_batches = 0

for _, y in test_gen:
    test_labels = np.append(test_labels, y)
    num_batches += 1
    if num_batches == math.ceil(test_examples / batch_size):
        break
print("Creating graph..,")
plot_roc(test_labels, test_gen)
print("Graph created")
print("Evaluating model...")
model.evaluate(validation_gen, verbose=2)
model.evaluate(test_gen, verbose=2)
print("Machine learning model completed")