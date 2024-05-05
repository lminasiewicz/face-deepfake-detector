import numpy as np
import pandas as pd
from os import listdir
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import History
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from seaborn import heatmap

IMG_HEIGHT = 300
IMG_WIDTH = 300
EPOCHS = 8
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.25
RANDOM_STATE = 2137
CONV1 = 64
CONV2 = 32
CONV3 = 32
DENSE1 = 64
DENSE2 = 16

# Determining the ID of next model
def get_next_id() -> int:
    id = 0
    for file in listdir("models/"):
        if file.startswith("face_deepfake_detector_") and file.endswith(".keras"):
            model_id = int(file.replace("face_deepfake_detector_", "").replace(".keras", ""))
            if model_id > id:
                id = model_id
    return id + 1


def get_model() -> Sequential:
    model = Sequential([
        Conv2D(CONV1, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D((3, 3)),
        Conv2D(CONV2, (3, 3), activation="relu"),
        MaxPooling2D((3, 3)),
        Conv2D(CONV3, (3, 3), activation="relu"),
        MaxPooling2D((3, 3)),
        Flatten(),
        Dense(DENSE1, activation="relu"),
        Dense(DENSE2, activation="relu"),
        Dense(2, activation="softmax")
    ])
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def get_confusion_matrix(model: Sequential, x_test: np.ndarray, y_test: np.ndarray) -> None:
    test_labels = np.argmax(y_test, axis=1)
    print(test_labels)
    predicted_labels = np.argmax(model.predict(x_test), axis=1)
    cm = confusion_matrix(test_labels, predicted_labels)
    plt.figure(figsize=(10, 7))
    heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


def get_model_diagnostics(history: History) -> None:
    # Accuracy Plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', color='grey')
    plt.legend()

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', color='grey')
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    # Load Labels
    labels_csv = pd.read_csv("./dataset/data.csv")
    labels = labels_csv["label"]
    total_files = labels_csv.shape[0]

    # Load Images
    images = np.ndarray(shape=(total_files, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint16)
    files = listdir("dataset/all/")
    for i in range(total_files):
        images[i] = img_to_array(load_img("dataset/all/" + files[total_files - i - 1]))

    # Preprocess labels
    for i in range(len(labels)):
        labels[i] = 0 if labels[i] == "fake" else 1
    labels = to_categorical(labels)

    # Split Dataset into Training and Validation Sets
    x_train, x_test, y_train, y_test = train_test_split(images, labels, random_state=RANDOM_STATE, test_size=VALIDATION_SPLIT)

    # Define Model
    model = get_model()

    # Fit Model
    history = History()
    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT, callbacks=[history])
    model.summary()

    # Confusion Matrix
    get_confusion_matrix(model, x_test, y_test)

    # Accuracy & Loss Plots
    get_model_diagnostics(history)

    # Save Model
    id = get_next_id()
    model.save(f"models/face_deepfake_detector_{id}.keras")

    # Log Model Information
    with open("models/model_parameters.txt", "a") as file:
        file.write(f"MODEL_ID: {id} {{\n\tIMG_HEIGHT: {IMG_HEIGHT}\n\tIMG_WIDTH: {IMG_WIDTH}\n\tEPOCHS: {EPOCHS}\n\tBATCH_SIZE: {BATCH_SIZE}\n\tVALIDATION_SPLIT: {VALIDATION_SPLIT}\n\tRANDOM_STATE: {RANDOM_STATE}\n\tCONV1: {CONV1}\n\tCONV2: {CONV2}\n\tCONV3: {CONV3}\n\tDENSE1: {DENSE1}\n\tDENSE2: {DENSE2}\n}}\n\n")

if __name__ == "__main__":
    main()