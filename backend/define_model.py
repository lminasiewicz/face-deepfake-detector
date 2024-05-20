import numpy as np
import pandas as pd
from dotenv import load_dotenv
from os import listdir, getenv
from tensorflow.keras.preprocessing.image import load_img, img_to_array, smart_resize, ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import History
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from seaborn import heatmap

load_dotenv()
IMG_HEIGHT = int(getenv("IMG_HEIGHT"))
IMG_WIDTH = int(getenv("IMG_WIDTH"))
EPOCHS = int(getenv("EPOCHS"))
BATCH_SIZE = int(getenv("BATCH_SIZE"))
VALIDATION_SPLIT = float(getenv("VALIDATION_SPLIT"))
RANDOM_STATE = int(getenv("RANDOM_STATE"))
CONV1 = int(getenv("CONV1"))
CONV2 = int(getenv("CONV2"))
CONV3 = int(getenv("CONV3"))
DENSE1 = int(getenv("DENSE1"))
DENSE2 = int(getenv("DENSE2"))
DENSE3 = int(getenv("DENSE3"))
DATASET_DIR = getenv("DATASET_DIR")

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
        Conv2D(CONV1, (2, 2), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(CONV2, (2, 2), activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(CONV3, (2, 2), activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(CONV3, (2, 2), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(DENSE1, activation="relu"),
        Dense(DENSE2, activation="relu"),
        Dense(DENSE3, activation="relu"),
        Dense(1, activation="sigmoid")
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


def define_and_train_model(images: np.ndarray = np.array([]), labels: np.ndarray = np.array([]), generator: bool = False, save: bool = False, save_path: str = ""):
    # Define Model
    model = get_model()
    trainer = None
    if generator:
        generator = ImageDataGenerator(rescale=(1.0/255.0), horizontal_flip=True, zoom_range=0.15, rotation_range=20)
        trainer = generator.flow_from_directory('dataset2/separated/train/',
            class_mode='binary', batch_size=BATCH_SIZE, target_size=(IMG_HEIGHT, IMG_WIDTH))
        tester = generator.flow_from_directory('dataset2/separated/test/',
            class_mode='binary', batch_size=BATCH_SIZE, target_size=(IMG_HEIGHT, IMG_WIDTH))

    # Fit Model
    history = History()
    if generator:
        model.fit_generator(trainer, steps_per_epoch=len(trainer), validation_data=tester,
                            validation_steps=len(tester), epochs=EPOCHS, callbacks=[history])
    else:
        x_train, x_test, y_train, y_test = train_test_split(images, labels, random_state=RANDOM_STATE, test_size=VALIDATION_SPLIT)
        model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT, callbacks=[history])
        # Confusion Matrix
        get_confusion_matrix(model, x_test, y_test)
    model.summary()

    # Accuracy & Loss Plots
    get_model_diagnostics(history)

    # Save Model
    if save:
        id = get_next_id()
        if save_path == "":
            model.save(f"models/face_deepfake_detector_{id}.keras")
        else:
            save_path = save_path.replace("{id}", str(id))
            model.save(save_path)
    


def main() -> None:
    # Load Labels
    labels_csv = pd.read_csv(f"./{DATASET_DIR}/data.csv")
    labels = labels_csv["label"]
    total_files = labels_csv.shape[0]

    # Load Images
    images = np.ndarray(shape=(total_files, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint16)
    files = listdir(f"{DATASET_DIR}/all/")
    for i in range(total_files):
        images[i] = smart_resize(img_to_array(load_img(f"{DATASET_DIR}/all/" + files[total_files - i - 1])), (IMG_HEIGHT, IMG_WIDTH))

    # Preprocess labels
    for i in range(len(labels)):
        labels[i] = 0 if labels[i] == "fake" else 1
    labels = to_categorical(labels)

    # Define and train model
    id = get_next_id()
    define_and_train_model(images=images, labels=labels, save=True)

    # Log Model Information
    with open("models/model_parameters.txt", "a") as file:
        file.write(f"MODEL_ID: {id} {{\n\tIMG_HEIGHT: {IMG_HEIGHT}\n\tIMG_WIDTH: {IMG_WIDTH}\n\tEPOCHS: {EPOCHS}\n\tBATCH_SIZE: {BATCH_SIZE}\n\tVALIDATION_SPLIT: {VALIDATION_SPLIT}\n\tRANDOM_STATE: {RANDOM_STATE}\n\tCONV1: {CONV1}\n\tCONV2: {CONV2}\n\tCONV3: {CONV3}\n\tDENSE1: {DENSE1}\n\tDENSE2: {DENSE2}\n}}\n\n")

if __name__ == "__main__":
    main()