from define_model import get_next_id, define_and_train_model
import numpy as np
import pandas as pd
from os import listdir, getenv
from dotenv import load_dotenv
from tensorflow.keras.preprocessing.image import load_img, img_to_array, smart_resize
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

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
DATASET_DIR = getenv("DATASET_DIR")


def main() -> None:
    # Make sure the images are in dataset2/separated/train and dataset2/separated/test

    # Define and train model
    id = get_next_id()
    define_and_train_model(generator=True, save=True, save_path="models/face_deepfake_detector_V2_{id}.keras")

    # Log Model Information
    with open("models/model_parameters.txt", "a") as file:
        file.write(f"MODEL_ID: {id} {{\n\tIMG_HEIGHT: {IMG_HEIGHT}\n\tIMG_WIDTH: {IMG_WIDTH}\n\tEPOCHS: {EPOCHS}\n\tBATCH_SIZE: {BATCH_SIZE}\n\tVALIDATION_SPLIT: {VALIDATION_SPLIT}\n\tRANDOM_STATE: {RANDOM_STATE}\n\tCONV1: {CONV1}\n\tCONV2: {CONV2}\n\tCONV3: {CONV3}\n\tDENSE1: {DENSE1}\n\tDENSE2: {DENSE2}\n}}\n\n")



if __name__ == "__main__":
    main()