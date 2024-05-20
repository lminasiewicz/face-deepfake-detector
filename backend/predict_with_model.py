from tensorflow.keras.models import load_model
import numpy as np
import cv2
from dotenv import load_dotenv
from os import getenv

load_dotenv()

MODEL_ID = 4
MODEL_VERSION = 2
IMG_HEIGHT = int(getenv("IMG_HEIGHT"))
IMG_WIDTH = int(getenv("IMG_WIDTH"))


def get_model(id: int, version: int = 1):
    if version == 1:
        model = load_model(f"models/face_deepfake_detector_{id}.keras", compile=False)
        return model
    elif version == 2:
        model = load_model(f"models/face_deepfake_detector_V2_{id}.keras", compile=False)
        return model


def predict_photo(model, photo: np.ndarray) -> dict[str, str]:
    image = cv2.resize(photo, (IMG_HEIGHT, IMG_WIDTH))
    prediction = model.predict(np.array([image]))[0]
    response = {}
    print(prediction)
    if prediction[0] > 0.4 and prediction[0] < 0.6:
        response = {"verdict": "unsure", "certainty": ""}
    elif prediction[0] > 0.4:
        response = {"verdict": "fake", "certainty": str(round(prediction[0], 2)*100)}
    else:
        response = {"verdict": "real", "certainty": str(round(prediction[1], 2)*100)}
    return response


def prediction(photo: np.ndarray) -> str:
    model = get_model(MODEL_ID, version=MODEL_VERSION)
    return predict_photo(model, photo)


# def main() -> None:
#     model = get_model(MODEL_ID)
#     predict_photo(model, photo)


# if __name__ == "__main__":
#     main()