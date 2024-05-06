from tensorflow.keras.models import load_model
import numpy as np
import cv2

MODEL_ID = 3
IMG_HEIGHT = 300
IMG_WIDTH = 300


def get_model(id: int):
    model = load_model(f"models/face_deepfake_detector_{id}.keras", compile=False)
    return model


def predict_photo(model, photo: np.ndarray) -> dict[str, str]:
    image = cv2.resize(photo, (IMG_HEIGHT, IMG_WIDTH))
    prediction = model.predict(np.array([image]))[0]
    response = {}
    if abs(prediction[0] - prediction[1]) < 0.2:
        response = {"verdict": "unsure", "certainty": ""}
    elif prediction[0] > prediction[1]:
        response = {"verdict": "fake", "certainty": str(round(prediction[0], 2)*100)}
    else:
        response = {"verdict": "real", "certainty": str(round(prediction[1], 2)*100)}
    return response


def prediction(photo: np.ndarray) -> str:
    model = get_model(MODEL_ID)
    return predict_photo(model, photo)


# def main() -> None:
#     model = get_model(MODEL_ID)
#     predict_photo(model, photo)


# if __name__ == "__main__":
#     main()