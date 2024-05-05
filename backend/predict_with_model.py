from tensorflow.keras.models import load_model
import numpy as np
import cv2

MODEL_ID = 3
IMG_HEIGHT = 300
IMG_WIDTH = 300


def get_model(id: int):
    model = load_model(f"models/face_deepfake_detector_{id}.keras", compile=False)
    return model


def predict_photo(model, photo: cv2.MatLike) -> str:
    photo = cv2.imread()
    image = cv2.resize(photo, (IMG_HEIGHT, IMG_WIDTH))
    print(model.predict(np.array([image])))


def main() -> None:
    model = get_model(MODEL_ID)
    predict_photo(model)


if __name__ == "__main__":
    main()