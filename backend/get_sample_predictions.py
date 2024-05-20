from predict_with_model import get_model, predict_photo
from dotenv import load_dotenv
from os import getenv, listdir
import matplotlib.pyplot as plt
from random import shuffle
import cv2

load_dotenv()
MODEL_ID = 4
MODEL_VERSION = 2
IMG_HEIGHT = int(getenv("IMG_HEIGHT"))
IMG_WIDTH = int(getenv("IMG_WIDTH"))

def main() -> None:
    model = get_model(MODEL_ID, version=MODEL_VERSION)

    image_samples = []
    for image_filename in listdir("sample_photos/"):
        image_samples.append((cv2.resize(cv2.imread(f"sample_photos/{image_filename}"), (256, 256)), image_filename[:4]))
    shuffle(image_samples)
    images = [tup[0] for tup in image_samples]
    labels = [tup[1] for tup in image_samples]

    predicted_labels = []
    plt.figure(figsize=(10,10))
    for i in range(25):
        predicted_labels.append(predict_photo(model, images[i])["verdict"])
        plt.subplot(5,7,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.xlabel(f"True: {labels[i]}\nPredicted: {predicted_labels[i]}")
    plt.savefig("sample_images_V2_4.png")


if __name__ == "__main__":
    main()