from flask import Flask, request
from flask_cors import CORS
from cv2 import imdecode, IMREAD_COLOR
import numpy as np
from predict_with_model import prediction


UPLOAD_FOLDER = "temp_uploads/"
ALLOWED_EXTENSIONS = set(["jpg", "png", "bmp"])
app = Flask(__name__)
CORS(app)


@app.route("/is_deepfake", methods=["POST"])
def get_prediction():
    file = request.files["file"]
    image = imdecode(np.frombuffer(file.read(), np.uint8), IMREAD_COLOR)
    verdict = prediction(image)
    print(verdict)
    return verdict, 200
    



if __name__ == "__main__":
    app.run()