import os
import cv2
import numpy as np
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

# Create upload folder if not exists
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

IMG_SIZE = 64

# Load trained model
model = joblib.load("accident_model.pkl")


def prepare_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.flatten()   # 🔥 ADD THIS LINE
    return img


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    recommendation = None

    if request.method == "POST":
        if 'file' not in request.files:
            return render_template("index.html")

        file = request.files['file']

        if file.filename == "":
            return render_template("index.html")

        if file:
            # your prediction logic here
            prediction = "Major Accident"
            recommendation = "Nearest Multi-Speciality Hospital"

    return render_template("index.html", prediction=prediction, recommendation=recommendation)


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
