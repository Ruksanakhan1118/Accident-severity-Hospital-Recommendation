import os
import cv2
import numpy as np
import joblib
from flask import Flask, render_template, request
import os
model = joblib.load("accident_model.pkl")


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
def predict(image_path):

    img = cv2.imread(image_path)   # ✅ Load image first
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (64, 64))   # resize (important)

    img = img.flatten()
    img = img.reshape(1, -1)

    prediction = model.predict(img)
    body_part = prediction[0]

    # Simple severity logic
    if body_part == "head":
        severity = "Major"
    else:
        severity = "Minor"

    hospital_details = "City Hospital - Emergency Ward Available"

    return body_part, severity, hospital_details


@app.route('/', methods=['GET', 'POST'])
def home():

    if request.method == 'POST':

        file = request.files['image']

        if file:
            filepath = os.path.join("static", file.filename)
            file.save(filepath)

            body_part, severity, hospital_details = predict(filepath)

            return render_template(
                'index.html',
                uploaded_image=filepath,
                body_part=body_part,
                severity=severity,
                hospital_details=hospital_details
            )

    return render_template('index.html')

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
