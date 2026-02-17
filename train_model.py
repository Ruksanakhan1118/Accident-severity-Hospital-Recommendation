import os
import cv2
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Image size
IMG_SIZE = 64

# Dataset path
DATASET_PATH = "Dataset/train"


def load_data():
    data = []
    labels = []

    print("Loading images...")

    for category in os.listdir(DATASET_PATH):
        category_path = os.path.join(DATASET_PATH, category)

        if not os.path.isdir(category_path):
            continue

        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)

            try:
                img = cv2.imread(img_path)

                if img is None:
                    continue

                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = img.flatten()

                data.append(img)
                labels.append(category)

            except Exception as e:
                print("Error loading image:", e)

    return np.array(data), np.array(labels)


def train_model():
    X, y = load_data()

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training model...")
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    print("Evaluating model...")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Save model
    joblib.dump(model, "accident_model.pkl")
    print("Model saved as accident_model.pkl")


if __name__ == "__main__":
    train_model()
