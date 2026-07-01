
import os

print("Number of cat images:", len(os.listdir("cats_set")))
print("Number of dog images:", len(os.listdir("dogs_set")))

import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt


IMG_SIZE = 32

X = []
y = []


for img_name in os.listdir("cats_set"):
    img_path = os.path.join("cats_set", img_name)

    img = cv2.imread(img_path)

    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        X.append(img.flatten())
        y.append(0)      # Cat = 0

# Load dog images
for img_name in os.listdir("dogs_set"):
    img_path = os.path.join("dogs_set", img_name)

    img = cv2.imread(img_path)

    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        X.append(img.flatten())
        y.append(1)      # Dog = 1


X = np.array(X, dtype=np.float32) / 255.0
y = np.array(y)

X = X / 255.0

print("Dataset Loaded Successfully!")
print("Features shape:", X.shape)
print("Labels shape:", y.shape)



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))



svm = SVC(
    kernel='rbf',
    C=10,
    gamma='scale'
)

print("Training SVM...")

svm.fit(X_train, y_train)

print("Training Complete!")



y_pred = svm.predict(X_test)

print("Prediction Complete!")


accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Cat", "Dog"]
)

fig, ax = plt.subplots(figsize=(6,6))
disp.plot(cmap="Blues", ax=ax, colorbar=True)

plt.title("Confusion Matrix - SVM Classifier")
plt.tight_layout()

plt.savefig("SCT_ML_3_CONFUSION.png", dpi=300)
plt.show()

import random

plt.figure(figsize=(12,8))

indices = random.sample(range(len(X_test)), 6)

for i, idx in enumerate(indices):

    image = X_test[idx].reshape(32,32)

    prediction = "Dog" if y_pred[idx]==1 else "Cat"
    actual = "Dog" if y_test[idx]==1 else "Cat"

    plt.subplot(2,3,i+1)
    plt.imshow(image, cmap='gray')
    plt.title(f"Pred:{prediction}\nActual:{actual}")
    plt.axis("off")

plt.tight_layout()
plt.savefig("SCT_ML_3_PREDICTIONS.png")
plt.show()

accuracy_percent = accuracy*100

plt.figure(figsize=(5,5))
plt.bar(["SVM Accuracy"], [accuracy_percent])
plt.ylim(0,100)
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy")

plt.text(0, accuracy_percent+2,
         f"{accuracy_percent:.2f}%",
         ha="center",
         fontsize=12)

plt.savefig("SCT_ML_3_ACCURACY.png")
plt.show()