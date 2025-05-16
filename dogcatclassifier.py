import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from skimage.feature import hog
import matplotlib.pyplot as plt


def load_images_from_folder(folder, size=(64, 64), max_images=None):
    data = []
    raw_images = []
    count = 0

    for filename in os.listdir(folder):
        if filename.lower().endswith(".jpg") or filename.lower().endswith(".png"):
            label = 0 if "cat" in filename.lower() else 1  # 0: cat, 1: dog
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)

            if img is None:
                continue  # Skip unreadable images

            # Resize and preprocess
            img_resized = cv2.resize(img, size)
            gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            gray = gray / 255.0

            # HOG feature extraction
            features = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)

            data.append((features, label))
            raw_images.append((img_resized, label, filename))  # Keep raw image for plotting

            count += 1
            if max_images and count >= max_images:
                break

    return data, raw_images


data, raw_images = load_images_from_folder("./train", max_images=None)
X = [d[0] for d in data]
y = [d[1] for d in data]

X_train, X_test, y_train, y_test, raw_train, raw_test = train_test_split(
    X, y, raw_images, test_size=0.2, random_state=42
)


clf = svm.SVC(kernel='rbf', class_weight='balanced')
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f" Accuracy: {accuracy * 100:.2f}%")

cm = confusion_matrix(y_test, y_pred)
print(" Confusion Matrix:")
print(cm)


print("\n Showing 5 predictions...")

plt.figure(figsize=(15, 6))
for i in range(5):
    idx = np.random.randint(0, len(raw_test))
    img, true_label, fname = raw_test[idx]
    predicted_label = y_pred[idx]

    title = f"True: {'Cat' if true_label == 0 else 'Dog'} | Pred: {'Cat' if predicted_label == 0 else 'Dog'}"
    color = 'green' if true_label == predicted_label else 'red'

    plt.subplot(1, 5, i + 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title, color=color)
    plt.axis('off')

plt.tight_layout()
plt.show()
