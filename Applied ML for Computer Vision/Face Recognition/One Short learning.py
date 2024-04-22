import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier



# Directories for the raw and processed images
current_path = os.getcwd()
faces_dir = current_path + '/Faces/'
ariel_sharon_raw_dir = faces_dir + 'ariel_sharon_raw/'
ariel_sharon_processed_dir = faces_dir + 'ariel_sharon/'
if not os.path.exists(ariel_sharon_processed_dir):
    os.makedirs(ariel_sharon_processed_dir)
# Load the Haar Cascade for face detection
haarcascade_path = current_path + '/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haarcascade_path)

# segment faces using face_cascade
def segment_faces(image_path, output_path_base, face_index):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=2)
    # If multiple faces are detected, choose the largest one
    if len(faces) > 1:
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        faces = [faces[0]]
    # save only the largest face
    if len(faces) == 1:
        (x, y, w, h) = faces[0]
        face = img[y:y + h, x:x + w]
        face = cv2.resize(face, (100, 100))
        output_path = f"{output_path_base}ariel_sharon{face_index}.jpg"
        # store the segmented faces
        cv2.imwrite(output_path, face)
        return face_index + 1

face_index = 1
for filename in os.listdir(ariel_sharon_raw_dir):
    output_path_base = ariel_sharon_processed_dir
     # Increment the index for each face
    face_index = segment_faces(os.path.join(ariel_sharon_raw_dir, filename), output_path_base, face_index)

# flatter the image for model building
X, y = [], []
labels = {'ariel_sharon': 0, 'chris_evans': 1, 'chris_hemsworth': 2, 'mark_ruffalo': 3, 'robert_downey_jr': 4, 'scarlett_johansson': 5}

for label, value in labels.items():
    dir_path = faces_dir + label + '/'
    for filename in os.listdir(dir_path):
        img = cv2.imread(dir_path + filename, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (100, 100)).flatten()
        X.append(img)
        y.append(value)

X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Use PCA and Random Forest to do the classification on test dataset of all 6 characters
pca = PCA(n_components=20)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_pca, y_train)

# Predict and calculate accuracy
y_pred = rf_classifier.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)
print(f' Accuracy: {accuracy * 100:.2f}%')
