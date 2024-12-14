import cv2 as cv
import matplotlib.pyplot as plt

image_path = "ryangosling-scaled.jpg"
img = cv.imread(image_path)

if img is None:
    print("Error: Image not found")
else:
    grayScale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    print(f"Grayscale image shape: {grayScale.shape}")

    face_classifier = cv.CascadeClassifier(
        cv.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_classifier.detectMultiScale(
        grayScale, scaleFactor=1.1, minNeighbors=25, minSize=(40, 40)
    )

    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 10)

    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    plt.figure(figsize=(5, 5))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()
