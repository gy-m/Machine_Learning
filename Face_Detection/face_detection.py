import cv2 as cv

# Read image from your local file system
original_image = cv.imread('Face_Detection\pic.jpg')

# Convert color image to grayscale for Viola-Jones
grayscale_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)

# Load the classifier and create a cascade object for face detection
face_cascade = cv.CascadeClassifier('Face_Detection\haarcascade_frontalface_alt.xml')

detected_faces = face_cascade.detectMultiScale(grayscale_image, 1.3)

for (column, row, width, height) in detected_faces:
    cv.rectangle(
        original_image,
        (column, row),
        (column + width, row + height),
        (0, 255, 0),
        2
    )

small = cv.resize(original_image, (0,0), fx=0.3, fy=0.3)
cv.imshow('Image', small)
cv.waitKey(0)
cv.destroyAllWindows()