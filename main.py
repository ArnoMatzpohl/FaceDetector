import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Get Video from webcam
videoStream = cv2.VideoCapture(0)

while True:
    # Read frame of video stream
    _, image = videoStream.read()

    # Convert into gray image
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect the face(s)
    faces = face_cascade.detectMultiScale(grayImage, 1.1, 4)

    # Draw a green rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display
    cv2.imshow('Face Detector', image)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Shows the video stream with the green rectangle
videoStream.release()