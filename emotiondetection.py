import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras.utils import img_to_array

# Face Classifier
face_classifier = cv2.CascadeClassifier(r'C:\Users\hp\Downloads\emotion detection\emotion detection\haarcascade_frontalface_default.xml')
# Loading the model
classifier = load_model(r'C:\Users\hp\Downloads\emotion detection\emotion detection\h5 file')

# Different types of emotions
emotion_labels = ['Angry', 'Disgust', 'Fear',
                  'Happy', 'Neutral', 'Sad', 'Surprise']

# Capturing video using webcam
cap = cv2.VideoCapture(0)

while True:
    # reading the video frame-by-frame
    _, frame = cap.read()
    labels = []
    # converting each frame from BGR to Gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    # adding rectangle box outside faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        # converting image to an image array
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # predicting the emotion
            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            label_position = (x, y)
            cv2.putText(frame, label, label_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            # in case no face is detected
            cv2.putText(frame, 'No Faces', (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # output window title
    cv2.imshow('Emotion Detector', frame)
    # press 'q' to close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()