import cv2

# Load pre-trained models for face detection, gender, and age prediction
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gender_model = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')
age_model = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')

# Define the labels for gender and age prediction
gender_list = ['Male', 'Female']
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Function to predict gender and age for each face
def predict_gender_and_age(face):
    # Preprocess the face for gender prediction
    blob_gender = cv2.dnn.blobFromImage(face, scalefactor=1.0, size=(227, 227), mean=(78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    gender_model.setInput(blob_gender)
    gender_preds = gender_model.forward()

    # Preprocess the face for age prediction
    blob_age = cv2.dnn.blobFromImage(face, scalefactor=1.0, size=(227, 227), mean=(78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    age_model.setInput(blob_age)
    age_preds = age_model.forward()

    # Get the gender and age labels with the highest confidence
    gender = gender_list[gender_preds[0].argmax()]
    age = age_list[age_preds[0].argmax()]

    return gender, age

# Function to detect faces in a frame and predict gender and age for each face
def detect_faces_and_predict(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop through each face and predict gender and age
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        gender, age = predict_gender_and_age(face)

        # Draw the predicted gender and age on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f'{gender}, {age}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the frame with predictions
    cv2.imshow('Gender and Age Detection', frame)

# Function to capture frames from webcam and perform gender and age detection
def perform_detection():
    # Create a VideoCapture object for the webcam
    cap = cv2.VideoCapture(0)  # 0 for the default webcam, you may need to change this if you have multiple cameras
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # Check if the frame was captured successfully
        if not ret:
            break
        
        # Detect faces in the frame and predict gender and age
        detect_faces_and_predict(frame)
        
        # Check for the 'q' key to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('stop'):
            break
    
    # Release the VideoCapture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

perform_detection()
