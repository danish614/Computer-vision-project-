import cv2
from fer import FER


cap = cv2.VideoCapture(0)


detector = FER(mtcnn=True)


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]

        
        result = detector.detect_emotions(face_img)
        emotion_label = ""
        if result:
            emotions = result[0]["emotions"]
            
            emotion_label = max(emotions, key=emotions.get)
            prob = emotions[emotion_label]
            cv2.putText(frame, f"{emotion_label} ({prob:.2f})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        
        eyes = eye_cascade.detectMultiScale(gray[y:y+h, x:x+w])
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame[y:y+h, x:x+w], (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
            cv2.putText(frame, "Eye", (x+ex, y+ey-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

    cv2.imshow("Face + Eyes + Emotion", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
