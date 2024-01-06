import cv2

# Charger le modèle de détection de visage
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Chemins vers les fichiers de modèle de genre et d'âge
gender_model = 'gender_net.caffemodel'
gender_proto = 'gender_deploy.prototxt'
age_model = 'age_net.caffemodel'
age_proto = 'age_deploy.prototxt'

# Charger les modèles de classification du genre et de l'âge
gender_net = cv2.dnn.readNet(gender_model, gender_proto)
age_net = cv2.dnn.readNet(age_model, age_proto)

gender_list = ['Male', 'Female']
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Démarrer la capture vidéo
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Préparation de l'image du visage pour la classification du genre et de l'âge
        face_img = frame[y:y+h, x:x+w].copy()
        blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), (104, 117, 123), swapRB=True)

        # Prédiction du genre
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]

        # Prédiction de l'âge
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]

        overlay_text = f"{gender}, {age}"
        cv2.putText(frame, overlay_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow('Age and Gender Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
