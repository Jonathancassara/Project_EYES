import cv2

# Charger le modèle de détection de visage pré-entraîné de OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Démarrer la capture vidéo à partir de la première caméra connectée
cap = cv2.VideoCapture(0)

while True:
    # Capturer image par image
    ret, frame = cap.read()

    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détecter les visages dans l'image
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Dessiner des rectangles autour des visages détectés
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Afficher l'image résultante
    cv2.imshow('Face Detection', frame)

    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la capture vidéo et fermer les fenêtres ouvertes
cap.release()
cv2.destroyAllWindows()
