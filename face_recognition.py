import cv2
import json

recognizer = cv2.face.LBPHFaceRecognizer.create()
recognizer.read("face-model.yml") # face model from face_training.py
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
font = cv2.FONT_HERSHEY_COMPLEX

labels_file = 'labels.json'
names = {"0": "None"} # Default

try:
    with open(labels_file, 'r') as f:
        loaded_labels = json.load(f)
        names.update(loaded_labels) # Gabungkan dengan data yang sudah dimuat
except FileNotFoundError:
    print("[WARNING] File 'labels.json' tidak ditemukan. Nama tidak akan ditampilkan.")

cap = cv2.VideoCapture(1)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=20, minSize=(30,30))

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        
        # Ambil potongan wajah
        face_crop = gray[y:y+h, x:x+w]
        # Terapkan Histogram Equalization
        equalized_face = cv2.equalizeHist(face_crop)
        
        id, confidence = recognizer.predict(equalized_face)
        
        # print(f"ID Terdeteksi: {id}, Confidence: {confidence}")
        
        # Variabel default
        name = "Tidak Dikenal"
        Confidence_text = "{0}%".format(round(100-confidence))
        
        # cek confidence
        if confidence < 40:
            name = names.get(str(id), "Tidak Dikenal")
        
        # tampilkan nama dan confidence di frame
        cv2.putText(frame, str(name), (x+5, y-5), font, 1, (255,0,0), 2)
        cv2.putText(frame, str(Confidence_text), (x+5, y+h-5), font, 1, (255,255,0), 1)
        
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()