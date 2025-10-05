import cv2
import os
import json

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(1)
dataset_path = "dataset/"

if not os.path.exists(dataset_path):
    os.mkdir(dataset_path)
    
person_id = input("Masukan ID Orang(angka): ")
person_name = input("Masukan Nama Orang: ")
print(f"\n[INFO]) Dataset akan dibuat untuk ID: {person_id} dan Nama: {person_name}")

labels_file = 'labels.json'

try:
    with open(labels_file, 'r') as f:
        labels = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    labels = {} # buat catatan baru jika file belum ada atau kosong

# update ID dan nama ke catatan
labels[person_id] = person_name

# simpan kembali semua data ke file json
with open(labels_file, 'w') as f:
    json.dump(labels, f, indent=4)
    
print(f"[INFO] Nama '{person_name}' telah disimpan untuk ID {person_id} di {labels_file}")

count = 0
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 20, minSize=(30, 30))
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        count+=1
        
        # Ambil potongan wajah
        face_crop = gray[y:y+h, x:x+w]
        # Terapkan Histogram Equalization
        equalized_face = cv2.equalizeHist(face_crop)
        
        cv2.imwrite(dataset_path+"Person-"+str(person_id) +"-"+str(count)+".jpg", equalized_face)
    
    cv2.imshow("Camera", frame)
    
    if cv2.waitKey(1) == ord('q'):
        break
    elif count == 50: #stop when 30 photos have been taken 
        break

cap.release()
cv2.destroyAllwindows()