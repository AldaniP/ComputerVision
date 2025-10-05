import cv2
import os

# Inisialisasi Cascade Classifier dan Webcam
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(1) 

# Path utama untuk dataset
dataset_path = "dataset/"

# Buat folder dataset utama jika belum ada
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
    
person_name = input("Masukkan Nama Orang (tanpa spasi): ")

# Buat path spesifik untuk orang ini di dalam folder dataset
person_dataset_path = os.path.join(dataset_path, person_name)

# Buat folder untuk orang ini jika belum ada
if not os.path.exists(person_dataset_path):
    os.makedirs(person_dataset_path)

print(f"\n[INFO] Dataset akan dibuat untuk: {person_name}. Lihat ke kamera dan tunggu...")
print(f"Gambar akan disimpan di: {person_dataset_path}")

count = 0
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 20, minSize=(30, 30))
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        count += 1
        
        face_crop = gray[y:y+h, x:x+w]
        equalized_face = cv2.equalizeHist(face_crop)
        
        # Simpan gambar ke folder orang yang bersangkutan
        file_name = f"{count}.jpg"
        cv2.imwrite(os.path.join(person_dataset_path, file_name), equalized_face)
    
    cv2.imshow("Camera", frame)
    
    if cv2.waitKey(1) == ord('q'):
        break
    elif count == 50: # Mengambil 50 foto
        break

print(f"\n[INFO] {count} gambar telah diambil dan disimpan.")
cap.release()
cv2.destroyAllWindows()