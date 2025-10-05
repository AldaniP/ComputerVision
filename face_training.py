import cv2
import numpy as np
import os
import json

dataset_path = "dataset/"
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer.create()

# Periksa apakah folder dataset ada dan tidak kosong
if not os.path.exists(dataset_path) or len(os.listdir(dataset_path)) == 0:
    print("Folder 'dataset' tidak ditemukan atau kosong. Harap buat dataset terlebih dahulu.")
    exit()

print("[INFO] Mempersiapkan data training...")

faces = []
ids = []
labels = {}
current_id = 1

# Dapatkan daftar nama folder (nama orang) di dalam folder dataset
person_names = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

# Iterasi melalui setiap folder orang
for name in person_names:
    person_path = os.path.join(dataset_path, name)
    
    # Simpan pasangan ID dan Nama ke dictionary labels
    if name not in labels.values():
        labels[str(current_id)] = name
        person_id = current_id
        current_id += 1
    
    # Dapatkan semua path gambar di dalam folder orang ini
    image_paths = [os.path.join(person_path, f) for f in os.listdir(person_path)]
    
    # Iterasi melalui setiap gambar
    for image_path in image_paths:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Tambahkan wajah (numpy array) dan id (int) ke list training
        faces.append(img)
        ids.append(person_id)

print(f"[INFO] Ditemukan {len(labels)} orang: {list(labels.values())}")
print("[INFO] Data selesai disiapkan.")

# Simpan mapping ID-Nama ke file labels.json
labels_file = 'labels.json'
with open(labels_file, 'w') as f:
    json.dump(labels, f, indent=4)
print(f"[INFO] Mapping nama disimpan di {labels_file}")

# Latih model
print("\n[INFO] Melatih model pengenalan wajah...")
recognizer.train(faces, np.array(ids))
print("[INFO] Training selesai!")

# Simpan model
recognizer.write("face-model.yml") 
print("[INFO] Model disimpan sebagai 'face-model.yml'")