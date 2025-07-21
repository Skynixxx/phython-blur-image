from cvzone.FaceDetectionModule import FaceDetector
import cv2
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Gunakan dialog file untuk memilih file
Tk().withdraw()  # Hindari jendela utama Tkinter
file_path = askopenfilename(title="Pilih file gambar", filetypes=[("Image files", "*.jpg *.jpeg *.png")])

# Debugging tambahan
print(f"Jalur file: {file_path}")
print(f"File ada: {os.path.exists(file_path)}")

# Baca gambar
img = cv2.imread(file_path)

# Periksa apakah gambar berhasil dibaca
if img is None:
    print(f"Error: Gambar tidak ditemukan atau tidak dapat dibaca di jalur {file_path}.")
else:
    detector = FaceDetector()
    img, bboxs = detector.findFaces(img, draw=False)

    if bboxs:
        for bbox in bboxs:
            x, y, w, h = bbox['bbox']
            face_img = img[y:y+h, x:x+w]
            blurred_face = cv2.GaussianBlur(face_img, (71, 71), 0)
            img[y:y+h, x:x+w] = blurred_face

    # Mengecilkan ukuran gambar
    scale_percent = 50  # Skala ukuran gambar menjadi 50% dari ukuran asli
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # Autosave gambar yang telah diproses
    save_path = os.path.splitext(file_path)[0] + "_processed.jpg"
    cv2.imwrite(save_path, resized_img)
    print(f"Gambar telah disimpan di: 'C:/Python/Python-Projects/python-blur-image/processed.jpg'")

    cv2.imshow("Image", resized_img)
    cv2.waitKey(0)