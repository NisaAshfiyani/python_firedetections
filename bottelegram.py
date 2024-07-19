from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2
import mysql.connector
import io
from PIL import Image
import requests
import os
import pygame

app = Flask(__name__)

# Load model YOLO
model = YOLO("best1.pt")

# Inisialisasi kamera
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    raise Exception("Tidak ada kamera")

# Koneksi MySQL
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="laravel_db"
)
cursor = db.cursor()

# Telegram bot details
bot_token = '6923718356:AAH94JBzqAL0zGqSYJWAJSErCY0pnXKvQn4'
chat_id = '5151874874'

# Path file suara alarm
alarm_sound_file = 'alarm.wav'

def save_frame_to_db(frame, description):
    # Konversi frame ke byte array
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    byte_arr = io.BytesIO()
    pil_image.save(byte_arr, format='JPEG')
    byte_arr = byte_arr.getvalue()

    # Insert gambar dan deskripsi ke dalam database
    cursor.execute("INSERT INTO detections (image, description) VALUES (%s, %s)", (byte_arr, description))
    db.commit()

def generate_frames():
    while True:
        # Membaca frame dari kamera
        ret, frame = cam.read()
        if not ret:
            break

        # Melakukan prediksi
        results = model.predict(frame)

        # Menggambar bounding box dan label pada frame
        detection_made = False
        descriptions = set()
        for result in results:
            boxes = result.boxes.xyxy  # Mendapatkan koordinat box
            scores = result.boxes.conf  # Mendapatkan skor kepercayaan
            classes = result.boxes.cls  # Mendapatkan ID kelas

            for box, score, cls in zip(boxes, scores, classes):
                x1, y1, x2, y2 = map(int, box)
                label = f"{model.names[int(cls)]}: {score:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                descriptions.add(model.names[int(cls)])
                if model.names[int(cls)] in ['api kecil', 'api besar']:
                    detection_made = True

        # Simpan frame ke database dan kirim ke Telegram jika ada deteksi "api kecil" atau "api besar"
        if detection_made:
            description = ", ".join(descriptions)
            save_frame_to_db(frame, description)
            send_latest_image_to_telegram(description)  # Kirim gambar otomatis ke Telegram
            play_alarm_sound()  # Putar suara alarm

        # Encode frame ke JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Mengirimkan frame dalam format byte
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def send_latest_image_to_telegram(description):
    # Ambil gambar terbaru dari database
    cursor.execute("SELECT image, description, timestamp FROM detections ORDER BY id DESC LIMIT 1")
    result = cursor.fetchone()
    image_data = result[0]
    timestamp = result[2]

    # Simpan gambar ke file sementara
    image_path = 'detections-image.jpg'
    with open(image_path, "wb") as file:
        file.write(image_data)

    # Kirim pesan ke Telegram dengan deskripsi dan instruksi keamanan
    url = f'https://api.telegram.org/bot6923718356:AAH94JBzqAL0zGqSYJWAJSErCY0pnXKvQn4/sendPhoto'
    caption = f"PERINGATAN KEBAKARAN!\n\nTindakan:\nSegera evakuasi ke tempat yang aman.\nIkuti petunjuk dari petugas keamanan.\nJangan gunakan lift.\nHubungi pemadam kebakaran di [Nomor Telepon Pemadam Kebakaran].\n\nCatatan:\nHarap sebarkan pesan ini kepada semua orang di sekitar Anda.\nTetap tenang dan ikuti instruksi dengan aman.\n\nPengirim:\n[Sistem Keamanan PT. Fire Detection]\n\nDescription: {description}\nTimestamp: {timestamp}"
    with open(image_path, 'rb') as photo_file:
        files = {'photo': photo_file}
        data = {'chat_id': chat_id, 'caption': caption}
        response = requests.post(url, files=files, data=data)

    # Hapus file sementara setelah dikirim
    os.remove(image_path)

    if response.status_code == 200:
        print("Gambar berhasil dikirim")
    else:
        print("Gagal mengirim gambar")

def play_alarm_sound():
    # Inisialisasi pygame untuk memutar suara
    pygame.mixer.init()
    pygame.mixer.music.load(alarm_sound_file)
    pygame.mixer.music.play()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/live_stream')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
