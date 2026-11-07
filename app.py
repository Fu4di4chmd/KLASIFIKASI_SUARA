import os
import librosa
import numpy as np
from flask import Flask, render_template, request, jsonify
from moviepy import AudioFileClip
from sklearn.ensemble import RandomForestClassifier
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ==========================================
# BAGIAN 1: FUNGSI MACHINE LEARNING
# ==========================================
def muat_audio_aman(file_path):
    try:
        y, sr = librosa.load(file_path, duration=5.0)
        return y, sr
    except Exception:
        temp_wav = "temp_baca.wav"
        klip = AudioFileClip(file_path)
        klip.write_audiofile(temp_wav, logger=None)
        klip.close()
        y, sr = librosa.load(temp_wav, duration=5.0)
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        return y, sr

def ekstrak_fitur(file_path):
    try:
        y, sr = muat_audio_aman(file_path)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)
        rms = np.mean(librosa.feature.rms(y=y).T, axis=0)
        return np.hstack([mfcc, cent, rms])
    except Exception as e:
        print(f"Gagal memproses {file_path}: {e}")
        return None

# Latih Model Global saat server berjalan
print("Mengekstrak dataset dan melatih model...")
daftar_folder = {"after-knalpot_standart": 0, "after-knalpot_brong": 1}
fitur_list, label_list = [], []

for nama_folder, label in daftar_folder.items():
    if os.path.exists(nama_folder):
        for nama_file in os.listdir(nama_folder):
            if nama_file.endswith(('.mp3', '.wav', '.m4a')):
                path_lengkap = os.path.join(nama_folder, nama_file)
                fitur = ekstrak_fitur(path_lengkap)
                if fitur is not None:
                    fitur_list.append(fitur)
                    label_list.append(label)

X = np.array(fitur_list)
y = np.array(label_list)

model = RandomForestClassifier(n_estimators=100, random_state=42)
if len(X) > 0:
    model.fit(X, y)
    print("Model berhasil dilatih dan siap digunakan!")
else:
    print("Peringatan: Tidak ada data audio ditemukan di folder dataset!")

# ==========================================
# BAGIAN 2: FLASK WEB ROUTING
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({"error": "Tidak ada file audio"}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({"error": "File kosong"}), 400

    temp_path = secure_filename("rekaman_browser.wav")
    file.save(temp_path)

    # 1. Cek tingkat kebisingan/volume suara
    try:
        y, sr = librosa.load(temp_path, duration=3.0)
        volume = np.max(np.abs(y)) # Mengambil amplitudo tertinggi
    except:
        volume = 0

    # Jika volume sangat kecil (hening), kembalikan SEPI
    # if volume < 0.01: 
    #     if os.path.exists(temp_path):
    #         os.remove(temp_path)
    #     return jsonify({"hasil": "SEPI"})

    # 2. Jika ada suara signifikan, lakukan ekstraksi dan prediksi ML
    fitur_uji = ekstrak_fitur(temp_path)
    if os.path.exists(temp_path):
        os.remove(temp_path) 

    if fitur_uji is not None:
        prediksi = model.predict([fitur_uji])[0]
        # 0 = Standart, 1 = Brong
        hasil = "BRONG" if prediksi == 1 else "STANDART"
        return jsonify({"hasil": hasil})
    else:
        return jsonify({"error": "Gagal mengekstrak fitur"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)