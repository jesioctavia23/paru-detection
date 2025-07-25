from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
import librosa
import numpy as np
import pickle
import traceback
import matplotlib
matplotlib.use('Agg')  # Agar matplotlib tidak membuka jendela GUI
import matplotlib.pyplot as plt
import librosa.display
from time import perf_counter

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
STATIC_FOLDER = os.path.join(os.path.dirname(__file__), 'static')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

log_path = os.path.join(os.path.dirname(__file__), 'log.txt')
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.pkl')

# Load model dan scaler
with open(model_path, 'rb') as f:
    model = pickle.load(f)
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def landing():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')


@app.route('/predict', methods=['POST'])
def predict():
    filepath = None
    try:
        with open(log_path, 'a') as log_file:
            log_file.write('\n===== MASUK ENDPOINT /predict =====\n')

        if 'file' not in request.files:
            return render_template("upload.html", error="Tidak ada file yang dikirim")

        file = request.files['file']
        if not file or not file.filename:
            return render_template("upload.html", error="File tidak valid")

        file_content = file.read()
        if not file_content or len(file_content) < 1000:
            return render_template("upload.html", error="File kosong atau terlalu kecil")

        file.stream.seek(0)
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        t_start = perf_counter()

        # Load audio
        try:
            import soundfile as sf
            y, sr = sf.read(filepath)
            y = np.array(y, dtype=np.float32)
        except:
            try:
             y, sr = librosa.load(filepath, sr=22050, duration=1.5, res_type='kaiser_fast')
            except Exception:
                return render_template("upload.html", error="File audio tidak valid.")

        if len(y) < 1000:
            return render_template("upload.html", error="File audio terlalu pendek untuk diproses.")

        # Ekstraksi MFCC
        try:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc.T, axis=0).reshape(1, -1)
        except Exception:
            return render_template("upload.html", error="Gagal ekstraksi MFCC dari audio.")

        # Prediksi
        mfcc_scaled = scaler.transform(mfcc_mean)
        prediction = model.predict(mfcc_scaled)[0]

        t_end = perf_counter()
        elapsed_ms = (t_end - t_start) * 1000

        # Simpan grafik waveform
        waveform_path = os.path.join(STATIC_FOLDER, 'waveform.png')
        mfcc_path = os.path.join(STATIC_FOLDER, 'mfcc.png')

        plt.figure(figsize=(10, 3))
        plt.plot(y)
        plt.title("Gelombang Suara")
        plt.tight_layout()
        plt.savefig(waveform_path)
        plt.close()

        # Simpan grafik MFCC
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfcc, x_axis='time', sr=sr)
        plt.colorbar()
        plt.title("MFCC")
        plt.tight_layout()
        plt.savefig(mfcc_path)
        plt.close()

        return render_template(
            "result.html",
            prediction=prediction,
            elapsed_time=elapsed_ms,
            waveform_img='/static/waveform.png',
            mfcc_img='/static/mfcc.png'
        )

    except Exception as e:
        with open(log_path, 'a') as log_file:
            log_file.write(traceback.format_exc())
        return "<h4 style='color:red'>Terjadi error saat prediksi. Cek file log.txt</h4>", 500

    finally:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
