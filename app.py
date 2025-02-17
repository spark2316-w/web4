import streamlit as st
import numpy as np
import scipy.signal as signal
import librosa
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import io
from pydub import AudioSegment

# Global Constants
FS = 44100
LOW_FREQ = 7000
HIGH_FREQ = 10000
KNOCK_THRESHOLD = 0.00005  # ค่า threshold สำหรับตรวจจับเสียงเคาะ

# ------------------------- ฟังก์ชันหลัก -------------------------
def plot_svm_decision_boundary(model, X, y):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    Z_class = model.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
    Z_class = Z_class.reshape(xx.shape)
    plt.contourf(xx, yy, Z_class, levels=[-0.5, 0.5, 1.5],
                 colors=['#CCFFCC', '#FFCCCC'], alpha=0.7)
    
    Z_dec = model.decision_function(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
    Z_dec = Z_dec.reshape(xx.shape)
    plt.contour(xx, yy, Z_dec, levels=[0], colors='k', linewidths=2)
    
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, edgecolors='k', cmap='bwr')
    
    support_vectors = model.support_vectors_
    support_vectors_pca = pca.transform(support_vectors)
    plt.scatter(support_vectors_pca[:, 0], support_vectors_pca[:, 1],
                s=100, facecolors='none', edgecolors='r', label='Support Vectors')
    
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('SVM Decision Boundary with Support Vectors')
    plt.legend()
    
    fig = plt.gcf()
    plt.close(fig)
    return fig

def extract_features(y, sr):
    sos = signal.butter(10, [LOW_FREQ, HIGH_FREQ], btype='bandpass', fs=sr, output='sos')
    y_filtered = signal.sosfilt(sos, y)
    mfccs = librosa.feature.mfcc(y=y_filtered, sr=sr, n_mfcc=40)
    mfcc_means = np.mean(mfccs, axis=1)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y_filtered, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y_filtered, sr=sr))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y_filtered))
    features = np.concatenate((mfcc_means, [spectral_centroid, spectral_bandwidth, zero_crossing_rate]))
    return features, y_filtered

def load_data_from_uploads(good_files, bad_files):
    features, labels = [], []
    sr_final = None
    for file in good_files:
        file.seek(0)
        if file.name.lower().endswith(".m4a"):
            file_bytes = file.read()
            file_buffer = convert_m4a_to_wav(file_bytes)
            audio, sr = librosa.load(file_buffer, sr=None)
        else:
            audio, sr = librosa.load(file, sr=None)
        feature, _ = extract_features(audio, sr)
        features.append(feature)
        labels.append(0)
        sr_final = sr
    for file in bad_files:
        file.seek(0)
        if file.name.lower().endswith(".m4a"):
            file_bytes = file.read()
            file_buffer = convert_m4a_to_wav(file_bytes)
            audio, sr = librosa.load(file_buffer, sr=None)
        else:
            audio, sr = librosa.load(file, sr=None)
        feature, _ = extract_features(audio, sr)
        features.append(feature)
        labels.append(1)
        sr_final = sr
    return np.array(features), np.array(labels), sr_final

def train_model(good_files, bad_files):
    features, labels, sr = load_data_from_uploads(good_files, bad_files)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=50)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = SVC(kernel='linear')
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    joblib.dump(model, 'best_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    return model, scaler, accuracy, features, labels, sr

def apply_bandpass_filter(y, sr):
    fft = np.fft.fft(y)
    freqs = np.fft.fftfreq(len(fft), d=1/sr)
    mask = ((freqs >= LOW_FREQ) & (freqs <= HIGH_FREQ)) | ((freqs <= -LOW_FREQ) & (freqs >= -HIGH_FREQ))
    fft_filtered = np.zeros_like(fft)
    fft_filtered[mask] = fft[mask]
    filtered_y = np.fft.ifft(fft_filtered).real
    return filtered_y

def extract_features_for_inference(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_means = np.mean(mfccs, axis=1)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
    features = np.concatenate((mfcc_means, [spectral_centroid, spectral_bandwidth, zero_crossing_rate]))
    return features

def load_model_and_scaler():
    try:
        best_model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
    except Exception as e:
        st.error("Error loading model or scaler: " + str(e))
        best_model, scaler = None, None
    return best_model, scaler

def predict_realtime(audio, sr):
    filtered_audio = apply_bandpass_filter(audio, sr)
    features = extract_features_for_inference(filtered_audio, sr)
    features = features.reshape(1, -1)
    best_model, scaler = load_model_and_scaler()
    features = scaler.transform(features)
    prediction = best_model.predict(features)
    return "Good Material" if prediction[0] == 0 else "Faulty Material"

def convert_m4a_to_wav(audio_bytes):
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="m4a")
    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    wav_io.seek(0)
    return wav_io

def plot_fft(y, sr, title="FFT Spectrum"):
    fft_vals = np.abs(np.fft.fft(y))
    freqs = np.fft.fftfreq(len(fft_vals), d=1/sr)
    idx = np.where(freqs >= 0)
    freqs = freqs[idx]
    fft_vals = fft_vals[idx]
    
    plt.figure(figsize=(8, 4))
    plt.plot(freqs, fft_vals)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title(title)
    plt.tight_layout()
    fig = plt.gcf()
    plt.close(fig)
    return fig

# ------------------------- ส่วนติดต่อผู้ใช้ด้วย Streamlit -------------------------
st.sidebar.title("เมนูหลัก")
option = st.sidebar.selectbox("เลือกหมวดหมู่", ["หน้าแรก", "Train-model", "Real-Time", "Drop File"])

if option == "หน้าแรก":
    st.title("ยินดีต้อนรับสู่ Knock Detection App")
    st.image("image2.jpg", caption="Sunrise by the mountains")

elif option == "Train-model":
    st.title("ฝึกโมเดล (Train Model)")
    st.write("อัปโหลดไฟล์เสียงสำหรับ Good Material และ Faulty Material")
    good_files = st.file_uploader("เลือกไฟล์เสียง Good:", type=["wav", "mp3", "ogg", "m4a"], accept_multiple_files=True)
    bad_files = st.file_uploader("เลือกไฟล์เสียง Bad:", type=["wav", "mp3", "ogg", "m4a"], accept_multiple_files=True)
    if st.button("เริ่มฝึกโมเดล"):
        if not good_files or not bad_files:
            st.error("กรุณาอัปโหลดไฟล์ทั้งสำหรับ Good และ Bad")
        else:
            with st.spinner("กำลังฝึกโมเดล..."):
                model, scaler_obj, accuracy, features, labels, sr = train_model(good_files, bad_files)
            st.write(f"ความแม่นยำของโมเดล: {accuracy*100:.2f}%")
            fig = plot_svm_decision_boundary(model, features, labels)
            st.pyplot(fig)
            st.success("ฝึกโมเดลและบันทึกโมเดลเรียบร้อยแล้ว")
            st.info("หลังจากฝึกโมเดลใหม่ กรุณารีเฟรชหน้าเว็บเพื่อโหลดโมเดลล่าสุดในส่วน Real-Time และ Drop File")

elif option == "Real-Time":
    st.title("Real-Time Detection")
    st.write("บันทึกเสียงและวิเคราะห์เสียงแบบเรียลไทม์")
    audio_value = st.audio_input("Record a voice message")
    if audio_value is not None:
        audio_bytes = audio_value.read()
        st.audio(audio_bytes, format="audio/wav")
        if st.button("วิเคราะห์เสียง"):
            audio_buffer = io.BytesIO(audio_bytes)
            try:
                audio, sr = librosa.load(audio_buffer, sr=FS, mono=True)
                st.write(f"Audio loaded. Sampling Rate = {sr} Hz")
                energy = np.sum(audio**2) / len(audio)
                st.write(f"Energy: {energy:.6f}")

                
                if energy < KNOCK_THRESHOLD:
                    st.warning("ไม่พบเสียงเคาะในไฟล์เสียง")
                else:
                    filtered_audio = apply_bandpass_filter(audio, sr)
                    
                    # แสดงกราฟ Time Domain
                    t = np.linspace(0, len(audio)/sr, len(audio))
                    t_filtered = np.linspace(0, len(filtered_audio)/sr, len(filtered_audio))
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                    ax1.plot(t, audio, label="Original Audio")
                    ax1.set_title("Original Audio")
                    ax1.set_xlabel("Time (s)")
                    ax1.set_ylabel("Amplitude")
                    ax1.legend()
                    ax2.plot(t_filtered, filtered_audio, color="orange", label="Filtered Audio")
                    ax2.set_title("Filtered Audio")
                    ax2.set_xlabel("Time (s)")
                    ax2.set_ylabel("Amplitude")
                    ax2.legend()
                    fig.tight_layout()
                    st.pyplot(fig)
                    
                    # FFT ของสัญญาณต้นฉบับ
                    fft_fig_orig = plot_fft(audio, sr, title="FFT Spectrum (Original Audio)")
                    st.pyplot(fft_fig_orig)
                    # FFT ของสัญญาณที่ผ่านการกรอง
                    fft_fig_filt = plot_fft(filtered_audio, sr, title="FFT Spectrum (Filtered Audio)")
                    st.pyplot(fft_fig_filt)
                    
                    result = predict_realtime(audio, sr)
                    st.write("ผลการวิเคราะห์:", result)
            except Exception as e:
                st.error("Error processing audio: " + str(e))

elif option == "Drop File":
    st.title("Drop File Detection")
    st.write("หรืออัปโหลดไฟล์เสียง (wav, mp3, ogg, m4a) เพื่อวิเคราะห์เสียง")
    uploaded_file = st.file_uploader("เลือกไฟล์เสียง", type=["wav", "mp3", "ogg", "m4a"])
    if uploaded_file is not None:
        file_name = uploaded_file.name.lower()
        file_bytes = uploaded_file.read()
        if file_name.endswith("m4a"):
            audio_buffer = convert_m4a_to_wav(file_bytes)
            file_format = "audio/wav"
        else:
            audio_buffer = io.BytesIO(file_bytes)
            if file_name.endswith("wav"):
                file_format = "audio/wav"
            elif file_name.endswith("mp3"):
                file_format = "audio/mp3"
            elif file_name.endswith("ogg"):
                file_format = "audio/ogg"
            else:
                file_format = "audio/wav"
        st.audio(audio_buffer, format=file_format)
        audio_buffer.seek(0)
        try:
            audio, sr = librosa.load(audio_buffer, sr=FS, mono=True)
            st.write(f"Audio loaded. Sampling Rate = {sr} Hz")
            if st.button("วิเคราะห์เสียง"):
                energy = np.sum(audio**2) / len(audio)
                st.write(f"Energy: {energy:.6f}")
                
                # FFT ของสัญญาณต้นฉบับ
                fft_fig_orig = plot_fft(audio, sr, title="FFT Spectrum (Original Audio)")
                st.pyplot(fft_fig_orig)
                
                if energy < KNOCK_THRESHOLD:
                    st.warning("ไม่พบเสียงเคาะในไฟล์เสียง")
                else:
                    filtered_audio = apply_bandpass_filter(audio, sr)
                    t = np.linspace(0, len(audio)/sr, len(audio))
                    t_filtered = np.linspace(0, len(filtered_audio)/sr, len(filtered_audio))
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                    ax1.plot(t, audio, label='Original Signal')
                    ax1.set_title('Original Signal')
                    ax1.set_xlabel('Time (s)')
                    ax1.set_ylabel('Amplitude')
                    ax2.plot(t_filtered, filtered_audio, label='Filtered Signal', color='orange')
                    ax2.set_title('Filtered Signal')
                    ax2.set_xlabel('Time (s)')
                    ax2.set_ylabel('Amplitude')
                    fig.tight_layout()
                    st.pyplot(fig)
                    
                    # FFT ของสัญญาณที่ผ่านการกรอง
                    fft_fig_filt = plot_fft(filtered_audio, sr, title="FFT Spectrum (Filtered Audio)")
                    st.pyplot(fft_fig_filt)
                    
                    result = predict_realtime(audio, sr)
                    st.write("ผลการวิเคราะห์:", result)
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการโหลดไฟล์เสียง: {e}")
