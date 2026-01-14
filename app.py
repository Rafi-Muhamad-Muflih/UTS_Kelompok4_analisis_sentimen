# ==========================================
# DEPLOYMENT STREAMLIT - SENTIMENT ANALYSIS
# (SINKRON DENGAN UPDATE COLAB TERBARU)
# ==========================================

import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

# 1. KONFIGURASI HALAMAN
st.set_page_config(
    page_title="Shopee Sentiment LR",
    page_icon="🛒",
    layout="centered"
)

# 2. DOWNLOAD NLTK RESOURCE
@st.cache_resource
def download_nltk():
    nltk.download('stopwords')

download_nltk()

# 3. LOAD ASSET (MODEL & TF-IDF)
@st.cache_resource
def load_assets():
    # Pastikan file .pkl hasil export dari Colab sudah diupload ke GitHub/Folder App
    model = joblib.load("logistic_regression_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    return model, tfidf

try:
    model_lr, tfidf_vectorizer = load_assets()
except Exception as e:
    st.error(f"⚠️ Gagal memuat model. Pastikan 'logistic_regression_model.pkl' dan 'tfidf_vectorizer.pkl' tersedia.")
    st.stop()

# 4. PREPROCESSING (IDENTIK DENGAN TAHAP 2 COLAB)
def clean_text_deploy(text):
    # Inisialisasi Stopwords
    stop_words = set(stopwords.words('indonesian'))
    
    # Daftar kata kunci negatif (Dikeluarkan dari stopwords agar tidak dihapus)
    kata_negatif = {
        'tidak', 'kurang', 'jangan', 'bukan', 'belom', 'belum', 'gak', 'ga',
        'jelek', 'buruk', 'kecewa', 'mahal', 'lelet', 'parah', 'nyesel',
        'rugi', 'penipu', 'bohong', 'rusak', 'pecah', 'asli', 'palsu',
        'cacat', 'lama', 'lemot', 'kapok', 'tipu', 'sampah', 'aseli', 'hancur'
    }
    stop_words = stop_words - kata_negatif

    # Kata-kata sampah (Noise) sesuai update Colab Anda
    kata_tambahan = {'aja', 'aj', 'ya ya', 'rb', 'si', 'nih', 'isi', 'jam', 'doang', 'ya', 'buruan'} 
    stop_words.update(kata_tambahan)

    # Proses Pembersihan
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text) # Hapus URL
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Hapus karakter non-huruf
    
    words = text.split()
    words = [w for w in words if w not in stop_words] # Filter Stopwords
    
    text = ' '.join(words)
    text = re.sub(r'\s+', ' ', text).strip() # Hapus spasi berlebih
    return text

# 5. CUSTOM UI (SHOPEE STYLE)
st.markdown("""
    <style>
    .stApp { background-color: #000000; color: #FFFFFF; }
    .main-title {
        color: #EE4D2D; font-size: 40px; font-weight: bold; text-align: center; margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #EE4D2D !important; color: white !important;
        width: 100%; border-radius: 10px; font-weight: bold; height: 50px;
    }
    .result-card {
        padding: 25px; border-radius: 15px; text-align: center; margin-top: 20px;
        box-shadow: 0 4px 15px rgba(238, 77, 45, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="main-title">🛒 Shopee Sentiment Analysis</div>', unsafe_allow_html=True)
st.write("<p style='text-align: center;'>Menggunakan Model <b>Logistic Regression</b> (UAS Kelompok 4)</p>", unsafe_allow_html=True)

# 6. INPUT FORM
with st.container():
    user_input = st.text_area("✍️ Masukkan ulasan pelanggan:", height=150, placeholder="Contoh: Barangnya jelek banget, pengiriman lama...")
    analyze_btn = st.button("MULAI ANALISIS")

# 7. LOGIKA PREDIKSI
if analyze_btn:
    if not user_input.strip():
        st.warning("⚠️ Mohon masukkan teks ulasan terlebih dahulu.")
    else:
        with st.spinner('Sedang memproses...'):
            # Tahap Preprocessing & Transformasi
            cleaned_input = clean_text_deploy(user_input)
            vectorized_input = tfidf_vectorizer.transform([cleaned_input])
            
            # Prediksi & Probabilitas
            prediction = model_lr.predict(vectorized_input)[0]
            probs = model_lr.predict_proba(vectorized_input)[0]
            confidence = max(probs) * 100

            # Tampilan Hasil
            if prediction == 1:
                bg_color = "rgba(25, 135, 84, 0.2)" # Hijau Transparan
                border_color = "#198754"
                label = "POSITIF 😊"
            else:
                bg_color = "rgba(220, 53, 69, 0.2)" # Merah Transparan
                border_color = "#dc3545"
                label = "NEGATIF 😡"

            st.markdown(f"""
                <div class="result-card" style="background-color: {bg_color}; border: 2px solid {border_color};">
                    <h2 style="color: white; margin-bottom: 5px;">{label}</h2>
                    <p style="font-size: 18px; color: white;">Tingkat Keyakinan: <b>{confidence:.2f}%</b></p>
                </div>
                """, unsafe_allow_html=True)

            # Detail Expander
            with st.expander("🔍 Lihat Detail Pemrosesan"):
                st.write(f"**Teks Setelah Preprocessing:** `{cleaned_input}`")
                st.caption("Model menggunakan Logistic Regression dengan penanganan kata negatif dan penghapusan noise.")

st.markdown("<br><hr><p style='text-align: center; font-size: 12px; opacity: 0.7;'>© 2024 UAS Kelompok 4 - STT Cipasung</p>", unsafe_allow_html=True)