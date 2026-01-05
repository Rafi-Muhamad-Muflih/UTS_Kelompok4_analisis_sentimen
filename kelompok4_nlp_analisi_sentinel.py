import streamlit as st
import joblib
import re
import os

# 1. KONFIGURASI HALAMAN
st.set_page_config(page_title="Sentimen Shopee SVM 3D", page_icon="🛒", layout="centered")

# 2. CUSTOM CSS (DARK THEME + BLACK TEXT IN CARDS)
st.markdown("""
<style>
    .stApp {
        background: #000000;
        color: #ffffff;
    }

    .main-title {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        background: -webkit-linear-gradient(#EE4D2D, #FF7337);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }

    .input-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }

    .result-card {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.8), 0 0 20px rgba(238, 77, 45, 0.2);
        color: #000000 !important;
        text-align: center;
        transform: perspective(1000px) rotateX(2deg);
    }
    
    .result-card h3, .result-card p, .result-card b {
        color: #000000 !important;
    }

    .pos-text { color: #1b5e20 !important; font-size: 32px; font-weight: bold; }
    .neg-text { color: #b71c1c !important; font-size: 32px; font-weight: bold; }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #EE4D2D 0%, #FF7337 100%);
        color: white !important;
        border: none;
        padding: 12px;
        border-radius: 10px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# 3. LOAD MODEL SVM & TF-IDF
@st.cache_resource
def load_svm_assets():
    # Mengasumsikan file ada di folder yang sama dengan app.py
    model = joblib.load('svm_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    return model, tfidf

try:
    model_svm, tfidf_vectorizer = load_svm_assets()
except Exception as e:
    st.error(f"Gagal memuat aset model: {e}")
    st.info("Pastikan file svm_model.pkl dan tfidf_vectorizer.pkl ada di folder yang sama.")
    st.stop()

# 4. PREPROCESSING
def clean(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

# 5. UI UTAMA
st.markdown('<h1 class="main-title">🛒 Analisis Sentimen e-commerce shopee</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; opacity:0.7; color:white;">Analisis Sentimen Berbasis SVM</p>', unsafe_allow_html=True)

st.markdown('<div class="input-card" style="text-align:center;"> Create By Kelompok 4</div>', unsafe_allow_html=True)
input_user = st.text_area("✍️ Masukkan ulasan pelanggan:", height=120)
analyze_button = st.button("MULAI ANALISIS")
st.markdown('</div>', unsafe_allow_html=True)

# Logika Prediksi
if analyze_button:
    if input_user.strip():
        with st.spinner('Sedang menganalisis teks...'):
            # Tahap 1: Pembersihan
            cleaned_text = clean(input_user)
            
            # Tahap 2: Transformasi TF-IDF
            vectorized_text = tfidf_vectorizer.transform([cleaned_text])
            
            # Tahap 3: Prediksi
            prediction = model_svm.predict(vectorized_text)[0]
            
            # Tahap 4: Skor Keyakinan (Probabilitas)
            try:
                probs = model_svm.predict_proba(vectorized_text)[0]
                confidence = max(probs) * 100
            except:
                confidence = 100.0  # Jika probability=False saat training

            # Penentuan Label
            if prediction == 1:
                hasil, gaya_text = "POSITIF 😊", "pos-text"
            else:
                hasil, gaya_text = "NEGATIF 😡", "neg-text"
            
            # Tampilkan Hasil
            st.markdown(f"""
            <div class="result-card">
                <h3 style="margin-bottom:10px;">HASIL ANALISIS</h3>
                <hr style="border: 0.5px solid rgba(0,0,0,0.1);">
                <p class="{gaya_text}">{hasil}</p>
                <p style="font-size: 18px;">Tingkat Keyakinan: <b>{confidence:.2f}%</b></p>
                <div style="margin-top:15px; font-size:12px; opacity:0.6;">
                    Metode: Support Vector Machine | Kelompok 4
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Masukkan ulasan terlebih dahulu.")

st.markdown('<br><p style="text-align:center; opacity:0.2; font-size:12px; color:white;">UAS KELOMPOK 4 - 2024</p>', unsafe_allow_html=True)