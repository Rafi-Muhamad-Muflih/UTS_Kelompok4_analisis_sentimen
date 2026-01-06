import streamlit as st
import joblib
import re
import nltk
import os
from nltk.corpus import stopwords

# --- DOWNLOAD RESOURCES ---
@st.cache_resource
def download_nltk():
    nltk.download('stopwords')

download_nltk()

# 1. KONFIGURASI HALAMAN
st.set_page_config(page_title="Sentimen Shopee LR", page_icon="🛒", layout="centered")

# 2. CUSTOM CSS
st.markdown("""
<style>
    .stApp { background: #000000; color: #ffffff; }
    .main-title {
        font-size: 3rem; font-weight: 800; text-align: center;
        background: -webkit-linear-gradient(#EE4D2D, #FF7337);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .input-card {
        background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(10px);
        border-radius: 20px; padding: 25px; border: 1px solid rgba(255, 255, 255, 0.1);
        margin-top: 20px;
    }
    .result-card {
        background: rgba(255, 255, 255, 0.9); border-radius: 20px; padding: 30px;
        color: #000000 !important; text-align: center; margin-top: 20px;
    }
    .pos-text { color: #1b5e20 !important; font-size: 32px; font-weight: bold; }
    .neg-text { color: #b71c1c !important; font-size: 32px; font-weight: bold; }
    .stButton>button { 
        width: 100%; 
        background: linear-gradient(135deg, #EE4D2D 0%, #FF7337 100%); 
        color: white !important; 
        font-weight: bold;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# 3. LOAD MODEL & TF-IDF
@st.cache_resource
def load_assets():
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "logistic_regression_model.pkl")
    tfidf_path = os.path.join(base_dir, "tfidf_vectorizer.pkl")
    model = joblib.load(model_path)
    tfidf = joblib.load(tfidf_path)
    return model, tfidf

try:
    model_lr, tfidf_vectorizer = load_assets()
except Exception as e:
    st.error(f"Gagal memuat aset model: {e}")
    st.info("Pastikan file 'logistic_regression_model.pkl' dan 'tfidf_vectorizer.pkl' sudah diunggah.")
    st.stop()

# 4. PREPROCESSING (Sama persis dengan Colab)
def clean_for_deploy(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    stop_words = set(stopwords.words('indonesian'))
    kata_negatif = {
        'tidak', 'kurang', 'jangan', 'bukan', 'belom', 'belum', 'bangsat', 'lambat',
        'gak', 'ga', 'jelek', 'buruk', 'kecewa', 'mahal', 'jancok', 'tailah', 'lelet',
        'parah', 'nyesel', 'rugi', 'penipu', 'bohong', 'rusak', 'pecah', 'asli', 'palsu',
        'kw', 'cacat', 'lama', 'lemot', 'kapok', 'tipu', 'sampah', 'aseli',
    }
    stop_words = stop_words - kata_negatif
    
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words).strip()

# 5. UI UTAMA
st.markdown('<h1 class="main-title">🛒 Analisis Sentimen Shopee</h1>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; color:white; margin-bottom:20px;">Created By Kelompok 4</p>', unsafe_allow_html=True)
    
    input_user = st.text_area("✍️ Masukkan ulasan pelanggan:", height=120, placeholder="Contoh: Barangnya jelek sekali, saya sangat kecewa...")
    analyze_button = st.button("MULAI ANALISIS")
    st.markdown('</div>', unsafe_allow_html=True)

if analyze_button:
    if input_user.strip():
        with st.spinner('Sedang menganalisis sentimen...'):
            # Tahap 1: Pembersihan
            cleaned_text = clean_for_deploy(input_user)
            # Tahap 2: Transformasi TF-IDF
            vectorized_text = tfidf_vectorizer.transform([cleaned_text])
            # Tahap 3: Prediksi
            prediction = model_lr.predict(vectorized_text)[0]
            # Tahap 4: Probabilitas
            try:
                probs = model_lr.predict_proba(vectorized_text)[0]
                confidence = max(probs) * 100
            except:
                confidence = 100.0

            hasil = "POSITIF 😊" if prediction == 1 else "NEGATIF 😡"
            gaya = "pos-text" if prediction == 1 else "neg-text"
            
            st.markdown(f"""
            <div class="result-card">
                <h3 style="color: black;">HASIL ANALISIS</h3>
                <hr style="border: 0.5px solid rgba(0,0,0,0.1);">
                <p class="{gaya}">{hasil}</p>
                <p style="color: black; font-size: 18px;">Tingkat Keyakinan: <b>{confidence:.2f}%</b></p>
                <div style="margin-top:15px; font-size:12px; opacity:0.6; color: black;">
                    Metode: Logistic Regression | Custom Stopwords | Oversampling
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Mohon masukkan teks ulasan terlebih dahulu.")

st.markdown('<br><p style="text-align:center; opacity:0.5; font-size:12px; color:white;">UAS KELOMPOK 4 - 2024/2025</p>', unsafe_allow_html=True)