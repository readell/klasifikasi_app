import streamlit as st
st.set_page_config(page_title="Dashboard Prediksi Chat", layout="wide")
import pandas as pd
import pickle
import re
import os
import nltk
nltk.download('punkt_tab')
from nltk import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import matplotlib.pyplot as plt

# ============= CSS Custom =============
st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 0rem;
}
.metric-card {
    background: #f8f9fa;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    margin-bottom: 10px;
}
div.stButton > button:first-child {
    background-color: #007bff;
    color: white;
    border-radius: 8px;
    height: 50px;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
    border: none;
}
div.stButton > button:hover {
    background-color: #0056b3;
}
</style>
""", unsafe_allow_html=True)

# ============= Preprocessing =============
def clean_tweet_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().strip()
    return text

nltk.download('punkt')
def tokenized(text):
    return nltk.word_tokenize(text)

nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('indonesian')
def remove_stopwords(text):
    more_stopwords = ["kalo", "aja", "gak", "kah"]
    stopwords.extend(more_stopwords)
    return [word for word in text if word not in stopwords]

def stemming(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return [stemmer.stem(token) for token in text]

# ============= Load Model =============
base_dir = os.path.dirname(os.path.abspath(__file__))
model_fraud = pickle.load(open(os.path.join(base_dir, 'model_nb.sav'), 'rb'))
vectorizer = pickle.load(open(os.path.join(base_dir, 'tfidf_model.sav'), 'rb'))

# ============= Header Dashboard =============
st.markdown("""
<h1 style='font-size: 35px; font-weight: bold;'> Dashboard Prediksi Chat</h1>
<p style='font-size: 16px; color: #555;'>Dashboard analisis percakapan prostitusi dan tidak prostitusi.</p>
<hr>
""", unsafe_allow_html=True)

# ============= Sidebar =============
with st.sidebar:
    st.header("Upload File")
    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])
    start_process = st.button("Mulai Analisis")
    st.markdown("---")
    st.subheader("ℹ Petunjuk")
    st.write("• File berformat CSV")
    st.markdown("""
        ### Proses Analisis
        Dashboard ini menggunakan model algoritma Naive Bayes.
        File yang diunggah dalam format CSV akan diproses melalui beberapa tahapan:
        
        - Cleaning → menghapus karakter tidak penting  
        - Normalisasi  
        - Tokenisasi & Stopwords Removal  
        - Stemming  
        - Prediksi → menentukan apakah chat bersifat "Positive" atau "Negative"
        
        **Positive** = Tidak ada indikasi prostitusi  
        **Negative** = Ada indikasi prostitusi
        """)
# ============= Konten Utama =============
if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    # Data awal hanya ditampilkan jika tombol belum ditekan
    if not start_process:
        st.subheader("Data Awal")
        st.dataframe(data, use_container_width=True)

    if "message" not in data.columns:
        st.error("Format atau jenis file tidak cocok")
        st.stop()

    data = data.dropna(subset=["message"])
    data = data[data["message"].str.strip() != ""]

    if start_process:

        # ======== PREPROCESSING ========
        data['message'] = data['message'].apply(clean_tweet_text)
        data['message'] = data['message'].apply(tokenized)
        data['message'] = data['message'].apply(remove_stopwords)
        data['message'] = data['message'].apply(stemming)
        data['message'] = data['message'].apply(lambda x: " ".join(x))
        data = data[data['message'].str.strip() != ""]

        data_vectors = vectorizer.transform(data['message'])
        data['predicted_sentiment'] = model_fraud.predict(data_vectors)

        # Normalisasi nilai sentiment
        data['predicted_sentiment'] = data['predicted_sentiment'].astype(str).str.strip().str.lower()

        # Mapping teks ke angka
        mapping = {
            'positive': 1,
            'negative': 0,
        }
        data['predicted_sentiment'] = data['predicted_sentiment'].map(mapping)

        # Hitung jumlah
        total_data = len(data)
        total_positive = (data['predicted_sentiment'] == 1).sum()
        total_negative = (data['predicted_sentiment'] == 0).sum()

        # Buat variabel sentiment_counts (MENCEGAH ERROR)
        sentiment_counts = pd.Series({
            "Positive": total_positive,
            "Negative": total_negative
        })

        # ============= Kartu Informasi =============
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"<div class='metric-card'><h3>{total_data}</h3><p>Total Data</p></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-card'><h3 style='color:#2ecc71'>{total_positive}</h3><p>Positive/Tidak Indikasi Prostitusi</p></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='metric-card'><h3 style='color:#e74c3c'>{total_negative}</h3><p>Negative/Indikasi Prostitusi</p></div>", unsafe_allow_html=True)

        st.markdown("---")

        # ============= Tabel & Grafik Berdampingan =============
        st.subheader("Hasil Analisis")

        col_table, col_chart = st.columns([2, 1])

        with col_table:
            st.markdown("### Hasil Prediksi")
            st.dataframe(data[['message', 'predicted_sentiment']], use_container_width=True)

        with col_chart:
            st.markdown("### Distribusi Sentimen")
            fig, ax = plt.subplots(figsize=(3, 2))
            ax.bar(sentiment_counts.index, sentiment_counts.values, alpha=0.8)

            for i, val in enumerate(sentiment_counts.values):
                ax.text(i, val + 0.1, str(val), ha='center', fontsize=7)

            ax.set_xlabel("Label", fontsize=7)
            ax.set_ylabel("Jumlah", fontsize=7)
            ax.set_title("Distribusi Prediksi", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)

            st.download_button(
                label="⬇ Download Hasil",
                data=data.to_csv(index=False),
                file_name="hasil_prediksi.csv",
                mime="text/csv"
            )

        # ============= Informasi Distribusi =============

