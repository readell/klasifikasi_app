import streamlit as st
st.set_page_config(page_title="Dashboard Prediksi Chat", layout="wide")
import pandas as pd
import pickle
import re
import nltk
from nltk import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

def clean_tweet_text(text):
    if not isinstance(text, str):
        return ""  # atau bisa return np.nan
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # hapus selain huruf dan spasi
    text = text.lower().strip()
    return text

def tokenized(text):
    #return re.split(r'W+', text)
    text = nltk.word_tokenize(text)
    return text

nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('indonesian')
def remove_stopwords(text):
    more_stopwords = ["kalo", "aja", "gak", "kah"]
    stopwords.extend(more_stopwords)
    text = [word for word in text if word not in stopwords]
    return text   

def stemming(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    text = [stemmer.stem(token) for token in text]
    return text

model_fraud = pickle.load(open('model_nb.sav', 'rb'))
vectorizer = pickle.load(open('tfidf_model.sav', 'rb'))


col1, col2 = st.columns([2, 2.5]) 
with col1:
    st.title('Dashboard Prediksi Chat')
    # upload file
    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

    
with col2:
    # tambah keterangan 
    st.markdown("### Keterangan")
    st.markdown("""
        Dashboard ini menggunakan model **Naïve Bayes**.  
        Chat yang diunggah dalam format **CSV** akan diproses melalui beberapa tahap:             
        ✅ **Pembersihan Teks**: Menghapus karakter khusus, angka, dan mengonversi ke huruf kecil.  
        ✅ **Normalisasi**: Mengubah kata singkatan menjadi bentuk baku.  
        ✅ **Tokenisasi & Stopwords Removal**: Memisahkan kata dan menghapus kata-kata tidak penting.  
        ✅ **Stemming**: Mengubah kata ke bentuk dasarnya.  
        ✅ **Prediksi**: Model akan memprediksi apakah chat bersifat **Negatif** atau **Positif**.  
        """, unsafe_allow_html=True)

with col1:
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.dataframe(data)

        if 'message' in data.columns:
            data = data[['message']] 

            data = data.dropna(subset=['message'])
            data = data[data['message'].str.strip() != '']

        if st.button('Mulai Analisis'):
        	# Pembersihan teks
            data['message'] = data['message'].apply(clean_tweet_text)
            
            # Normalisasi teks
            #data['message'] = data['message'].apply(text_normalize)

            # Tokenisasi
            data['message'] = data['message'].apply(tokenized)
            

            # Menghapus stopwords
            data['message'] = data['message'].apply(remove_stopwords)
           

            # Stemming
            data['message'] = data['message'].apply(stemming)
           

            data['message'] = data['message'].apply(lambda x: ' '.join(x))

            # ❗Hapus baris yang kosong setelah preprocessing
            data = data[data['message'].str.strip() != '']

            st.write("Hasil Preprocessing:")
            st.dataframe(data['message'], use_container_width=True) 

            data_input_vec = vectorizer.transform(data['message'])
            predictions = model_fraud.predict(data_input_vec)
            data['predicted_sentiment'] = predictions

            st.write("Hasil Prediksi:")
            st.dataframe(data[['message', 'predicted_sentiment']], use_container_width=True)  # Menampilkan kolom 'message' dan 'predicted_sentiment'

            sentiment_counts = data['predicted_sentiment'].value_counts()
            total_count = sentiment_counts.sum()
            sentiment_percentages = (sentiment_counts / total_count) * 100
            plt.style.use('seaborn-v0_8-darkgrid')

            st.markdown("<h5>📊 Hasil Klasifikasi:</h5>", unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize= (3, 2))

            colors = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f'][:len(sentiment_counts)]
            
            ax.bar(sentiment_counts.index, sentiment_counts.values, 
                    color=colors, width=0.5, alpha=0.8, edgecolor='black', linewidth=0.7)
                 
            for i, count in enumerate(sentiment_counts.values):
                percentage = f'{sentiment_percentages[i]:.2f}%'
                ax.text(i, count + 0.3, f'{count} ({percentage})', ha='center', va='bottom', fontsize=7, color='black')

            ax.set_ylim(0, max(sentiment_counts.values) * 1.3)
            ax.tick_params(axis='both', labelsize=7, color='black')  # Mengecilkan angka di sumbu X dan Y
            ax.yaxis.grid(True, linestyle='--', alpha=0.7)
            ax.set_title('Distribusi Sentimen', fontsize=8)
            ax.set_xlabel('Prediksi', fontsize=7)
            ax.set_ylabel('Jumlah', fontsize=7)
        
            plt.tight_layout()
            st.pyplot(fig)

    


