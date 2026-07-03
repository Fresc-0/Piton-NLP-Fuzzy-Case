import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Sayfa Tasarımı Ayarları
st.set_page_config(page_title="Müşteri Yorumu Analiz Sistemi", page_icon="📊", layout="centered")

st.title("📊 Müşteri Yorumu NLP & Bulanık Mantık Analiz Sistemi")
st.write("Piton Technology - Case Çalışması İnteraktif Arayüzü")
st.markdown("---")

# ==========================================
# 🧠 ARKA PLAN: MODEL VE VERİ HAZIRLIĞI (HIZLI SİMÜLASYON)
# ==========================================
@st.cache_resource
def modeli_hazirla():
    # Modelin çalışması için veri setinden küçük bir parça ile hızlıca eğitme yapıyoruz
    df = pd.read_csv('Reviews.csv').dropna(subset=['Text', 'Score']).head(5000)
    df['sentiment'] = df['Score'].apply(lambda x: 0 if x <= 2 else (1 if x == 3 else 2))
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        words = text.split()
        return " ".join([lemmatizer.lemmatize(word) for word in words if word not in stop_words])
    
    df['cleaned_text'] = df['Text'].apply(clean_text)
    
    tfidf = TfidfVectorizer(max_features=3000)
    X = tfidf.fit_transform(df['cleaned_text'])
    y = df['sentiment']
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model, tfidf, clean_text

model, tfidf, clean_text = modeli_hazirla()

# Bulanık Mantık Fonksiyonumuz
def hesapla_fuzzy_guvenilirlik(puan, kelime_sayisi, yorum_yasi_gun):
    puan_skoru = 1.0 if puan >= 4 else (0.5 if puan == 3 else 0.1)
    uzunluk_skoru = 1.0 if kelime_sayisi > 50 else (0.6 if 15 <= kelime_sayisi <= 50 else 0.2)
    tazelik_skoru = 1.0 if yorum_yasi_gun <= 30 else (0.6 if 31 <= yorum_yasi_gun <= 180 else 0.2)
    toplam_agirlik = (puan_skoru * 0.4) + (uzunluk_skoru * 0.4) + (tazelik_skoru * 0.2)
    return round(toplam_agirlik * 100, 2)

# ==========================================
# 🎨 ÖN YÜZ: KULLANICI GİRİŞ ALANLARI
# ==========================================
st.subheader("✍️ Test Etmek İçin Bir Müşteri Yorumu Girin")
user_input = st.text_area("Müşteri Yorumu (İngilizce):", "This product is absolutely amazing! The quality is great and it arrived very fast.")

col1, col2, col3 = st.columns(3)
with col1:
    puan = st.slider("Verilen Yıldız Puanı:", 1, 5, 5)
with col2:
    yorum_yasi = st.number_input("Yorumun Yaşı (Gün):", min_value=1, max_value=365, value=10)
with col3:
    kelime_sayisi = len(user_input.split())
    st.metric("Tespit Edilen Kelime Sayısı", kelime_sayisi)

# ANALİZ ET BUTONU
if st.button("🚀 Yorumu Uçtan Uca Analiz Et", type="primary"):
    if user_input.strip() == "":
        st.warning("Lütfen boş bırakmayın, bir yorum yazın.")
    else:
        # 1. NLP ile Duygu Analizi
        temiz_metin = clean_text(user_input)
        vektor = tfidf.transform([temiz_metin])
        tahmin = model.predict(vektor)[0]
        
        sonuclar = {0: "🔴 OLUMSUZ (Negative)", 1: "🟡 NÖTR (Neutral)", 2: "🟢 OLUMLU (Positive)"}
        
        # 2. Bulanık Mantık ile Güvenilirlik
        guven_skoru = hesapla_fuzzy_guvenilirlik(puan, kelime_sayisi, yorum_yasi)
        
        # Ekran Çıktıları
        st.markdown("---")
        st.subheader("📊 Analiz Sonuçları")
        
        c1, c2 = st.columns(2)
        with c1:
            st.info(f"**Yapay Zeka Duygu Tahmini:**\n\n{sonuclar[tahmin]}")
        with c2:
            if guven_skoru >= 70:
                st.success(f"**Bulanık Mantık Güvenilirlik Skoru:**\n\n%{guven_skoru} (Güvenilir Yorum)")
            elif 40 <= guven_skoru < 70:
                st.warning(f"**Bulanık Mantık Güvenilirlik Skoru:**\n\n%{guven_skoru} (Şüpheli/Orta Güvenlik)")
            else:
                st.error(f"**Bulanık Mantık Güvenilirlik Skoru:**\n\n%{guven_skoru} (Düşük Güvenilirlik / Potansiyel Sahte Yorum)")