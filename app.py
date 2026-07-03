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

# ==========================================
# 🌐 LOCALIZATION (ÇOKLU DİL DESTEĞİ) SÖZLÜĞÜ
# ==========================================
diller = {
    "English": {
        "baslik": "📊 Customer Review NLP & Fuzzy Logic Analysis System",
        "alt_baslik": "Piton Technology - Case Study Interactive Interface",
        "giris_header": "✍️ Enter a Customer Review to Test",
        "yorum_etiket": "Customer Review (English):",
        "varsayilan_yorum": "This product is absolutely amazing! The quality is great and it arrived very fast.",
        "slider_etiket": "Given Star Rating:",
        "yas_etiket": "Age of Review (Days):",
        "kelime_etiket": "Detected Word Count",
        "buton_metni": "🚀 Analyze Review End-to-End",
        "bos_uyari": "Please do not leave blank, write a review.",
        "sonuc_header": "📊 Analysis Results",
        "ai_tahmini": "**AI Sentiment Prediction:**",
        "fuzzy_skoru": "**Fuzzy Logic Reliability Score:**",
        "durum_guvenilir": "Guvenilir Yorum (Reliable Review)",
        "durum_supheli": "Supheli/Orta Guvenlik (Suspicious/Medium Reliability)",
        "durum_sahte": "Dusun Guvenilirlik / Potansiyel Sahte Yorum (Low Reliability / Potential Fake Review)",
        "olumsuz": "🔴 NEGATIVE",
        "notr": "🟡 NEUTRAL",
        "olumlu": "🟢 POSITIVE"
    },
    "Türkçe": {
        "baslik": "📊 Müşteri Yorumu NLP & Bulanık Mantık Analiz Sistemi",
        "alt_baslik": "Piton Technology - Case Çalışması İnteraktif Arayüzü",
        "giris_header": "✍️ Test Etmek İçin Bir Müşteri Yorumu Girin",
        "yorum_etiket": "Müşteri Yorumu (İngilizce):",
        "varsayilan_yorum": "This product is absolutely amazing! The quality is great and it arrived very fast.",
        "slider_etiket": "Verilen Yıldız Puanı:",
        "yas_etiket": "Yorumun Yaşı (Gün):",
        "kelime_etiket": "Tespit Edilen Kelime Sayısı",
        "buton_metni": "🚀 Yorumu Uçtan Uca Analiz Et",
        "bos_uyari": "Lütfen boş bırakmayın, bir yorum yazın.",
        "sonuc_header": "📊 Analiz Sonuçları",
        "ai_tahmini": "**Yapay Zeka Duygu Tahmini:**",
        "fuzzy_skoru": "**Bulanık Mantık Güvenilirlik Skoru:**",
        "durum_guvenilir": "Güvenilir Yorum",
        "durum_supheli": "Şüpheli / Orta Güvenlik",
        "durum_sahte": "Düşük Güvenilirlik / Potansiyel Sahte Yorum",
        "olumsuz": "🔴 OLUMSUZ",
        "notr": "🟡 NÖTR",
        "olumlu": "🟢 OLUMLU"
    }
}

# Sol Menüde Dil Seçimi Kutusu
secilen_dil = st.sidebar.selectbox("🌐 Dil / Language:", ["Türkçe", "English"])
txt = diller[secilen_dil]

st.title(txt["baslik"])
st.write(txt["alt_baslik"])
st.markdown("---")

# ==========================================
# 🧠 ARKA PLAN: MODEL VE VERİ HAZIRLIĞI
# ==========================================
@st.cache_resource
def modeli_hazirla():
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

def hesapla_fuzzy_guvenilirlik(puan, kelime_sayisi, yorum_yasi_gun):
    puan_skoru = 1.0 if puan >= 4 else (0.5 if puan == 3 else 0.1)
    uzunluk_skoru = 1.0 if kelime_sayisi > 50 else (0.6 if 15 <= kelime_sayisi <= 50 else 0.2)
    tazelik_skoru = 1.0 if yorum_yasi_gun <= 30 else (0.6 if 31 <= yorum_yasi_gun <= 180 else 0.2)
    toplam_agirlik = (puan_skoru * 0.4) + (uzunluk_skoru * 0.4) + (tazelik_skoru * 0.2)
    return round(toplam_agirlik * 100, 2)

# ==========================================
# 🎨 ÖN YÜZ: DİNAMİK METİNLER
# ==========================================
st.subheader(txt["giris_header"])
user_input = st.text_area(txt["yorum_etiket"], txt["varsayilan_yorum"])

col1, col2, col3 = st.columns(3)
with col1:
    puan = st.slider(txt["slider_etiket"], 1, 5, 5)
with col2:
    yorum_yasi = st.number_input(txt["yas_etiket"], min_value=1, max_value=365, value=10)
with col3:
    kelime_sayisi = len(user_input.split())
    st.metric(txt["kelime_etiket"], kelime_sayisi)

if st.button(txt["buton_metni"], type="primary"):
    if user_input.strip() == "":
        st.warning(txt["bos_uyari"])
    else:
        temiz_metin = clean_text(user_input)
        vektor = tfidf.transform([temiz_metin])
        tahmin = model.predict(vektor)[0]
        
        sonuclar = {0: txt["olumsuz"], 1: txt["notr"], 2: txt["olumlu"]}
        guven_skoru = hesapla_fuzzy_guvenilirlik(puan, kelime_sayisi, yorum_yasi)
        
        st.markdown("---")
        st.subheader(txt["sonuc_header"])
        
        c1, c2 = st.columns(2)
        with c1:
            st.info(f"{txt['ai_tahmini']}\n\n{sonuclar[tahmin]}")
        with c2:
            if guven_skoru >= 70:
                st.success(f"{txt['fuzzy_skoru']}\n\n%{guven_skoru} ({txt['durum_guvenilir']})")
            elif 40 <= guven_skoru < 70:
                st.warning(f"{txt['fuzzy_skoru']}\n\n%{guven_skoru} ({txt['durum_supheli']})")
            else:
                st.error(f"{txt['fuzzy_skoru']}\n\n%{guven_skoru} ({txt['durum_sahte']})")