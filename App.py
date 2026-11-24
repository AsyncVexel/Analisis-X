import streamlit as st
import pandas as pd
import plotly.express as px
import re
from transformers import pipeline

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Analisis Sentimen X (Twitter)",
    page_icon="📊",
    layout="wide"
)

# Fungsi caching agar model AI hanya dimuat sekali (efisiensi memori & kecepatan)
@st.cache_resource
def load_sentiment_model():
    # Menggunakan model IndoBERT yang sudah dilatih untuk sentimen teks Indonesia
    model_name = "w11wo/indonesian-roberta-base-sentiment-classifier"
    return pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)

# Fungsi pembersihan teks tweet (preprocessing ringan)
def clean_tweet(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Hapus URL
    text = re.sub(r'\@\w+|\#', '', text)  # Hapus mention & simbol hashtag
    text = re.sub(r'\s+', ' ', text).strip()  # Hapus spasi berlebih
    return text

# Memuat model
with st.spinner("Memuat model AI... (Hanya dilakukan sekali saat pertama kali dibuka)"):
    sentiment_pipeline = load_sentiment_model()

# --- ANTARMUKA PENGGUNA (GUI) ---
st.title("📊 Analisis Sentimen Postingan X (Twitter)")
st.markdown("Masukkan satu atau beberapa postingan X (pisahkan dengan baris baru) untuk menganalisis kecenderungan sentimennya secara otomatis menggunakan AI.")

# Sidebar untuk informasi & contoh input
with st.sidebar:
    st.header("Panduan Penggunaan")
    st.markdown("""
    1. Masukkan teks postingan pada kolom utama.
    2. Jika ingin menganalisis banyak tweet sekaligus, pisahkan tiap tweet dengan menekan **Enter** (baris baru).
    3. Klik tombol **Analisis Sentimen**.
    """)
    
    if st.button("Isi Contoh Tweet"):
        st.session_state.sample_text = (
            "Pelayanan kereta cepat hari ini sangat memuaskan, tepat waktu dan nyaman banget!\n"
            "Kecewa berat sama aplikasi ini, error terus dari pagi padahal lagi butuh cepat.\n"
            "Hari ini cuaca di Jakarta terpantau cerah berawan sejak pagi hari.\n"
            "Harga bahan pokok mulai naik lagi menjelang akhir bulan, bikin pusing.\n"
            "Mantap min, informasinya sangat membantu buat pemula seperti saya."
        )

# Area Input Teks
default_text = st.session_state.get("sample_text", "")
user_input = st.text_area(
    "Masukkan Teks Postingan X / Twitter:",
    value=default_text,
    height=180,
    placeholder="Contoh:\nProduk ini bagus banget kualitasnya!\nPengiriman lambat dan barang rusak."
)

# Tombol Eksekusi
if st.button("🚀 Analisis Sentimen", type="primary"):
    # Memisahkan input per baris dan membuang baris kosong
    raw_tweets = [line.strip() for line in user_input.split("\n") if line.strip()]
    
    if not raw_tweets:
        st.warning("Silakan masukkan minimal satu teks postingan terlebih dahulu.")
    else:
        with st.spinner("Sedang menganalisis sentimen..."):
            results = []
            
            # Pemetaan label model ke Bahasa Indonesia
            label_mapping = {
                "positive": "Positif",
                "negative": "Negatif",
                "neutral": "Netral"
            }
            
            for tweet in raw_tweets:
                cleaned = clean_tweet(tweet)
                # Jika teks habis setelah dibersihkan (misal hanya berisi link), gunakan teks asli
                target_text = cleaned if cleaned else tweet
                
                # Prediksi menggunakan model AI (maksimal 512 token)
                prediction = sentiment_pipeline(target_text[:512])[0]
                raw_label = prediction["label"].lower()
                score = prediction["score"]
                
                results.append({
                    "Postingan Asli": tweet,
                    "Teks Bersih": target_text,
                    "Sentimen": label_mapping.get(raw_label, raw_label.capitalize()),
                    "Tingkat Keyakinan (%)": round(score * 100, 2)
                })
                
            df_results = pd.DataFrame(results)

        st.divider()
        st.subheader("📈 Ringkasan Hasil Analisis")

        # Metrik Utama (KPI Cards)
        total_tweets = len(df_results)
        pos_count = len(df_results[df_results["Sentimen"] == "Positif"])
        neg_count = len(df_results[df_results["Sentimen"] == "Negatif"])
        neu_count = len(df_results[df_results["Sentimen"] == "Netral"])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Postingan", total_tweets)
        col2.metric("Positif 😊", f"{pos_count} ({pos_count/total_tweets*100:.1f}%)")
        col3.metric("Negatif 😡", f"{neg_count} ({neg_count/total_tweets*100:.1f}%)")
        col4.metric("Netral 😐", f"{neu_count} ({neu_count/total_tweets*100:.1f}%)")

        # Visualisasi Data
        st.subheader("📊 Visualisasi Distribusi Sentimen")
        chart_col1, chart_col2 = st.columns(2)

        # Palet warna konsisten
        color_map = {
            "Positif": "#2ECC71",
            "Negatif": "#E74C3C",
            "Netral": "#95A5A6"
        }

        sentiment_counts = df_results["Sentimen"].value_counts().reset_index()
        sentiment_counts.columns = ["Sentimen", "Jumlah"]

        with chart_col1:
            fig_pie = px.pie(
                sentiment_counts,
                names="Sentimen",
                values="Jumlah",
                title="Proporsi Sentimen",
                color="Sentimen",
                color_discrete_map=color_map,
                hole=0.4
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with chart_col2:
            fig_bar = px.bar(
                sentiment_counts,
                x="Sentimen",
                y="Jumlah",
                title="Jumlah Postingan per Kategori",
                color="Sentimen",
                color_discrete_map=color_map,
                text="Jumlah"
            )
            fig_bar.update_traces(textposition="outside")
            st.plotly_chart(fig_bar, use_container_width=True)

        # Tabel Detail Hasil
        st.subheader("📋 Detail Hasil per Postingan")
        st.dataframe(
            df_results[["Postingan Asli", "Sentimen", "Tingkat Keyakinan (%)"]],
            use_container_width=True,
            hide_index=True
        )

        # Fitur Unduh Hasil ke CSV
        csv_data = df_results.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Unduh Hasil Analisis (CSV)",
            data=csv_data,
            file_name="hasil_sentimen_x.csv",
            mime="text/csv"
        )
