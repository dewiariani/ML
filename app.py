import streamlit as st
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import joblib
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datetime import datetime
import pickle
with open('models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

MAXLEN = 20 

# Load Naive Bayes Sentiment Model
sentiment_model = joblib.load("models/model_naive_bayes (5).pkl")  # Sesuaikan path kalau beda

# Load TF-IDF Vectorizer untuk Naive Bayes
vectorizer_nb = joblib.load("models/tfidf_vectorizer (4).pkl")  # SIMPAN dari Google Colab
# Load Label Encoder
label_encoder = joblib.load("models/label_encoder (5).pkl") 

# Load SVM Model & its vectorizer
model = joblib.load("models/model_klasifikasi_svm.pkl")  # your SVM classifier
vectorizer = joblib.load("models/svm_vectorizer.pkl")  # make sure you saved this during training

# -------------------------------
# Fungsi untuk Model Rekomendasi
@st.cache_resource
def load_recommendation_model():
    try:
        model = load_model('models/model_lstm.h5')
        with open('models/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        return model, tokenizer
    except Exception as e:
        st.error(f"Gagal memuat model rekomendasi: {e}")
        return None, None

recommendation_model, tokenizer = load_recommendation_model()

def process_content_recommendation(df):
    """Fungsi untuk memproses rekomendasi konten dengan median dinamis"""
    try:
        # Hitung median dinamis dari dataset yang diupload
        dynamic_median_reach = df['Reach'].median()
        dynamic_median_likes = df['Likes'].median()
        st.caption(f"📄 Data yang diunggah berisi {len(df)} baris konten.")
        st.caption(f"📊 Median Dinamis - Reach: {dynamic_median_reach:.0f}, Likes: {dynamic_median_likes:.0f}")
       
        
        # Rule-based recommendation
        df['status_rekomendasi_rule'] = df.apply(
            lambda row: 'Direkomendasikan' if row['Reach'] >= dynamic_median_reach and row['Likes'] >= dynamic_median_likes 
            else 'Kurang Direkomendasikan', axis=1
        )
        
        # ML-based recommendation
        if recommendation_model and tokenizer:
            texts = df['Label'].astype(str).tolist()
            sequences = tokenizer.texts_to_sequences(texts)
            padded = pad_sequences(sequences, padding='post', maxlen=20)
            predictions = recommendation_model.predict(padded)
            df['prediksi_score'] = predictions
            df['status_rekomendasi_ml'] = np.where(
                predictions >= 0.5, 'Direkomendasikan', 'Kurang Direkomendasikan'
            )
        
        # Tambahkan waktu pemrosesan
        df['waktu_pemrosesan'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return df, dynamic_median_reach, dynamic_median_likes
    
    except Exception as e:
        st.error(f"Error dalam memproses rekomendasi: {e}")
        return None, None, None


if "users" not in st.session_state:
    st.session_state.users = {"ariani": "123456"}


# -------------------------------
def login_page():
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in st.session_state.users and st.session_state.users[username] == password:
            st.session_state.login = True
            st.session_state.username = username
            st.session_state.page = "dashboard"
            st.rerun()
        else:
            st.error("Username atau password salah!")


# -------------------------------
def register_page():
    st.title("📝 Register")
    nama = st.text_input("Nama Lengkap")
    username = st.text_input("Username")
    pass1 = st.text_input("Password", type="password")
    pass2 = st.text_input("Ulangi Password", type="password")
    if st.button("Daftar"):
        if username in st.session_state.users:
            st.error("Username sudah digunakan")
        elif pass1 != pass2:
            st.error("Password tidak cocok")
        else:
            st.session_state.users[username] = pass1
            st.success("Berhasil daftar! Silakan login")
            st.session_state.page = "login"

# -------------------------------
def dashboard():
    st.sidebar.title("Dashboard")
    st.sidebar.write(f"👤 Login sebagai: {st.session_state.username}")
    st.sidebar.markdown("---")
    st.sidebar.button("🚪 Logout", on_click=lambda: st.session_state.clear())

    st.title("📊 Dashboard Utama")
    st.markdown(f"#### Hi, selamat datang **{st.session_state['username'].capitalize()}**! 👋")
    st.markdown("Silakan pilih fitur untuk mulai analisis:")

    col1, col2, col3 = st.columns(3)

    st.markdown("""
        <style>
        .card {
            background-color: #ffe5e5;
            padding: 20px;
            border-radius: 5px 5px 0 0;
            color: black;
            text-align: center;
            position: relative;
            margin-bottom: 0;
        }

        div.stButton > button:first-child {
            background-color: #c00;
            color: white;
            padding: 10px 20px;
            border-radius: 0 0 5px 5px;
            border: none;
        }
        </style>
    """, unsafe_allow_html=True)

    with col1:
        st.markdown("""
            <div class="card">
                <h4>Sentimen</h4>
                <p>Pahami persepsi publik terhadap brand Anda melalui analisis sentimen dari berbagai sumber.</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Masuk Sentimen", key="sentimen_button", use_container_width=True, 
                    on_click=lambda: st.session_state.update({"page": "sentimen"})):
            pass

    with col2:
        st.markdown("""
            <div class="card">
                <h4> Klasifikasi</h4>
                <p>Klasifikasikan pesan masuk ke dalam kategori seperti layanan kampus, beasiswa, pendaftaran dsb.</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Masuk Klasifikasi", key="klasifikasi_button", use_container_width=True, 
                    on_click=lambda: st.session_state.update({"page": "klasifikasi"})):
            pass

    with col3:
        st.markdown("""
            <div class="card">
                <h4> Rekomendasi </h4>
                <p>Dapatkan rekomendasi konten berdasarkan pola interaksi audiens untuk sosial media</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Masuk Rekomendasi", key="rekomendasi_button", use_container_width=True, 
                    on_click=lambda: st.session_state.update({"page": "rekomendasi"})):
            pass

# -------------------------------
def halaman_sentimen():
    st.sidebar.button("← Kembali ke Dashboard", on_click=lambda: st.session_state.update({"page": "dashboard"}))
    st.title("💬 Analisis Sentimen")

    # SESSION STATE INI DITARUH DI SINI
    if "input_text" not in st.session_state:
        st.session_state.input_text = ""

    input_text = st.text_input("Masukkan komentar untuk dianalisis", key="input_text")
    uploaded_file = st.file_uploader("Atau upload file CSV", type="csv")
    if uploaded_file is not None:
        input_text = ""

    if st.button("🔍 Periksa Hasil"):
        # Analisis 1 komentar manual
        if input_text:
            input_vector = vectorizer_nb.transform([input_text])
            pred = sentiment_model.predict(input_vector)[0]
            label = label_encoder.inverse_transform([pred])[0]
            st.success(f"Hasil prediksi: **{label}**")

        # Analisis dari file CSV
        if uploaded_file:
            df = pd.read_csv(uploaded_file, encoding='latin1')  # atau cp1252 jika perlu

            # Pastikan ada kolom 'komentar'
            if 'komentar' not in df.columns:
                st.error("❗ CSV harus memiliki kolom bernama 'komentar'")
                return

            df['komentar'] = df['komentar'].astype(str).fillna('').apply(lambda x: '' if x.strip().lower() == 'none' else x.strip())


            sentimen_hasil = []
            for teks in df['komentar']:
                if pd.isna(teks) or str(teks).strip().lower() in ['none', 'nan', '']:
                    sentimen_hasil.append("Tidak Ada")
                else:
                    vector = vectorizer_nb.transform([teks])
                    pred = sentiment_model.predict(vector)[0]
                    label = label_encoder.inverse_transform([pred])[0]
                    sentimen_hasil.append(label)

            df['Sentimen'] = sentimen_hasil
            df['komentar'] = df['komentar'].fillna('None')

            # Untuk chart dan ringkasan — hanya data valid
            filtered_df = df[df['Sentimen'] != "Tidak Ada"]
            sentiment_counts = filtered_df['Sentimen'].value_counts()
            total_data = len(filtered_df)

            st.subheader("Hasil Sentimen")
            st.dataframe(df[['komentar', 'Sentimen']])

          

            # Tampilkan Chart
            st.subheader("Distribusi Sentimen")
            sentiment_counts = df['Sentimen'].value_counts()
            total_data = len(df)

            col1, col2 = st.columns([2.5, 1.5])

            with col1:
                fig, ax = plt.subplots(figsize=(6, 4))
                colors = ['#2ecc71', '#95a5a6', '#e74c3c']
                bars = ax.bar(sentiment_counts.index, sentiment_counts.values, color=colors[:len(sentiment_counts)])
                ax.set_ylabel("Jumlah")
                ax.set_xlabel("Sentimen")
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                st.pyplot(fig)


            with col2:
                st.markdown("""
                    <div style='margin-top: -55px;'>
                        <h3>Keterangan</h3>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown(f"<div style='font-size: 16px;'><b>Total data valid:</b> {total_data}</div>", unsafe_allow_html=True)
                st.markdown("---")
                for label, count in sentiment_counts.items():
                  st.markdown(f"<div style='font-size: 15px;'>• <b>{label}:</b> {count} komentar</div>", unsafe_allow_html=True)
            # Tombol download hasil
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Hasil sebagai CSV", csv, "hasil_sentimen.csv", "text/csv")

        elif not input_text:
            st.warning("Masukkan teks atau upload file terlebih dahulu.")

# -------------------------------
def halaman_klasifikasi():
    st.sidebar.button("← Kembali ke Dashboard", on_click=lambda: st.session_state.update({"page": "dashboard"}))
    st.title("📂 Klasifikasi Pesan")

    input_text = st.text_input("Masukkan pesan untuk diklasifikasikan")
    uploaded_file = st.file_uploader("Atau upload file CSV", type="csv")
    if uploaded_file is not None:
        st.session_state.input_text = ""
        input_text = ""

    if st.button("🔍 Lihat Hasil"):
        if input_text:
            pesan_bersih = input_text.lower()
            pred = model.predict([pesan_bersih])[0]
            st.success(f"Kategori Pesan: **{pred}**")

        elif uploaded_file:
            df = pd.read_csv(uploaded_file, encoding='latin1')

            # Pastikan ada kolom 'pesan_user'
            if 'pesan_user' not in df.columns:
                st.error("❗ CSV harus punya kolom 'pesan_user'")
            else:
                df['pesan_user'] = df['pesan_user'].astype(str).fillna('').apply(
                lambda x: '' if x.strip().lower() == 'none' else x.strip()
            )
            
            kategori_hasil = []
            for pesan in df['pesan_user']:
                if pesan == "":
                    kategori_hasil.append("Tidak Ada")
                else:
                    pesan_bersih = pesan.lower()
                    pred = model.predict([pesan_bersih])[0]
                    kategori_hasil.append(pred)

            df['kategori_pesan'] = kategori_hasil

            st.subheader("Hasil Klasifikasi")
            st.dataframe(df[['pesan_user', 'kategori_pesan']])

            # Hanya data valid untuk chart
            filtered_df = df[df['kategori_pesan'] != "Tidak Ada"]
            kategori_counts = filtered_df['kategori_pesan'].value_counts()
            total_data = len(filtered_df)

            # 📊 Visualisasi + Ringkasan Kategori
            st.subheader("Distribusi Kategori Pesan")
            kategori_counts = df['kategori_pesan'].value_counts()
            total_data = len(df)

            col1, col2 = st.columns([3, 1.1])

            with col1:
                fig, ax = plt.subplots(figsize=(10, 5))  # Lebar grafik ditambah
                ax.tick_params(axis='x', rotation=30)    # Putar label biar gak numpuk
                colors = plt.cm.tab20.colors             # 🎨 Warna-warni (max 20 kategori)
                ax.bar(kategori_counts.index, kategori_counts.values, color=colors[:len(kategori_counts)])
                ax.set_ylabel("Jumlah")
                ax.set_xlabel("Kategori")
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                st.pyplot(fig)

            with col2:
                st.markdown("""
                    <div style='margin-top: -55px;'>
                        <h3>Keterangan</h3>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown(f"<div style='font-size: 16px;'><b>Total data:</b> {total_data}</div>", unsafe_allow_html=True)
                st.markdown("---")
                for label, count in kategori_counts.items():
                    st.markdown(f"<div style='font-size: 15px;'><b>{label}:</b> {count} pesan</div>", unsafe_allow_html=True)

            # 💾 Tombol Download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("💾 Download Hasil", data=csv, file_name="hasil_klasifikasi.csv", mime="text/csv")
        
        else:
            st.warning("⚠️ Masukkan teks atau upload file terlebih dahulu!")


# -------------------------------

def halaman_rekomendasi():
    st.sidebar.button("← Kembali ke Dashboard", on_click=lambda: st.session_state.update({"page": "dashboard"}))
    st.title("🏆 Rekomendasi Konten")

    uploaded_file = st.file_uploader("Upload Data Konten (.csv)", type=["csv"])

    if uploaded_file:
        if not uploaded_file.name.endswith(".csv"):
            st.error("❌ Format file harus .csv")
            return
        st.session_state["uploaded_content"] = uploaded_file.getvalue()

    periksa = st.button("🔍 Periksa Hasil")

    if periksa and "uploaded_content" in st.session_state:
        try:
            df = pd.read_csv(io.BytesIO(st.session_state["uploaded_content"]))
    

            if "Label" not in df.columns:
                st.error("❌ File harus mengandung kolom 'Reach, Likes'")
                return

            with st.spinner("⏳ Menganalisis konten..."):
                result_df, _, _ = process_content_recommendation(df)

            # st.success("✅ Analisis selesai!")

            # Top rekomendasi unik (per label)
            def get_top_unique(df, sort_column, ascending=False):
                unique_labels = set()
                top_contents = []
                sorted_df = df.sort_values(by=sort_column, ascending=ascending)
                for _, row in sorted_df.iterrows():
                    if row['Label'] not in unique_labels:
                        unique_labels.add(row['Label'])
                        top_contents.append(row)
                        if len(top_contents) == 5:
                            break
                return pd.DataFrame(top_contents)

            rule_based = result_df[result_df['status_rekomendasi_rule'] == 'Direkomendasikan']
            top_rule = get_top_unique(rule_based, ['Reach', 'Likes'])

            top_ml = pd.DataFrame()
            if 'status_rekomendasi_ml' in result_df.columns:
                ml_based = result_df[result_df['status_rekomendasi_ml'] == 'Direkomendasikan']
                top_ml = get_top_unique(ml_based, 'prediksi_score')

            # === Tampilan Rekomendasi ===
            st.markdown("### Konten Yang Direkomendasikan")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Berdasarkan AI")
                if not top_ml.empty:
                    for i, (_, row) in enumerate(top_ml.iterrows(), 1):
                        clean_label = str(row['Label']).replace("*", "").strip()
                        st.markdown(f"""
                        **{i}. {clean_label}**  
                        Skor AI: **{row['prediksi_score']*100:.0f}%**  
                        🔗 [Lihat Konten]({row['Permalink']})
                        """)
                else:
                    st.info("Tidak ada rekomendasi dari AI.")



            with col2:
                st.markdown("### Berdasarkan Aturan")
                if not top_rule.empty:
                    for i, (_, row) in enumerate(top_rule.iterrows(), 1):
                        st.markdown(f"""
                        **{i}. {row['Label']}**  
                        Reach: {row['Reach']} | Likes: {row['Likes']}  
                        🔗 [Lihat Konten]({row['Permalink']})
                        """)
                else:
                    st.info("Tidak ada rekomendasi dari aturan.")

            # === Contoh konten tidak direkomendasikan ===
            st.markdown("---")
            unrecommended = result_df[
                (result_df['status_rekomendasi_rule'] != 'Direkomendasikan') &
                (result_df.get('status_rekomendasi_ml', '') != 'Direkomendasikan')
            ]

            if not unrecommended.empty:
                st.markdown("### Konten yang Tidak Direkomendasikan")
                contoh = unrecommended.sample(min(3, len(unrecommended)))
                for _, row in contoh.iterrows():
                    st.markdown(f"- {row['Label']} | 🔗 [Lihat Konten]({row['Permalink']})")

           
# --- Grafik Baru: Top 5 Label Berdasarkan Total Likes (Konten yang Direkomendasikan) ---
          # --- Grafik Baru: Top 5 Label dari Konten yang Ditampilkan ---
             #st.markdown("---")
           # st.subheader("📊 Top 5 Label dari Konten Direkomendasi yang Ditampilkan")

            # Gabungkan top 5 dari AI dan Rule
            #combined_top = pd.concat([top_ml, top_rule], ignore_index=True)

            # if not combined_top.empty:
                # Group berdasarkan Label, hitung total Likes
              #  chart_data = combined_top.groupby('Label')['Likes'].sum().sort_values(ascending=False).head(5)

               # fig, ax = plt.subplots(figsize=(10, 5))
                #bars = ax.bar(chart_data.index, chart_data.values, color='#00BFFF')
                #ax.set_title("Top 5 Label (Berdasarkan Likes dari Konten Direkomendasi yang Ditampilkan)", fontsize=14)
                #ax.set_ylabel("Total Likes")
                #ax.set_xlabel("Label")
                #ax.tick_params(axis='x', rotation=45)

                # Tambahkan angka di atas bar
                #for bar in bars:
                 #   height = bar.get_height()
                  #  ax.annotate(f'{int(height)}',
                   #             xy=(bar.get_x() + bar.get_width() / 2, height),
                    #            xytext=(0, 5),
                     #           textcoords="offset points",
                      #          ha='center', va='bottom')#

              #  st.pyplot(fig)
            # else:
              #   st.info("📭 Tidak ada konten direkomendasi yang ditampilkan.")


            # === Data Lengkap ===
            with st.expander("🔍 Lihat Data Lengkap"):
                tampilan = result_df.copy()
                if 'Permalink' in tampilan.columns:
                    tampilan['Permalink'] = tampilan['Permalink'].apply(lambda x: f"[Link]({x})")
                st.dataframe(tampilan)

        except Exception as e:
            st.error(f"❌ Terjadi error saat memproses file: {str(e)}")
    elif periksa:
        st.warning("⚠️ Silakan upload file CSV terlebih dahulu.")


# -------------------------------
# -------------------------------
# Routing
# ------------------------------- 
if "page" not in st.session_state:
    st.session_state.page = "login"
if "login" not in st.session_state:
    st.session_state.login = False

if not st.session_state.login:
    if st.session_state.page == "login":
        login_page()
        if st.button("Belum punya akun? Daftar"):
            st.session_state.page = "register"
    elif st.session_state.page == "register":
        register_page()
        if st.button("Sudah punya akun? Login"):
            st.session_state.page = "login"
else:
    if st.session_state.page == "dashboard":
        dashboard()
    elif st.session_state.page == "sentimen":
        halaman_sentimen()
    elif st.session_state.page == "klasifikasi":
        halaman_klasifikasi()
    elif st.session_state.page == "rekomendasi":
        halaman_rekomendasi()