import streamlit as st
import pandas as pd
import re
from wordcloud import WordCloud, STOPWORDS
import nltk
from textblob import TextBlob
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Mengunduh dataset stopwords NLTK dan WordNetLemmatizer untuk stemming
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Fungsi untuk membersihkan teks Twitter
def clean_twitter_text(text):
    """Membersihkan teks tweet, termasuk menghapus karakter khusus dan spasi berlebihan."""
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)  # Menghapus mentions
    text = re.sub(r'#\w+', '', text)  # Menghapus hashtags
    text = re.sub(r'RT[\s]+', '', text)  # Menghapus RT
    text = re.sub(r'https?://\S+', '', text)  # Menghapus URL
    text = re.sub(r'\s+', ' ', text)  # Menghapus spasi berlebihan
    text = text.strip()  # Menghapus spasi di awal dan akhir teks
    return text

# Kamus untuk normalisasi kata-kata tertentu
norm = {
    ' gugel ': ' google ',
    ' buku ': ' book ',
    ' kucing ': ' cat ',
    # Tambahkan penyesuaian lain sesuai kebutuhan Anda
}

# Fungsi untuk melakukan normalisasi pada teks
def normalisasi(str_text):
    for i in norm:
        str_text = str_text.replace(i, norm[i])
    return str_text

# Fungsi untuk menghapus stop words dari teks
stop_words = set(nltk.corpus.stopwords.words('indonesian'))
more_stop_words = ["tidak"]
stop_words.update(more_stop_words)

def remove_stopwords(str_text):
    words = str_text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

# Fungsi untuk stemming
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def stemming(text):
    words = nltk.word_tokenize(text)
    stemmed_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(stemmed_words)

# Judul aplikasi
st.markdown('# PEMANFAATAN *WORD CLOUD* PADA ANALISIS SENTIMEN DALAM MENGGALI PERSEPSI PUBLIK')

# Markdown untuk deskripsi
st.markdown('Studi Kasus Menggunakan *Naive Bayes* dan *TextBlob*')

# Sidebar untuk navigasi
st.sidebar.title('Analisis Sentimen')
st.sidebar.markdown("*Natural Language Processing*")
st.sidebar.subheader('Kelompok 13')
st.sidebar.write('Paulina Agusia - 535220048')
st.sidebar.write('Vincent Calista - 535220075')
st.sidebar.write('Merry Manurung - 535220263')

# Radio button untuk memilih dataset
choice = st.sidebar.radio('Sentiment Type', ('Gibran', 'Prabowo'))

# Membaca data dari file CSV berdasarkan pilihan
if choice == 'Gibran':
    raw = pd.read_csv('data_gibran.csv', index_col=0)
else:
    raw = pd.read_csv('prabowo.csv', index_col=0)

df = raw[['full_text', 'username', 'created_at']].copy()  # Memilih kolom yang diperlukan
df.drop_duplicates(subset=['full_text'], inplace=True)  # Menghapus duplikasi berdasarkan teks lengkap

# Membersihkan teks tweet dan mengubahnya menjadi huruf kecil
df['full_text'] = df['full_text'].apply(clean_twitter_text).str.lower()
df['full_text'] = df['full_text'].apply(normalisasi)
df['full_text'] = df['full_text'].apply(remove_stopwords)
df['full_text'] = df['full_text'].apply(stemming)

# Fungsi untuk prediksi sentimen menggunakan TextBlob
def predict_sentiment(text):
    try:
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity
        if polarity > 0:
            return "Positive"
        elif polarity == 0:
            return "Neutral"
        else:
            return "Negative"
    except Exception as e:
        st.error(f"Error during sentiment analysis: {str(e)}")
        return None

# Menerapkan prediksi sentimen ke setiap baris dalam DataFrame
df['textblob_sentiment'] = df['full_text'].apply(predict_sentiment)

# Membersihkan nilai-nilai yang bernilai None pada kolom 'sentiment'
df.dropna(subset=['textblob_sentiment'], inplace=True)

# Membuat word cloud berdasarkan kategori sentimen TextBlob
for sentiment in ['Positive', 'Neutral', 'Negative']:
    st.subheader(f"*Word Cloud for {sentiment} Tweets (TextBlob)*")
    sentiment_words = ' '.join(df[df['textblob_sentiment'] == sentiment]['full_text'])
    wordcloud = WordCloud(
        width=800,
        height=400,
        random_state=21,
        background_color='white',
        colormap='RdPu',
        collocations=False,
        stopwords=STOPWORDS,
    ).generate(sentiment_words)

    # Plot word cloud
    st.image(wordcloud.to_array(), caption=f"*Word Cloud for {sentiment} Tweets (TextBlob)*")

# Menyiapkan data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(df['full_text'], df['textblob_sentiment'], test_size=0.2, random_state=42)

# Membuat pipeline untuk preprocessing dan pelatihan model Naive Bayes
model_pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# Melatih model
model_pipeline.fit(X_train, y_train)

# Memprediksi sentimen pada data uji
y_pred = model_pipeline.predict(X_test)

# Menampilkan akurasi dan laporan klasifikasi
# accuracy = accuracy_score(y_test, y_pred)
# st.write(f"Accuracy: {accuracy:.2f}")
# st.write("Classification Report:")
# st.text(classification_report(y_test, y_pred))

# Menerapkan prediksi sentimen ke setiap baris dalam DataFrame menggunakan model Naive Bayes
df['naive_bayes_sentiment'] = model_pipeline.predict(df['full_text'])

# Membuat word cloud berdasarkan kategori sentimen Naive Bayes
for sentiment in ['Positive', 'Neutral', 'Negative']:
    st.subheader(f"*Word Cloud for {sentiment} Tweets (Naive Bayes)*")
    sentiment_words = ' '.join(df[df['naive_bayes_sentiment'] == sentiment]['full_text'])
    wordcloud = WordCloud(
        width=800,
        height=400,
        random_state=21,
        background_color='white',
        colormap='RdPu',
        collocations=False,
        stopwords=STOPWORDS,
    ).generate(sentiment_words)

    # Plot word cloud
    st.image(wordcloud.to_array(), caption=f"*Word Cloud for {sentiment} Tweets (Naive Bayes)*")

# Membuat tabel perbandingan sentimen antara TextBlob dan Naive Bayes
st.subheader("Perbandingan Sentimen (*TextBlob vs Naive Bayes*)")

# Menghitung jumlah tweet berdasarkan sentimen untuk TextBlob dan Naive Bayes
textblob_counts = df['textblob_sentiment'].value_counts().reindex(['Positive', 'Neutral', 'Negative'], fill_value=0)
naive_bayes_counts = df['naive_bayes_sentiment'].value_counts().reindex(['Positive', 'Neutral', 'Negative'], fill_value=0)

# Membuat DataFrame perbandingan
comparison_df = pd.DataFrame({
    'Sentiment': ['Positive', 'Neutral', 'Negative'],
    'TextBlob': textblob_counts,
    'Naive Bayes': naive_bayes_counts
})

# Menampilkan tabel perbandingan
st.write(comparison_df)

# Checkbox untuk menampilkan data mentah
if st.checkbox("*Show Raw Data*"):
    st.write(raw.head(50))

# Checkbox untuk menampilkan data yang telah dibersihkan
if st.checkbox("*Show Cleaned Data*"):
    st.write(df.head(50))

# Tokenisasi teks
tokenized = df['full_text'].apply(lambda x: x.split())

# Checkbox untuk menampilkan data yang telah di-tokenisasi
if st.checkbox("*Show Tokenized Data*"):
    st.write(tokenized.head(50))

# Membuat dataframe untuk analisis sentimen TextBlob
sentiment_counts_textblob = df['textblob_sentiment'].value_counts()
sentiment_df_textblob = pd.DataFrame({'Sentiment': sentiment_counts_textblob.index, 'Tweets': sentiment_counts_textblob.values})

# Membuat grafik batang untuk sentimen TextBlob
fig_textblob = px.bar(sentiment_df_textblob, x='Sentiment', y='Tweets', color='Tweets', height=500, title="Sentiment Analysis (TextBlob)")
st.plotly_chart(fig_textblob)

# Membuat dataframe untuk analisis sentimen Naive Bayes
sentiment_counts_naive_bayes = df['naive_bayes_sentiment'].value_counts()
sentiment_df_naive_bayes = pd.DataFrame({'Sentiment': sentiment_counts_naive_bayes.index, 'Tweets': sentiment_counts_naive_bayes.values})

# Membuat grafik batang untuk sentimen Naive Bayes
fig_naive_bayes = px.bar(sentiment_df_naive_bayes, x='Sentiment', y='Tweets', color='Tweets', height=500, title="Sentiment Analysis (Naive Bayes)")
st.plotly_chart(fig_naive_bayes)

# Menambahkan fitur input teks untuk analisis sentimen di bagian bawah halaman utama
st.subheader("Masukkan Kata/Kalimat untuk Analisis Sentimen")
user_input = st.text_input("Masukkan teks di sini...")

if user_input:
    # Membersihkan dan memproses input pengguna
    cleaned_input = clean_twitter_text(user_input).lower()
    cleaned_input = normalisasi(cleaned_input)
    cleaned_input = remove_stopwords(cleaned_input)
    cleaned_input = stemming(cleaned_input)
    
    # Prediksi sentimen menggunakan TextBlob
    textblob_sentiment = predict_sentiment(cleaned_input)
    
    # Prediksi sentimen menggunakan model Naive Bayes
    naive_bayes_sentiment = model_pipeline.predict([cleaned_input])[0]
    
    st.write(f"**Hasil Analisis Sentimen:**")
    st.write(f"**TextBlob:** {textblob_sentiment}")
    st.write(f"**Naive Bayes:** {naive_bayes_sentiment}")

# Membuat dataframe untuk analisis sentimen TextBlob
sentiment_counts_textblob = df['textblob_sentiment'].value_counts()
sentiment_df_textblob = pd.DataFrame({'Sentiment': sentiment_counts_textblob.index, 'Tweets': sentiment_counts_textblob.values})
