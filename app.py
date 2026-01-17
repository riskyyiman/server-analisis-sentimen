import os
import re
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from google_play_scraper import Sort, reviews
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
CORS(app)

# --- KONFIGURASI PATH DINAMIS ---
# Mendapatkan jalur folder tempat app.py ini berada
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan jalur lengkap ke file aset
MODEL_PATH = os.path.join(BASE_DIR, 'best_model_sentiment_gojek.h5')
TOKENIZER_PATH = os.path.join(BASE_DIR, 'tokenizer.pkl')
LE_PATH = os.path.join(BASE_DIR, 'label_encoder.pkl')

# --- MEMUAT ASET MODEL ---
# Memastikan file ada sebelum dimuat untuk menghindari error saat startup
if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH) and os.path.exists(LE_PATH):
    model = load_model(MODEL_PATH)
    tokenizer = joblib.load(TOKENIZER_PATH)
    le = joblib.load(LE_PATH)
    print("✅ Berhasil: Model dan aset AI telah dimuat dari jalur absolut.")
else:
    print("❌ Error: Salah satu file (model, tokenizer, atau label_encoder) tidak ditemukan di folder server.")

def clean_text(text):
    """Membersihkan teks ulasan sebelum prediksi"""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@[^s]+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_app_id(url):
    """Mengekstrak App ID dari URL Google Play Store"""
    match = re.search(r'id=([a-zA-Z0-9._]+)', url)
    return match.group(1) if match else None

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    app_url = data.get('url')
    target_count = int(data.get('count', 60))

    app_id = extract_app_id(app_url)
    if not app_id:
        return jsonify({'error': 'URL Play Store tidak valid'}), 400

    try:
        # Ambil data 3x lipat lebih banyak untuk mencari ulasan Netral/Negatif yang langka
        fetch_count = min(target_count * 3, 500) 
        
        result, _ = reviews(
            app_id, lang='id', country='id',
            sort=Sort.NEWEST, count=fetch_count
        )
        
        if not result:
            return jsonify({'error': 'Tidak ada ulasan ditemukan'}), 404

        df = pd.DataFrame(result)
        df['cleaned'] = df['content'].apply(clean_text)

        # Prediksi Massal
        sequences = tokenizer.texts_to_sequences(df['cleaned'])
        padded = pad_sequences(sequences, maxlen=100)
        predictions = model.predict(padded, verbose=0)
        df['sentiment'] = le.inverse_transform(np.argmax(predictions, axis=-1))

        # --- LOGIKA PENYEIMBANGAN ---
        # Target per kategori (sepertiga dari target total)
        per_class_limit = target_count // 3
        
        df_pos = df[df['sentiment'] == 'Positive'].head(per_class_limit)
        df_neu = df[df['sentiment'] == 'Neutral'].head(per_class_limit)
        df_neg = df[df['sentiment'] == 'Negative'].head(per_class_limit)

        # Gabungkan dan acak
        df_balanced = pd.concat([df_pos, df_neu, df_neg])

        # Jika total masih kurang, isi dengan sisa data yang ada sampai mencapai target_count
        if len(df_balanced) < target_count:
            remaining_needed = target_count - len(df_balanced)
            used_indices = df_balanced.index
            df_extra = df[~df.index.isin(used_indices)].head(remaining_needed)
            df_balanced = pd.concat([df_balanced, df_extra])

        df_balanced = df_balanced.sample(frac=1).reset_index(drop=True)
        
        # Summary dihitung dari data yang diseimbangkan untuk tampilan dashboard
        summary = df_balanced['sentiment'].value_counts().to_dict()
        examples = df_balanced[['content', 'sentiment']].to_dict(orient='records')

        return jsonify({
            'app_id': app_id,
            'total_scraped': len(df),
            'summary': summary,
            'examples': examples
        })

    except Exception as e:
        return jsonify({'error': f"Gagal memproses data: {str(e)}"}), 500
    
if __name__ == "__main__":
    # Mengambil port dari environment variable Render, default ke 5000 jika test lokal
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)