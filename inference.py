import sys
import os

# --- BU SATIRLARI EN ÜSTE EKLE ---
# Mevcut dosyanın (inference.py) bulunduğu klasörü Python yoluna ekle
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import tensorflow as tf
import numpy as np
import yaml
import os
import sys
import time
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm

# Kendi yazdığımız loader modülü
from utils.loader import create_dataset

# GPU Bellek Ayarı (Olası hataları önlemek için)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def load_config(config_path="config.yaml"):
    # Config dosyasını güvenli yükleme
    if not os.path.exists(config_path):
        print(f"[HATA] Config dosyası bulunamadı: {config_path}")
        sys.exit(1)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_metrics(features, labels):
    """
    Re-ID performans metriklerini hesaplar (mAP, Rank-1, Rank-5).
    """
    print("[INFO] Metrikler hesaplanıyor (Cosine Similarity)...")

    # Cosine Distance Matrisi (0 ile 2 arasındadır, 0 tam benzerliktir)
    dist_mat = cosine_distances(features, features)

    # --- KRİTİK DÜZELTME ---
    # np.inf (sonsuz) sklearn'de hata verir.
    # Bunun yerine kendisiyle olan mesafeyi maksimum değer olan 2.0 yapıyoruz.
    np.fill_diagonal(dist_mat, 2.0)

    num_samples = len(labels)
    ranks = np.zeros(10)
    aps = []

    for i in range(num_samples):
        query_pid = labels[i]
        dists = dist_mat[i]

        # Doğru eşleşmelerin (Ground Truth) indeksleri
        pos_indices = np.where(labels == query_pid)[0]
        # Kendisini çıkar (Listedeki i. eleman sorgunun kendisidir)
        pos_indices = pos_indices[pos_indices != i]

        if len(pos_indices) == 0:
            continue

        y_true = np.zeros(num_samples)
        y_true[pos_indices] = 1

        # Skor: Mesafe 0'a ne kadar yakınsa, benzerlik o kadar yüksektir.
        # sklearn skorun artan yönde iyi olmasını bekler, bu yüzden negatifi alınır.
        y_score = -dists

        # mAP (Bu sefer hata vermeyecek)
        ap = average_precision_score(y_true, y_score)
        aps.append(ap)

        # Rank-k
        sorted_indices = np.argsort(dists)
        for r in range(10):
            if sorted_indices[r] in pos_indices:
                ranks[r:] += 1
                break

    mAP = np.mean(aps)
    cmc = ranks / len(aps)

    return mAP, cmc


def main():
    # 1. Ayarları Yükle
    cfg = load_config()

    print("=" * 60)
    print(f"   OPERATIONAL HYBRID NETWORK: INFERENCE ENGINE")
    print("=" * 60)

    # 2. SavedModel'i Yükle (KODSUZ YÜKLEME)
    # Config dosyasındaki MODEL_DIR genellikle 'saved_model' olmalı
    model_path = cfg['MODEL']['MODEL_DIR']

    if not os.path.exists(model_path):
        print(f"[HATA] Model klasörü bulunamadı: {model_path}")
        print("Lütfen 'saved_model' klasörünün dizinde olduğundan emin olun.")
        sys.exit(1)

    print(f"[INFO] Model yükleniyor: {model_path} ...")
    start_time = time.time()
    try:
        # Kod gizleme başarısı burada: Sınıf tanımları olmadan yükleme
        loaded_model = tf.saved_model.load(model_path)
        infer = loaded_model.signatures["serving_default"]
        print(f"[BAŞARILI] Model yüklendi ({time.time() - start_time:.2f}s).")
    except Exception as e:
        print(f"[KRİTİK HATA] Model yüklenemedi: {e}")
        sys.exit(1)

    # 3. Veri Setini Hazırla
    print("[INFO] Veri seti hazırlanıyor...")
    try:
        ds, num_samples = create_dataset(
            list_path=cfg['DATASET']['PROCESSED_LIST'],
            img_dir=cfg['DATASET']['IMAGE_DIR'],
            batch_size=cfg['TEST']['BATCH_SIZE'],
            img_size=(cfg['DATASET']['HEIGHT'], cfg['DATASET']['WIDTH'])
        )
    except Exception as e:
        print(f"[HATA] Veri yükleyici hatası: {e}")
        print("Lütfen config.yaml içindeki dosya yollarını kontrol edin.")
        sys.exit(1)

    # 4. Özellik Çıkarımı (Inference Loop)
    print(f"[INFO] Özellik çıkarımı başladı ({num_samples} görüntü)...")
    all_feats = []
    all_labels = []

    # TQDM ilerleme çubuğu ile döngü
    for img_batch, label_batch in tqdm(ds, desc="Processing"):
        # Model Tahmini
        outputs = infer(img_batch)

        # SavedModel çıktı sözlüğünden embedding'i al
        # Genellikle ilk anahtar doğru olandır
        key = list(outputs.keys())[0]
        batch_emb = outputs[key]

        # L2 Normalizasyon
        batch_emb = tf.math.l2_normalize(batch_emb, axis=1)

        all_feats.append(batch_emb.numpy())
        all_labels.extend(label_batch.numpy())

    all_feats = np.vstack(all_feats)
    all_labels = np.array(all_labels)

    print(f"[INFO] Tamamlandı. Özellik Boyutu: {all_feats.shape}")

    # 5. Sonuçları Hesapla ve Bas
    mAP, cmc = compute_metrics(all_feats, all_labels)

    print("\n" + "=" * 60)
    print(f"🧪  TEST SONUÇLARI ({cfg['DATASET']['NAME']})")
    print("=" * 60)
    print(f"🎯 mAP:       {mAP:.2%}")
    print(f"🥇 Rank-1:    {cmc[0]:.2%}")
    print(f"🥈 Rank-5:    {cmc[4]:.2%}")
    print(f"🥉 Rank-10:   {cmc[9]:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()