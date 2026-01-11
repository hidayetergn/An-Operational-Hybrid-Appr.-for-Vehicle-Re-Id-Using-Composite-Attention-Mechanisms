# Dosya: utils/loader.py
import tensorflow as tf
import os
from tensorflow.keras.applications.efficientnet import preprocess_input


def load_and_preprocess_image(path, label, img_size=(224, 224)):
    """Resmi okur ve EfficientNet formatına getirir."""
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, img_size)
    image = preprocess_input(image)  # 0-255 arası değerleri normalize eder
    return image, label


def create_dataset(list_path, img_dir, batch_size=32, img_size=(224, 224)):
    """Metin listesinden tf.data.Dataset oluşturur."""
    if not os.path.exists(list_path):
        raise FileNotFoundError(f"Liste dosyası bulunamadı: {list_path}")

    image_paths = []
    labels = []

    # Listeyi ayrıştır
    with open(list_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                fname = parts[0]
                vid = int(parts[1])

                # Uzantı kontrolü
                full_path = os.path.join(img_dir, fname + ".jpg")
                if not os.path.exists(full_path):
                    full_path = os.path.join(img_dir, fname)

                image_paths.append(full_path)
                labels.append(vid)

    print(f"[INFO] Toplam {len(image_paths)} görüntü yüklenecek.")

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(lambda x, y: load_and_preprocess_image(x, y, img_size),
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset, len(labels)
