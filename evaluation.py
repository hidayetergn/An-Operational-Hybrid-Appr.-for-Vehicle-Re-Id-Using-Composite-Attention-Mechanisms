import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances


# 1. MODELİ YÜKLE (Senin model fonksiyonlarını ve ağırlıklarını kullanacak)
def run_full_evaluation(model, query_ds, gallery_ds):
    print("🚀 Başlatılıyor: Kapsamlı Değerlendirme...")

    # Feature Extraction (Öznitelik Çıkarımı)
    q_feat, q_lbl = get_feats(query_ds)
    g_feat, g_lbl = get_feats(gallery_ds)

    # --- METRİKLER ---
    # 1. mAP ve CMC (Rank-1, Rank-5)
    mAP, cmc = compute_standard_metrics(q_feat, q_lbl, g_feat, g_lbl)

    # 2. Silhouette Score (Kümeleme Kalitesi)
    all_feats = np.vstack([q_feat, g_feat])
    all_labels = np.hstack([q_lbl, g_lbl])
    sil_score = silhouette_score(all_feats, all_labels)

    # 3. Inter-Class / Intra-Class Ratio
    # (Aynı sınıflar arası ortalama mesafe / Farklı sınıflar arası ortalama mesafe)
    ratio = calculate_inter_intra_ratio(all_feats, all_labels)

    # --- GÖRSELLEŞTİRME ---
    # 4. t-SNE Çizimi
    plot_tsne(all_feats, all_labels, save_path="tsne_result.png")

    # 5. Top-5 Rank Görselleştirme (Rastgele 5 Query için)
    visualize_top5_results(q_feat, q_lbl, g_feat, g_lbl, query_ds, gallery_ds)

    print(f"\n📊 SONUÇLAR:")
    print(f"mAP: {mAP:.2%}")
    print(f"Rank-1: {cmc[0]:.2%}")
    print(f"Silhouette Score: {sil_score:.4f}")
    print(f"Inter/Intra Ratio: {ratio:.4f}")