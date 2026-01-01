# An Operational Hybrid Approach for Vehicle Re-Id Using Composite Attention Mechanisms
Vehicle Re-Identification (V-ReID) is a fundamental yet challenging task in Intelligent Transportation Systems, particularly in scenarios involving non-overlapping camera networks and Unmanned Aerial Vehicle (UAV) surveillance. Traditional Convolutional Neural Networks (CNNs) rely predominantly on linear operations for feature extraction.

Official implementation of the proposed Vehicle Re-Identification (Re-ID) framework. This repository provides the evaluation scripts and pre-trained weights to reproduce the results presented in the paper.

🚀 Key Contributions
ONN-CNN Hybrid Backbone: Utilizing Operational Neural Networks to capture complex, non-linear patterns that standard CNNs often overlook.

Tri-Stream Attention: Simultaneous processing of spatial, channel, and global self-attention to identify fine-grained details (e.g., logos, stickers, roof racks).

GeM Pooling: Implementation of trainable Generalized Mean Pooling for better feature saliency compared to traditional Global Average Pooling.

Metric Robustness: Optimization via a stabilized Hard Triplet Loss combined with Label Smoothing Cross-Entropy.

📊 Performance & Evaluation
To ensure transparency and reproducibility, we provide an automated evaluation script. This script allows users to verify the following metrics:

Retrieval Accuracy: mAP, Rank-1, and Rank-5.

Embedding Quality: Representation cosine distance and similarity of the Hardest 50 IDs

Discriminative Power: Inter-class to Intra-class distance ratios.

Pre-trained Weights
Download the weights from the following link and place them in the weights/ directory:

Download best_model.weights.h5

Dataset
The models are evaluated on the VeRi-776 dataset. Please download the dataset from here: https://www.kaggle.com/datasets/abhyudaya12/veri-vehicle-re-identification-dataset

The following command will load the pre-trained weights, perform inference on the test set, and generate all metrics and visual plots (t-SNE, Top-5 Rankings):
python evaluation.py --weights weights/best_model.weights.h5 --data_dir ./data/te
