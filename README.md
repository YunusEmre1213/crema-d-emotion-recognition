# CREMA-D Speech Emotion Recognition

Speech emotion recognition model trained on the [CREMA-D dataset](https://www.kaggle.com/datasets/ejlok1/cremad) using PyTorch and torchaudio.

## 📌 Overview
This project implements an advanced Speech Emotion Recognition (SER) pipeline to classify six human emotions (Anger, Disgust, Fear, Happy, Neutral, Sad) from raw audio recordings. To solve common challenges in SER, it utilizes a custom Energy-Based Voice Activity Detection (VAD) algorithm to isolate emotional utterances and employs a state-of-the-art hybrid architecture combining WavLM, SE-TCN, and Temporal Attention for highly accurate classification.

## 🛠 Technologies Used
- Python
- PyTorch, torchaudio
- Pandas, NumPy
- scikit-learn (GroupKFold, classification metrics, SVM)
- Transformers / HuggingFace (WavLM Base+)
- Matplotlib, Seaborn (for XAI and confusion matrix visualizations)

## 📊 Dataset
[CREMA-D dataset](https://www.kaggle.com/datasets/ejlok1/cremad) — Crowd-sourced Emotional Multimodal Actors Dataset

## 🎯 Results
The proposed architecture (WavLM Ultimate) was rigorously evaluated using a 5-Fold Cross-Validation setup with strictly isolated speakers (Zero Speaker Leakage). It was statistically compared against a traditional Machine Learning baseline (MFCC + SVM) and an ablation model (WavLM-Only) using paired T-tests.

- **Average Accuracy:** 75.07% ± 2.29
- **Average Macro F1-Score:** 75.14% ± 2.19

## 📓 Notebook
Full code and analysis available in the [notebook](./crema-d-proje.ipynb), also viewable on [Kaggle](https://www.kaggle.com/code/yunusemresyler/crema-d-proje).
