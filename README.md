# Amazon Customer Reviews Intelligence Platform

An end-to-end NLP pipeline that transforms Amazon product reviews into actionable insights through **Sentiment Classification**, **Product Clustering**, and **AI-generated buying guides**.

---

## Project Overview

This project analyzes **34,626 Amazon customer reviews** to automatically extract customer insights using Natural Language Processing (NLP).

The project consists of three main components:

- 😊 Sentiment Classification
- 📦 Product Category Clustering
- 🤖 AI-generated Category Summaries

The trained models and generated assets were integrated into a dedicated Streamlit application, maintained in a separate deployment repository.

---

## Dataset

- **Source:** Amazon Product Reviews (Kaggle)
- **Original dataset:** 34,660 reviews
- **Final cleaned dataset:** 34,626 reviews

Sentiment labels were created from product ratings:

| Rating | Sentiment |
|---------|-----------|
| 1–2 ⭐ | Negative |
| 3 ⭐ | Neutral |
| 4–5 ⭐ | Positive |

---

## Project Pipeline

```text
Raw Reviews
     │
     ▼
Data Cleaning & EDA
     │
     ▼
Sentiment Classification
     │
     ▼
Product Clustering
     │
     ▼
Generative AI Summaries
```

---

## Models

### 😊 Sentiment Classification

Several models were evaluated throughout the project.

| Model | Accuracy | Macro F1 |
|-----------------------------|---------:|---------:|
| Dummy Classifier | 93.3% | 0.322 |
| TF-IDF + Logistic Regression | 89.0% | 0.562 |
| TF-IDF + Tuned Linear SVM | 93.2% | 0.577 |
| **DistilBERT** | **95.2%** | **0.704** ✅ |

The project followed a progressive modeling approach:

- **Dummy Classifier** established the baseline.
- **TF-IDF + Logistic Regression** provided the first meaningful benchmark.
- **TF-IDF + Linear SVM** improved performance while remaining lightweight and highly interpretable.
- **DistilBERT** achieved the best overall performance after fine-tuning, significantly improving the Macro F1 score, particularly for the minority classes. (Needs GPU tu run as is considerably a hevier model)

---

### 📦 Product Clustering

Products were grouped according to their category information using an unsupervised learning pipeline based on:

- TF-IDF Vectorization
- Truncated SVD
- K-Means Clustering

The final solution produced four interpretable product categories:

- 📱 Fire Tablets
- 📚 Kindle E-Readers
- 🏠 Echo, Fire TV & Smart Home
- 🔌 Accessories & Cables

These clusters provide the foundation for the recommendation and generative AI stages.

---

### 🤖 Generative AI

Category buying guides were generated using **Google FLAN-T5 Small**.

Each prompt combines:

- Top-rated products
- Lowest-rated products
- Representative positive reviews
- Representative negative reviews

The generated summaries are exported as Markdown files and JSON assets for downstream applications.

---

## Repository Structure

```text
.
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── distilbert/
│   ├── sentiment_svm/
│   └── kmeans/
│
├── Classification/
├── Clustering/
├── Generative/
│
├── functions.py
└── README.md
```

---

## Results

- ✅ 34,626 cleaned customer reviews
- ✅ Four sentiment classification models evaluated
- ✅ Fine-tuned DistilBERT achieving **95.2% accuracy** and a **0.704 Macro F1** score
- ✅ Four interpretable product clusters
- ✅ AI-generated buying guides for every product category

---

## Deployment

The models and generated assets developed in this repository were integrated into a separate Streamlit application.

**🚀 Live Demo**

<https://reviews-dashboard-ironhackaiproject.streamlit.app>

**💻 Deployment Repository**

<https://github.com/Martigol2/reviews-dashboard>

---

## Authors

**Felipe Martignon**  
GitHub: <https://github.com/Martigol2>

**Casilda García**  
GitHub: <https://github.com/Casildagsf>

Developed as the final project for the **Ironhack AI Engineering Bootcamp**.