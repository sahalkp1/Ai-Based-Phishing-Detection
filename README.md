# 🛡️ AI-Based Phishing Detection System

> A multi-modal phishing detection system combining NLP (DistilBERT) and URL feature analysis (Random Forest) with a REST API and web interface.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-blue?style=flat-square)](https://sahalkp1.github.io/Ai-Based-Phishing-Detection)
[![API](https://img.shields.io/badge/API-HuggingFace%20Spaces-yellow?style=flat-square)](https://sahal12-phishing-detection-api.hf.space/docs)
[![Models](https://img.shields.io/badge/Models-HuggingFace%20Hub-orange?style=flat-square)](https://huggingface.co/sahal12/phishing-detection-models)

---

## 📌 Problem Statement

Phishing attacks are one of the most common cybersecurity threats, tricking users into revealing sensitive information through deceptive emails and URLs. Traditional rule-based systems fail to catch sophisticated phishing attempts.

This project builds an AI-powered multi-modal detection system that:
- Analyses **email text content** using a fine-tuned DistilBERT transformer
- Analyses **URL features** using a Random Forest classifier
- **Combines both scores** using a weighted risk engine for higher accuracy

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    USER INPUT                           │
│              Email Text  +  URL                         │
└────────────────┬───────────────┬────────────────────────┘
                 │               │
                 ▼               ▼
    ┌─────────────────┐   ┌─────────────────┐
    │  NLP Module     │   │  URL Module     │
    │  DistilBERT     │   │  Random Forest  │
    │  Fine-tuned     │   │  27 Features    │
    │  18,650 emails  │   │  48,812 URLs    │
    └────────┬────────┘   └────────┬────────┘
             │                     │
             │  NLP Score (0-1)    │  URL Score (0-1)
             │                     │
             └──────────┬──────────┘
                        ▼
           ┌────────────────────────┐
           │   Risk Scoring Engine  │
           │                        │
           │  Final Score =         │
           │  (0.6 × NLP Score) +   │
           │  (0.4 × URL Score)     │
           │                        │
           │  Score > 0.7 → PHISHING│
           │  Score ≤ 0.7 → SAFE   │
           └────────────┬───────────┘
                        ▼
           ┌────────────────────────┐
           │    FastAPI Backend     │
           │  HuggingFace Spaces    │
           └────────────┬───────────┘
                        ▼
           ┌────────────────────────┐
           │    Frontend UI         │
           │    GitHub Pages        │
           └────────────────────────┘
```

---

## 📁 Project Structure

```
Ai-Based-Phishing-Detection/
│
├── data/                          # Datasets
│   ├── phishing_url_ml_ready.csv  # URL features dataset (48,812 URLs)
│   ├── phishing_url_cleaned.csv   # URL dataset with raw URLs
│   └── Phishing_Email.csv         # Email text dataset (18,650 emails)
│
├── models/                        # Saved local model files
│
├── backend/                       # FastAPI backend
│   ├── app.py                     # Main API application
│   ├── Dockerfile                 # Docker config for HuggingFace
│   ├── requirements.txt           # Python dependencies
│   └── README.md                  # Backend documentation
│
├── frontend/                      # Web UI source
│   └── index.html                 # Single page application
│
├── docs/                          # GitHub Pages deployment
│   └── index.html                 # Live frontend
│
├── notebooks/                     # Jupyter/Colab notebooks
│   ├── phase3_nlp_training.ipynb  # DistilBERT fine-tuning
│   ├── phase4_url_detection.ipynb # URL model training
│   └── phase5_risk_engine.ipynb   # Risk scoring engine
│
├── README.md                      # This file
└── requirements.txt               # Project dependencies
```

---

## 🤖 Models & Datasets

### Datasets Used

| Dataset | Source | Size | Purpose |
|---------|--------|------|---------|
| Phishing Email Detection | Kaggle (subhajournal) | 18,650 emails | NLP model training |
| Phishing URL Dataset | Kaggle | 48,812 URLs | URL model training |

### Models

| Model | Type | Accuracy | Purpose |
|-------|------|----------|---------|
| DistilBERT | Transformer (fine-tuned) | ~97-98% | Email text classification |
| Random Forest | Ensemble ML | ~96-98% | URL feature classification |
| Risk Engine | Weighted combination | — | Final phishing verdict |

---


## 🔮 Future Work

- [ ] **Browser Extension** — real-time phishing detection while browsing
- [ ] **Gmail/Outlook Integration** — scan emails automatically
- [ ] **Multilingual Support** — detect phishing in non-English emails
- [ ] **Feedback Loop** — users can report false positives to retrain model
- [ ] **Domain Age API** — integrate WHOIS data for better URL scoring
- [ ] **Explainability** — highlight which words/features triggered detection
- [ ] **Mobile App** — Android/iOS app for on-the-go scanning

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| NLP Model | DistilBERT (HuggingFace Transformers) |
| URL Model | Scikit-learn Random Forest |
| Backend | FastAPI + Uvicorn |
| Model Hosting | HuggingFace Spaces + Hub |
| Frontend | HTML + CSS + JavaScript |
| Frontend Hosting | GitHub Pages |
| Training | Google Colab (T4 GPU) |

---

## 👤 Author

**Sahal KP**
- GitHub: [@sahalkp1](https://github.com/sahalkp1)
- HuggingFace: [@sahal12](https://huggingface.co/sahal12)

---

## 📄 License

This project is for educational purposes as part of a mini project submission.
