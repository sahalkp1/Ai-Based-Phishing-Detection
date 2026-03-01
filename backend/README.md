# 🛡️ Phishing Detection API — Phase 6

## Folder Structure

```
phase6_api/
│
├── main.py              ← FastAPI application (all endpoints)
├── requirements.txt     ← Python dependencies
├── README.md            ← This file
│
└── models/              ← PUT ALL YOUR MODEL FILES HERE
    ├── phishing_detector_model/   ← folder from Phase 3 zip
    │   ├── config.json
    │   ├── model.safetensors
    │   ├── tokenizer_config.json
    │   ├── tokenizer.json
    │   └── vocab.txt
    ├── url_rf_model.pkl           ← from Phase 4 zip
    ├── url_lr_model.pkl           ← from Phase 4 zip
    ├── url_scaler.pkl             ← from Phase 4 zip
    └── engine_config.pkl          ← from Phase 5 zip

```

---

## Setup & Run (Step by Step)

### Step 1 — Create the folder structure
```bash
mkdir phase6_api
cd phase6_api
mkdir models
```

### Step 2 — Copy your model files into models/
- Extract `phishing_detector_model.zip` → copy the `phishing_detector_model/` folder into `models/`
- Extract `phase5_risk_engine.zip` → copy all `.pkl` files into `models/`

### Step 3 — Install dependencies
```bash
pip3 install -r requirements.txt
```

### Step 4 — Run the API
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Step 5 — Open in browser
```
http://localhost:8000/docs
```
This opens the **automatic interactive documentation** (Swagger UI) — you can test all endpoints right there!

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET  | `/health` | Check if server is running |
| POST | `/check-text` | Analyse email text only (NLP) |
| POST | `/check-url` | Analyse URL only (Random Forest) |
| POST | `/check-combined` | Full analysis (NLP + URL combined) |

---

## Example Requests

### /check-text
```json
{
  "text": "Your account has been suspended. Click here to verify immediately."
}
```

### /check-url
```json
{
  "url": "http://paypal-secure-verify.tk/login.php"
}
```

### /check-combined
```json
{
  "text": "URGENT: Verify your account at http://paypal-verify.tk/login",
  "url": null
}
```

---

## Example Response (/check-combined)
```json
{
  "input": {
    "text": "URGENT: Verify your account...",
    "url": null,
    "urls_found": ["http://paypal-verify.tk/login"]
  },
  "scores": {
    "nlp_score": 0.9421,
    "url_score": 0.8734,
    "final_score": 0.9146,
    "formula": "(0.6 × 0.9421) + (0.4 × 0.8734) = 0.9146"
  },
  "result": {
    "is_phishing": true,
    "verdict": "PHISHING",
    "risk_level": "HIGH",
    "confidence": "91.46%"
  }
}
```
