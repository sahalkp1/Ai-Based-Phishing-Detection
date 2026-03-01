"""
🛡️ Phishing Detection API — HuggingFace Spaces Deployment
FastAPI backend combining NLP + URL models

Endpoints:
  GET  /health
  POST /check-text
  POST /check-url
  POST /check-combined
"""

import re
import math
import pickle
import warnings
from urllib.parse import urlparse

import torch
import torch.nn.functional as F
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from huggingface_hub import hf_hub_download
import os

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="🛡️ Phishing Detection API",
    description="Multi-modal phishing detection using NLP + URL analysis",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────
# Load Models
# ─────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Running on: {device}")

# Your HuggingFace model repo name — CHANGE THIS to your HF username/repo
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "YOUR_HF_USERNAME/phishing-detection-models")

print("Loading NLP model from HuggingFace Hub...")
nlp_tokenizer = DistilBertTokenizer.from_pretrained(HF_MODEL_REPO)
nlp_model     = DistilBertForSequenceClassification.from_pretrained(HF_MODEL_REPO)
nlp_model.to(device)
nlp_model.eval()
print("✅ NLP model loaded!")

print("Loading URL models from HuggingFace Hub...")
rf_path     = hf_hub_download(repo_id=HF_MODEL_REPO, filename="url_rf_model.pkl")
scaler_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename="url_scaler.pkl")
config_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename="engine_config.pkl")

with open(rf_path, "rb") as f:
    url_model = pickle.load(f)
with open(scaler_path, "rb") as f:
    url_scaler = pickle.load(f)
with open(config_path, "rb") as f:
    engine_config = pickle.load(f)

print("✅ URL models loaded!")

NLP_WEIGHT = engine_config.get("nlp_weight", 0.6)
URL_WEIGHT = engine_config.get("url_weight", 0.4)
THRESHOLD  = engine_config.get("threshold",  0.7)

URL_FEATURE_NAMES = [
    "url_length", "num_dots", "num_hyphens", "num_underscores",
    "num_slashes", "num_question_marks", "num_equal_signs", "num_at_signs",
    "num_ampersands", "num_percent", "num_digits", "digit_ratio",
    "num_suspicious_words", "has_ip_address", "is_https", "domain_length",
    "num_subdomains", "has_suspicious_tld", "path_length", "path_depth",
    "has_exe_extension", "has_php", "query_length", "num_query_params",
    "has_non_standard_port", "domain_entropy", "tld_encoded"
]

print("✅ All models ready!\n")

# ─────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────
class TextRequest(BaseModel):
    text: str
    class Config:
        json_schema_extra = {"example": {"text": "Your account has been suspended. Verify immediately."}}

class URLRequest(BaseModel):
    url: str
    class Config:
        json_schema_extra = {"example": {"url": "http://paypal-secure-verify.tk/login.php"}}

class CombinedRequest(BaseModel):
    text: str
    url: str = None
    class Config:
        json_schema_extra = {"example": {
            "text": "URGENT: Verify your PayPal account at http://paypal-verify.tk/login",
            "url": None
        }}

# ─────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────
def get_nlp_score(text: str) -> float:
    inputs = nlp_tokenizer(
        text, return_tensors="pt", truncation=True,
        max_length=256, padding=True
    ).to(device)
    with torch.no_grad():
        outputs = nlp_model(**inputs)
    probs = F.softmax(outputs.logits, dim=-1)
    return round(probs[0][1].item(), 4)


def extract_url_features(url: str) -> dict:
    f = {}
    f["url_length"]            = len(url)
    f["num_dots"]              = url.count(".")
    f["num_hyphens"]           = url.count("-")
    f["num_underscores"]       = url.count("_")
    f["num_slashes"]           = url.count("/")
    f["num_question_marks"]    = url.count("?")
    f["num_equal_signs"]       = url.count("=")
    f["num_at_signs"]          = url.count("@")
    f["num_ampersands"]        = url.count("&")
    f["num_percent"]           = url.count("%")
    f["num_digits"]            = sum(c.isdigit() for c in url)
    f["digit_ratio"]           = f["num_digits"] / len(url) if len(url) > 0 else 0
    suspicious_words = ["login","signin","verify","secure","account",
                        "update","banking","confirm","password","pay",
                        "free","lucky","win","bonus","click"]
    f["num_suspicious_words"]  = sum(w in url.lower() for w in suspicious_words)
    f["has_ip_address"]        = int(bool(re.search(r"(https?://)?(\\d{1,3}\\.){3}\\d{1,3}", url)))
    try:
        parsed = urlparse(url if url.startswith("http") else "http://" + url)
    except Exception:
        parsed = urlparse("")
    f["is_https"]              = int(parsed.scheme == "https")
    netloc                     = parsed.netloc or ""
    domain                     = netloc.split(":")[0]
    f["domain_length"]         = len(domain)
    f["num_subdomains"]        = max(0, len(domain.split(".")) - 2)
    parts                      = domain.split(".")
    tld                        = parts[-1].lower() if len(parts) >= 2 else ""
    suspicious_tlds            = ["ru","xyz","tk","ml","ga","cf","gq","top",
                                   "work","click","link","info","biz","online"]
    f["has_suspicious_tld"]    = int(tld in suspicious_tlds)
    path                       = parsed.path or ""
    f["path_length"]           = len(path)
    f["path_depth"]            = path.count("/")
    f["has_exe_extension"]     = int(bool(re.search(r"\.(exe|sh|bat|cmd|msi|ps1|vbs|js)$", path.lower())))
    f["has_php"]               = int(".php" in path.lower())
    query                      = parsed.query or ""
    f["query_length"]          = len(query)
    f["num_query_params"]      = query.count("&") + 1 if query else 0
    port                       = parsed.port
    f["has_non_standard_port"] = int(port is not None and port not in [80, 443])
    if domain:
        probs = [domain.count(c) / len(domain) for c in set(domain)]
        f["domain_entropy"]    = -sum(p * math.log2(p) for p in probs if p > 0)
    else:
        f["domain_entropy"]    = 0
    f["tld_encoded"]           = hash(tld) % 100
    return f


def get_url_score(url: str) -> float:
    features = extract_url_features(url)
    X = pd.DataFrame([features])[URL_FEATURE_NAMES]
    proba = url_model.predict_proba(X)[0]
    return round(float(proba[1]), 4)


def extract_urls_from_text(text: str) -> list:
    return re.findall(r"https?://[^\s<>\"{}|\\^`\[\]]+", text)


def get_risk_level(score: float) -> str:
    if score > 0.85:   return "HIGH"
    elif score > 0.7:  return "MEDIUM-HIGH"
    elif score > 0.4:  return "MEDIUM"
    else:              return "LOW"

# ─────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "device": str(device),
        "models_loaded": True,
        "nlp_weight": NLP_WEIGHT,
        "url_weight": URL_WEIGHT,
        "threshold": THRESHOLD
    }


@app.post("/check-text")
def check_text(req: TextRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    nlp_score   = get_nlp_score(req.text)
    is_phishing = nlp_score > THRESHOLD
    return {
        "input":       {"text": req.text[:200] + "..." if len(req.text) > 200 else req.text},
        "nlp_score":   nlp_score,
        "is_phishing": is_phishing,
        "verdict":     "PHISHING" if is_phishing else "SAFE",
        "risk_level":  get_risk_level(nlp_score),
        "confidence":  f"{nlp_score * 100:.2f}%"
    }


@app.post("/check-url")
def check_url(req: URLRequest):
    if not req.url.strip():
        raise HTTPException(status_code=400, detail="URL cannot be empty")
    url_score   = get_url_score(req.url)
    features    = extract_url_features(req.url)
    is_phishing = url_score > THRESHOLD
    signals = []
    if features["has_ip_address"]:        signals.append("IP address in URL")
    if not features["is_https"]:          signals.append("No HTTPS")
    if features["has_suspicious_tld"]:    signals.append("Suspicious TLD")
    if features["num_at_signs"] > 0:      signals.append("@ symbol present")
    if features["has_exe_extension"]:     signals.append("Executable file extension")
    if features["has_php"]:               signals.append(".php in path")
    if features["num_suspicious_words"]:  signals.append("Suspicious keywords found")
    if features["has_non_standard_port"]: signals.append("Non-standard port")
    return {
        "input":              {"url": req.url},
        "url_score":          url_score,
        "is_phishing":        is_phishing,
        "verdict":            "PHISHING" if is_phishing else "SAFE",
        "risk_level":         get_risk_level(url_score),
        "confidence":         f"{url_score * 100:.2f}%",
        "suspicious_signals": signals
    }


@app.post("/check-combined")
def check_combined(req: CombinedRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    nlp_score  = get_nlp_score(req.text)
    urls_found = [req.url] if req.url else extract_urls_from_text(req.text)
    if urls_found:
        url_scores  = [get_url_score(u) for u in urls_found]
        url_score   = round(sum(url_scores) / len(url_scores), 4)
        final_score = round((NLP_WEIGHT * nlp_score) + (URL_WEIGHT * url_score), 4)
        formula     = f"({NLP_WEIGHT} × {nlp_score}) + ({URL_WEIGHT} × {url_score}) = {final_score}"
    else:
        url_score   = None
        final_score = nlp_score
        formula     = f"NLP only (no URL found) = {final_score}"
    is_phishing = final_score > THRESHOLD
    return {
        "input":  {"text": req.text[:200] + "..." if len(req.text) > 200 else req.text,
                   "url": req.url, "urls_found": urls_found},
        "scores": {"nlp_score": nlp_score, "url_score": url_score,
                   "final_score": final_score, "formula": formula},
        "result": {"is_phishing": is_phishing,
                   "verdict":     "PHISHING" if is_phishing else "SAFE",
                   "risk_level":  get_risk_level(final_score),
                   "confidence":  f"{final_score * 100:.2f}%"}
    }
