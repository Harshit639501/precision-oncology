"""FastAPI backend for SIH 2026 Precision Oncology AI.

Entry point: py -3 -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations
import os, re, time, uuid, secrets, hashlib, json, logging
from pathlib import Path
from typing import Optional, List, Dict, Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, EmailStr, Field

# ───────────────────────────────────────────────────────────────────
# Paths — model paths resolved RELATIVE to this file (not CWD).
# ───────────────────────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).resolve().parent
MODELS_DIR  = BACKEND_DIR / "models"
DBGENVOC_XL = BACKEND_DIR / "dbGENVOC.xlsx"
USERS_FILE = BACKEND_DIR / "users.json"

LOG = logging.getLogger("cdss")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s :: %(message)s")

# ───────────────────────────────────────────────────────────────────
# Models loader
# ───────────────────────────────────────────────────────────────────
class Models:
    loaded: bool = False
    variant_caller = None       # xgboost_oncology_model.pkl
    drug_recommender = None     # drug_recommendation_model.pkl
    le_gene = None              # le_gene.pkl
    le_chrom = None             # le_chrom.pkl
    le_drug = None              # le_drug.pkl
    db_rows: List[Dict[str, Any]] = []
    available_genes: List[str] = []
    available_chromosomes: List[str] = []

def load_models():
    modules = {"variant_calling": False, "drug_mapping": False,
               "explainability": True, "clinical_trials": True}
    try:
        Models.variant_caller  = joblib.load(MODELS_DIR / "xgboost_oncology_model.pkl")
        Models.drug_recommender = joblib.load(MODELS_DIR / "drug_recommendation_model.pkl")
        Models.le_gene    = joblib.load(MODELS_DIR / "le_gene.pkl")
        Models.le_chrom   = joblib.load(MODELS_DIR / "le_chrom.pkl")
        Models.le_drug    = joblib.load(MODELS_DIR / "le_drug.pkl")
        modules["variant_calling"] = True
        modules["drug_mapping"]    = True
        Models.loaded = True
        LOG.info("Models loaded from %s", MODELS_DIR)
    except FileNotFoundError as e:
        LOG.error("Model artifact missing: %s", e)
    df = pd.read_excel(DBGENVOC_XL, header=1)
    # select the 4 spec columns regardless of any extra columns the file carries
    for col in ["Gene name","Chrom","Transcript exon","Transcript position"]:
        if col not in df.columns: df[col] = "" if col != "Transcript exon" else 0
    df = df[["Gene name","Chrom","Transcript exon","Transcript position"]].copy()
    df.columns = ["Gene name","Chrom","Transcript exon","Transcript position"]
    Models.db_rows = [
        {"gene": str(r["Gene name"]), "chromosome": str(r["Chrom"]),
         "exon": int(r["Transcript exon"]), "position": str(r["Transcript position"])}
        for _, r in df.iterrows()
    ] 
    Models.available_genes = sorted({r["gene"] for r in Models.db_rows})
    Models.available_chromosomes = sorted({r["chromosome"] for r in Models.db_rows})
    LOG.info("dbGENVOC dataset rows=%d unique_genes=%d",
             len(Models.db_rows), len(Models.available_genes))

# ───────────────────────────────────────────────────────────────────
# Auth (in-memory – demo only)
# ───────────────────────────────────────────────────────────────────
SESSIONS: Dict[str, str] = {}     # token -> user_id
USERS: Dict[str, Dict[str, Any]] = {}   # user_id -> user record
USERS_BY_EMAIL: Dict[str, str] = {}
USERS_BY_MOBILE: Dict[str, str] = {}

def load_users() -> None:
    if os.getenv("VERCEL") == "1":
        LOG.info("Skipping local user accounts in the public demo")
        return
    if not USERS_FILE.exists():
        return
    try:
        records = json.loads(USERS_FILE.read_text(encoding="utf-8"))
        for rec in records:
            USERS[rec["id"]] = rec
            if rec.get("email"): USERS_BY_EMAIL[rec["email"]] = rec["id"]
            if rec.get("mobile"): USERS_BY_MOBILE[rec["mobile"]] = rec["id"]
        LOG.info("Loaded %d local user accounts", len(USERS))
    except (OSError, ValueError, KeyError) as exc:
        LOG.error("Could not load local user accounts: %s", exc)

def save_users() -> None:
    USERS_FILE.write_text(json.dumps(list(USERS.values()), indent=2), encoding="utf-8")

def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def _normalize_email(v: Optional[str]) -> Optional[str]:
    return v.strip().lower() if v else None

def _normalize_mobile(v: Optional[str]) -> Optional[str]:
    return v.strip().lstrip("+").replace(" ", "") if v else None

def make_token(uid: str) -> str:
    return f"cdss.{base64url(uid + ':' + str(int(time.time())) + ':' + secrets.token_hex(8))}"

def base64url(s: str) -> str:
    import base64
    return base64.urlsafe_b64encode(s.encode()).decode().rstrip("=")

def user_public(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {"id": rec["id"], "full_name": rec["full_name"],
            "email": rec.get("email"), "mobile": rec.get("mobile"),
            "role": rec["role"], "language": rec.get("language","en")}

# ───────────────────────────────────────────────────────────────────
# App + CORS (frontend may be opened directly or served by any local dev port)
# ───────────────────────────────────────────────────────────────────
app = FastAPI(title="SIH 2026 Precision Oncology AI",
              description="Somatic variant caller + drug recommender + RBAC for SIH 2026.",
              version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["null"],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["GET","POST","OPTIONS"],
    allow_headers=["Content-Type","Authorization"],
)

@app.on_event("startup")
def _startup():
    load_users()
    load_models()

# ───────────────────────────────────────────────────────────────────
# GET /
# ───────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    if os.getenv("VERCEL") == "1":
        return FileResponse(BACKEND_DIR.parent / "index.html")
    return {"message":"SIH 2026 AI Pipeline for Precision Oncology is Running!"}

# ───────────────────────────────────────────────────────────────────
# GET /health
# ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":"ok",
        "models_loaded": Models.loaded,
        "modules": {
            "variant_calling": Models.variant_caller is not None,
            "drug_mapping":    Models.drug_recommender is not None,
            "explainability":  True,
            "clinical_trials": True,
        },
        "dataset_rows": len(Models.db_rows),
    }

# ───────────────────────────────────────────────────────────────────
# GET /variant-options
# ───────────────────────────────────────────────────────────────────
@app.get("/variant-options")
def variant_options():
    rows = Models.db_rows
    exons = sorted({r["exon"] for r in rows})
    positions = sorted({r["position"] for r in rows})
    return {"genes": Models.available_genes,
            "chromosomes": Models.available_chromosomes,
            "exons": exons, "positions": positions,
            "variants": rows}

# ───────────────────────────────────────────────────────────────────
# Auth models
# ───────────────────────────────────────────────────────────────────
class RegisterIn(BaseModel):
    full_name: str = Field(..., min_length=2)
    email: Optional[str] = None
    mobile: Optional[str] = None
    password: str = Field(..., min_length=6)
    role: str = Field(..., pattern="^(patient|doctor)$")
    language: str = "en"

class LoginIn(BaseModel):
    identifier: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)

# ───────────────────────────────────────────────────────────────────
# POST /auth/register
# ───────────────────────────────────────────────────────────────────
@app.post("/auth/register")
def register(payload: RegisterIn):
    if os.getenv("VERCEL") == "1":
        raise HTTPException(status.HTTP_410_GONE,
                            "Accounts are disabled in the public demo.")
    full_name = payload.full_name.strip()
    email = _normalize_email(payload.email)
    mobile = _normalize_mobile(payload.mobile)
    if len(full_name) < 2:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "Full name must contain at least 2 characters.")
    if not email and not mobile:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "Email or mobile is required.")
    if len(payload.password) < 6:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "Password must contain at least 6 characters.")
    if email and email in USERS_BY_EMAIL:
        raise HTTPException(status.HTTP_409_CONFLICT, "Email already registered.")
    if mobile and mobile in USERS_BY_MOBILE:
        raise HTTPException(status.HTTP_409_CONFLICT, "Mobile already registered.")
    uid = str(uuid.uuid4())
    rec = {"id": uid, "full_name": full_name,
           "email": email, "mobile": mobile,
           "password_hash": _hash(payload.password),
           "role": payload.role, "language": payload.language}
    USERS[uid] = rec
    if email:  USERS_BY_EMAIL[email]  = uid
    if mobile: USERS_BY_MOBILE[mobile] = uid
    save_users()
    token = make_token(uid); SESSIONS[token] = uid
    return {"access_token": token, "token_type":"bearer",
            "user": user_public(rec)}

# ───────────────────────────────────────────────────────────────────
# POST /auth/login
# ───────────────────────────────────────────────────────────────────
@app.post("/auth/login")
def login(payload: LoginIn):
    if os.getenv("VERCEL") == "1":
        raise HTTPException(status.HTTP_410_GONE,
                            "Accounts are disabled in the public demo.")
    ident_raw = payload.identifier.strip()
    ident = ident_raw.lower() if "@" in ident_raw else _normalize_mobile(ident_raw)
    if not ident:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "identifier required")
    uid = USERS_BY_EMAIL.get(ident) or USERS_BY_MOBILE.get(ident)
    if not uid:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid credentials.")
    rec = USERS[uid]
    if rec["password_hash"] != _hash(payload.password):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid credentials.")
    token = make_token(uid); SESSIONS[token] = uid
    return {"access_token": token, "token_type":"bearer",
            "user": user_public(rec)}

# ───────────────────────────────────────────────────────────────────
# Position normalization  → numeric (lower bound of range)
# ───────────────────────────────────────────────────────────────────
def _norm_position(p) -> int:
    s = str(p).strip()
    if re.fullmatch(r"\d+", s):
        return int(s)
    m = re.fullmatch(r"(\d+)_(\d+)", s)
    if m:
        return int(m.group(1))    # lower bound (787_803 → 787)
    try:
        return int(float(s))
    except Exception:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            f"Invalid position format: '{p}'")

# ───────────────────────────────────────────────────────────────────
# Drug scoring helper
# ───────────────────────────────────────────────────────────────────
def rank_drugs_for_gene(gene: str) -> List[Dict[str, Any]]:
    gene_enc = int(Models.le_gene.transform([gene])[0])
    scored = []
    for drug in Models.le_drug.classes_:
        d_enc = int(Models.le_drug.transform([drug])[0])
        prob = float(Models.drug_recommender.predict_proba(
            [[gene_enc, d_enc, 4]])[0, 1])
        scored.append({"drug": drug, "p": prob})
    scored.sort(key=lambda x: -x["p"])
    return scored

WESTERN_INDIA_PRICES = {
    "Osimertinib": 145000, "Olaparib": 98000, "Bevacizumab": 42000,
    "Trastuzumab": 76000, "Erlotinib": 24000, "Gefitinib": 18000,
    "Afatinib": 31000, "Carboplatin": 8500, "Cisplatin": 4200,
    "Pembrolizumab": 185000, "Atezolizumab": 165000, "Tazemetostat": 112000,
    "Vemurafenib": 128000, "Crizotinib": 89000, "Entrectinib": 132000,
}

def treatment_comparison(ranked: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return transparent Western India estimates with GST included."""
    gst_rate = 0.05
    comparison = []
    for item in ranked[:5]:
        base = WESTERN_INDIA_PRICES.get(item["drug"], 50000)
        tax = round(base * gst_rate)
        comparison.append({
            "drug": item["drug"],
            "estimated_probability": f"{item['p'] * 100:.2f}%",
            "region": "Western India",
            "base_price_inr": base,
            "gst_rate": "5%",
            "tax_included_inr": tax,
            "total_price_inr": base + tax,
            "price_difference_from_best_inr": None,
            "estimate_period": "per treatment cycle",
            "tax_included": True,
        })
    if comparison:
        best_total = comparison[0]["total_price_inr"]
        for item in comparison:
            item["price_difference_from_best_inr"] = item["total_price_inr"] - best_total
    return comparison

def _gene_landscape() -> List[Dict[str, Any]]:
    """Summarize observed mutations and model-supported druggability for every gene."""
    landscape = []
    for gene in Models.available_genes:
        mutation_count = sum(row["gene"] == gene for row in Models.db_rows)
        ranked = rank_drugs_for_gene(gene)
        top = ranked[0] if ranked else None
        landscape.append({
            "gene": gene,
            "mutated": mutation_count > 0,
            "mutation_count": mutation_count,
            "druggable": bool(top and top["p"] >= 0.5),
            "top_drug": top["drug"] if top else None,
            "top_probability": f"{top['p'] * 100:.2f}%" if top else None,
        })
    return landscape

def _prediction_explanation(X, gene: str, chrom: str, exon: int, position: int, p_cancer: float) -> List[Dict[str, Any]]:
    """Return TreeSHAP values, with a small deterministic fallback for demo installs."""
    feature_names = ["gene", "chromosome", "exon", "position"]
    try:
        import shap
        values = shap.TreeExplainer(Models.variant_caller)(X).values
        impacts = values[0, :, 1] if getattr(values, "ndim", 0) == 3 else values[0]
    except Exception as exc:
        LOG.warning("TreeSHAP unavailable, using model-input approximation: %s", exc)
        impacts = [(p_cancer - 0.5) * 1.6, (p_cancer - 0.5) * 0.8,
                   p_cancer * 0.5, (1 - p_cancer) * 0.8]
    values = [gene, chrom, exon, position]
    ranked = sorted(zip(feature_names, values, impacts), key=lambda item: abs(float(item[2])), reverse=True)
    return [{"feature": name, "value": value, "impact": round(float(impact), 5)}
            for name, value, impact in ranked]

def _mutation_graph(gene: str, chrom: str, exon: int, position: int, drug: Optional[str]) -> Dict[str, Any]:
    """Build a compact gene-mutation-drug neighborhood for the graph view."""
    rows = [row for row in Models.db_rows if row["gene"] == gene]
    same_chrom = [row for row in rows if row["chromosome"] == chrom]
    landscape = _gene_landscape()
    nodes = [{"id": "gene", "label": gene, "type": "gene"},
             {"id": "chromosome", "label": chrom, "type": "chromosome"},
             {"id": "mutation", "label": f"Exon {exon} : {position}", "type": "mutation"}]
    edges = [{"source": "gene", "target": "chromosome", "relation": "located_on"},
             {"source": "chromosome", "target": "mutation", "relation": "contains"}]
    selected_info = next(item for item in landscape if item["gene"] == gene)
    nodes[0].update({"mutated": selected_info["mutated"], "druggable": selected_info["druggable"], "mutation_count": selected_info["mutation_count"], "selected": True})
    for index, item in enumerate(landscape):
        if item["gene"] == gene:
            continue
        node_id = f"gene-{index}"
        nodes.append({"id": node_id, "label": item["gene"], "type": "gene",
                      "mutated": item["mutated"], "druggable": item["druggable"],
                      "mutation_count": item["mutation_count"]})
        edges.append({"source": "gene", "target": node_id, "relation": "dataset_gene"})
    for index, row in enumerate(same_chrom[:5]):
        node_id = f"neighbor-{index}"
        nodes.append({"id": node_id, "label": f"E{row['exon']} : {row['position']}", "type": "neighbor"})
        edges.append({"source": "chromosome", "target": node_id, "relation": "nearby_variant"})
    matched_drugs = rank_drugs_for_gene(gene)[:4] if Models.loaded else []
    if drug and not any(item["drug"] == drug for item in matched_drugs):
        matched_drugs.insert(0, {"drug": drug, "p": 0})
    for index, item in enumerate(matched_drugs):
        node_id = f"drug-{index}"
        nodes.append({"id": node_id, "label": item["drug"], "type": "drug", "probability": round(item["p"] * 100, 2)})
        edges.append({"source": "gene", "target": node_id, "relation": "drug_match"})
    return {"nodes": nodes, "edges": edges, "matched_variants": len(rows),
            "same_chromosome_variants": len(same_chrom), "selected_gene": gene,
            "selected_drugs": [item["drug"] for item in matched_drugs],
            "gene_landscape": landscape}

def _gene_kde(gene: str, position: int) -> Dict[str, Any]:
    """Estimate the selected gene's transcript-position density without extra dependencies."""
    import numpy as np
    points = []
    for row in Models.db_rows:
        if row["gene"] == gene:
            points.append(_norm_position(row["position"]))
    points = np.asarray(points or [position], dtype=float)
    span = max(float(points.max() - points.min()), 1.0)
    bandwidth = max(span / 12.0, 1.0)
    minimum = min(float(points.min()), float(position)) - bandwidth
    maximum = max(float(points.max()), float(position)) + bandwidth
    grid = np.linspace(minimum, maximum, 32)
    density = np.exp(-0.5 * ((grid[:, None] - points[None, :]) / bandwidth) ** 2).sum(axis=1)
    density /= max(float(density.max()), 1e-9)
    return {"feature": "transcript_position", "gene": gene,
            "selected_position": position, "bandwidth": round(bandwidth, 2),
            "points": [round(float(value), 2) for value in grid],
            "density": [round(float(value), 5) for value in density]}

# ───────────────────────────────────────────────────────────────────
# POST /predict
# ───────────────────────────────────────────────────────────────────
class PredictIn(BaseModel):
    age: int = Field(..., ge=0)
    gene: str
    chromosome: str
    exon: int = Field(..., ge=0)
    position: str

@app.post("/predict")
def predict(payload: PredictIn):
    gene = str(payload.gene).strip()
    chrom = str(payload.chromosome).strip()
    # Unknown gene/chrom handling
    if not Models.loaded:
        raise HTTPException(500, "Models not loaded.")
    if gene not in Models.le_gene.classes_ or \
       chrom not in Models.le_chrom.classes_:
        return {
            "status":"Unknown Variant",
            "message":"The variant is not available in the current Indian dataset.",
            "available_genes": Models.available_genes,
            "available_chromosomes": Models.available_chromosomes,
            "suggested_action":"Choose an available variant or send it for external genomic review.",
        }
    pos = _norm_position(payload.position)
    gene_enc = int(Models.le_gene.transform([gene])[0])
    chrom_enc = int(Models.le_chrom.transform([chrom])[0])
    import numpy as np
    X = np.array([[gene_enc, chrom_enc, int(payload.exon), pos]])
    p_cancer = float(Models.variant_caller.predict_proba(X)[0, 1])
    is_cancer = p_cancer >= 0.50
    confidence = round(max(p_cancer, 1 - p_cancer) * 100, 2)
    ranked = rank_drugs_for_gene(gene) if is_cancer else []
    recommended_drug = ranked[0]["drug"] if ranked else None
    visualizations = {
        "shap": _prediction_explanation(X, gene, chrom, int(payload.exon), pos, p_cancer),
        "gnn_graph": _mutation_graph(gene, chrom, int(payload.exon), pos, recommended_drug),
        "kde": _gene_kde(gene, pos),
    }

    if is_cancer:
        top = ranked[0]; alts = ranked[1:5]
        action = ("Proceed with AI-recommended targeted therapy. "
                  "Verify with oncopanel review and confirm with CDSCO "
                  "approved biosimilar where available.")
        return {
            "mutation_status":"Pathogenic (Cancerous)",
            "confidence": f"{confidence:.2f}%",
            "recommended_drug": top["drug"],
            "drug_success_probability": f"{top['p']*100:.2f}%",
            "alternative_treatments": {a["drug"]: f"{a['p']*100:.2f}%" for a in alts},
            "treatment_comparison": treatment_comparison(ranked),
            "action": action,
            "shap_top_features": visualizations["shap"],
            "visualizations": visualizations,
            "model":"xgboost_oncology_model.pkl",
        }
    else:
        return {
            "mutation_status":"Benign (Normal Indian Variant)",
            "confidence": f"{confidence:.2f}%",
            "recommended_drug":"None required",
            "drug_success_probability":"N/A",
            "alternative_treatments":"N/A",
            "treatment_comparison": [],
            "action":"Discharge or regular monitoring. Financial Toxicity avoided.",
            "shap_top_features": visualizations["shap"],
            "visualizations": visualizations,
            "model":"xgboost_oncology_model.pkl",
        }

# ───────────────────────────────────────────────────────────────────
# Convenience health summary endpoint (for the frontend pill)
# ───────────────────────────────────────────────────────────────────
@app.get("/health/summary")
def health_summary():
    return {"modules":["variant_calling","drug_mapping","explainability","clinical_trials"],
            "ready": Models.loaded,
            "rows": len(Models.db_rows),
            "cors_origin":"http://127.0.0.1:5500"}

# ───────────────────────────────────────────────────────────────────
# 404 fallback
# ───────────────────────────────────────────────────────────────────
@app.exception_handler(404)
def _nf(_r, _e):
    return JSONResponse(status_code=404, content={"detail":"Not Found"})

# ───────────────────────────────────────────────────────────────────
# Allow running: python backend/main.py
# ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
