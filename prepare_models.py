"""One-time helper to (1) write a synthetic dbGENVOC-style xlsx and
(2) train XGBoost + RandomForest + 3 LabelEncoders into backend/models/.
Run from /home/user/onco-cdss once. Idempotent.
"""
from pathlib import Path
import numpy as np, pandas as pd, joblib
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).parent
MODELS = ROOT / "backend" / "models"
MODELS.mkdir(parents=True, exist_ok=True)
XLSX = ROOT / "backend" / "dbGENVOC.xlsx"

# ── 1. Synthetic Indian variant dataset (dbGENVOC-shaped columns) ──
N_GENES = 20
np.random.seed(42)
genes = ["TP53","EGFR","BRCA1","BRCA2","KRAS","BRAF","ALK","ROS1","MET","HER2",
         "PIK3CA","PTEN","APC","RB1","NRAS","HRAS","CDKN2A","CDK4","MDM2","MYC"]
chroms = ["chr1","chr2","chr3","chr5","chr7","chr8","chr11","chr12","chr13","chr17","chr19","chr22"]

rows = []
for gene in genes:
    for chrom in chroms:
        # every gene gets 2–4 exons/positions on each chromosome
        for _ in range(np.random.randint(2, 5)):
            exon = int(np.random.randint(2, 22))
            start = int(np.random.randint(100, 9999))
            end = start + int(np.random.randint(2, 6))
            rows.append({
                "Gene name": gene, "Chrom": chrom,
                "Transcript exon": exon, "Transcript position": f"{start}_{end}",
            })
df = pd.DataFrame(rows).drop_duplicates(["Gene name","Chrom","Transcript position"])
df.to_excel(XLSX, sheet_name="dbGENVOC_subset", index=False,
            startrow=1)  # so the second row is the header
# append header + 1 blank leading row to match the "second row = header" rule
final = pd.read_excel(XLSX, header=1)
final.columns = ["Gene name","Chrom","Transcript exon","Transcript position"]
final["label"] = np.random.choice([0,1], len(final), p=[0.55, 0.45])
final.to_excel(XLSX, index=False, startrow=1)
print(f"Wrote dbGENVOC-shape xlsx: {len(final)} rows -> {XLSX}")

# ── 2. Train encoders + XGBoost caller ──────────────────────────────
le_gene = LabelEncoder().fit(final["Gene name"].astype(str))
le_chrom = LabelEncoder().fit(final["Chrom"].astype(str))
Xgb = final.copy()
Xgb["Gene name_encoded"] = le_gene.transform(Xgb["Gene name"].astype(str))
Xgb["Chrom_encoded"]     = le_chrom.transform(Xgb["Chrom"].astype(str))
Xgb["pos_norm"] = Xgb["Transcript position"].astype(str).apply(lambda s: int(str(s).split("_")[0]))
Xgb["exon_norm"] = Xgb["Transcript exon"].astype(int)
features = ["Gene name_encoded","Chrom_encoded","exon_norm","pos_norm"]
X = Xgb[features].values; y = Xgb["label"].values

variant_model = XGBClassifier(
    n_estimators=120, max_depth=4, learning_rate=0.15,
    subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
    random_state=0, verbosity=0
).fit(X, y)

# ── 3. Drug recommendation: synthetic CDSCO-mapping RF ──────────────
drugs = ["Osimertinib","Olaparib","Bevacizumab","Trastuzumab","Erlotinib",
         "Gefitinib","Afatinib","Carboplatin","Cisplatin","Pembrolizumab",
         "Atezolizumab","Tazemetostat","Vemurafenib","Crizotinib","Entrectinib"]
le_drug = LabelEncoder().fit(drugs)
# synthetic per-gene drug ranking signal
gene_to_drugs = {g: np.random.choice(drugs, size=5, replace=False) for g in genes}
drug_rows = []
for gene in genes:
    for drug in gene_to_drugs[gene]:
        s = np.random.poisson(0.6) + 2
        drug_rows.append({
            "Gene name_encoded": le_gene.transform([gene])[0],
            "Chrom_encoded": le_chrom.transform(["chr17"])[0],
            "Drug_encoded": le_drug.transform([drug])[0],
            "score": s,
            "success": 1 if s > 2 else 0,
        })
df_d = pd.DataFrame(drug_rows)
drug_model = RandomForestClassifier(
    n_estimators=80, max_depth=6, random_state=0
).fit(df_d[["Gene name_encoded","Drug_encoded","score"]].values,
      df_d["success"].values)

# ── 4. Save everything via joblib ──
joblib.dump(variant_model, MODELS / "xgboost_oncology_model.pkl")
joblib.dump(drug_model,     MODELS / "drug_recommendation_model.pkl")
joblib.dump(le_gene,        MODELS / "le_gene.pkl")
joblib.dump(le_chrom,       MODELS / "le_chrom.pkl")
joblib.dump(le_drug,        MODELS / "le_drug.pkl")
print("Saved 5 .pkl files into", MODELS)
