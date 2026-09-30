import fitz  # PyMuPDF
import re
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Nereus AI Validation Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def clean_val(v):
    return re.sub(r"\s+", " ", str(v or "")).strip()

def extract_pdf_data(stream: bytes):
    doc = fitz.open(stream=stream, filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n"
    
    text = full_text.upper()
    
    # B/L Number
    bl_match = re.search(r"(?:BILL\s+OF\s+LADING|B/?L(?:\s*(?:NO\.?|NUMBER))?)\s*[:#\-]?\s*([A-Z0-9\-]{7,25})", text)
    bl_no = bl_match.group(1) if bl_match else "NOT DETECTED"
    
    # Carrier
    carrier = "CMA CGM" if "CMA CGM" in text else "MEDITERRANEAN SHIPPING COMPANY" if "MEDITERRANEAN SHIPPING" in text or "MSC" in text else "MAERSK" if "MAERSK" in text else "OTHER CARRIER"
    
    # Weights & Packages
    pkgs_match = re.search(r"(\d[\d,]*)\s*(BALES|BAGS|PACKAGES|PKGS)", text)
    pkgs = int(pkgs_match.group(1).replace(",", "")) if pkgs_match else 0
    unit = pkgs_match.group(2) if pkgs_match else "PACKAGES"
    
    gross_match = re.search(r"GROSS\s*WEIGHT\s*[:\-]?\s*([\d,.]+)", text)
    gross = float(gross_match.group(1).replace(",", "")) if gross_match else 0.0
    
    net_match = re.search(r"NET\s*WEIGHT\s*[:\-]?\s*([\d,.]+)", text)
    net = float(net_match.group(1).replace(",", "")) if net_match else 0.0
    
    # Containers
    containers = list(set(re.findall(r"\b([A-Z]{4}\d{7})\b", text)))
    
    # Statutory
    gstin = re.search(r"\b(\d{2}[A-Z]{5}\d{4}[A-Z]\dZ[A-Z0-9])\b", text)
    gstin_val = gstin.group(1) if gstin else ""
    
    iec = re.search(r"IEC\s*[:\-]?\s*([A-Z0-9]{10})\b", text) or re.search(r"\b(\d{10})\b", text)
    iec_val = iec.group(1) if iec else (gstin_val[2:12] if gstin_val else "")
    
    # Authority
    on_behalf = bool(re.search(r"ON\s+BEHALF\s+OF|O/B|JOINTLY\s+AND\s+SEVERALLY", text))
    
    return {
        "bl_no": bl_no,
        "carrier": carrier,
        "packages": pkgs,
        "unit": unit,
        "gross_kg": gross,
        "net_kg": net,
        "container_count": len(containers),
        "containers": containers,
        "gstin": gstin_val,
        "iec": iec_val,
        "on_behalf": on_behalf,
        "raw_text_length": len(full_text)
    }

@app.post("/validate")
async def validate_document(file: UploadFile = File(...)):
    content = await file.read()
    extracted = extract_pdf_data(content)
    
    # Rulebook checks
    discrepancies = []
    
    # 1. Authority
    if not extracted["on_behalf"]:
        discrepancies.append({
            "field": "Shipper Authority Clause",
            "req": "'On Behalf Of' / Principal Clause required",
            "stated": "Not present",
            "status": "blocker",
            "fix": "Add 'On Behalf Of [Principal]' to Shipper details"
        })
    else:
        discrepancies.append({
            "field": "Shipper Authority Clause",
            "req": "'On Behalf Of' / Principal Clause",
            "stated": "Verified Present",
            "status": "match",
            "fix": "No action required"
        })
        
    # 2. Cargo Packaging Tare
    if extracted["packages"] > 0 and extracted["gross_kg"] > extracted["net_kg"]:
        diff = extracted["gross_kg"] - extracted["net_kg"]
        per_pkg = diff / extracted["packages"]
        discrepancies.append({
            "field": "Packaging Tare & Arithmetic",
            "req": "Packaging tare reconciles with commodity benchmark",
            "stated": f"{diff:.2f} kg total ({per_pkg:.2f} kg/unit)",
            "status": "match" if 0.9 <= per_pkg <= 2.8 else "warning",
            "fix": "Verify declared gross and net weights" if per_pkg < 0.9 or per_pkg > 2.8 else "No action required"
        })
        
    # 3. Statutory
    if extracted["gstin"]:
        discrepancies.append({
            "field": "GSTIN Validation",
            "req": "15-char valid format",
            "stated": extracted["gstin"],
            "status": "match",
            "fix": "No action required"
        })

    blockers = [d for d in discrepancies if d["status"] == "blocker"]
    
    return {
        "success": True,
        "verdict": "APPROVED" if len(blockers) == 0 else "CRITICAL BLOCKERS FOUND",
        "extracted": extracted,
        "discrepancies": discrepancies
    }
