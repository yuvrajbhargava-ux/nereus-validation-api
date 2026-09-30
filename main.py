import fitz  # PyMuPDF
import re
import io
import base64
import pytesseract
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Nereus AI Validation Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_pdf_data(stream: bytes):
    doc = fitz.open(stream=stream, filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n"
    
    if len(full_text.strip()) < 50:
        full_text = ""
        for page in doc:
            pix = page.get_pixmap(dpi=150)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            full_text += pytesseract.image_to_string(img) + "\n"
            
    text = full_text.upper()
    
    bl_match = (
        re.search(r"\b(MEDU[A-Z0-9]{7,12})\b", text) or
        re.search(r"\b(AEV\d{7})\b", text) or
        re.search(r"\b(DKA\d{7}[A-Z]?)\b", text) or
        re.search(r"\b(RTM\d{7}[A-Z]?)\b", text) or
        re.search(r"(?:BILL\s+OF\s+LADING|B/?L(?:\s*(?:NO\.?|NUMBER))?)\s*[:#\-]?\s*([A-Z0-9\-]{7,25})", text)
    )
    bl_no = bl_match.group(1).replace("O", "0") if bl_match and "MEDU" in bl_match.group(1) else (bl_match.group(1) if bl_match else "NOT DETECTED")
    
    if "MEDITERRANEAN SHIPPING" in text or "MSC" in text:
        carrier = "MEDITERRANEAN SHIPPING COMPANY"
    elif "CMA CGM" in text:
        carrier = "CMA CGM"
    elif "MAERSK" in text:
        carrier = "MAERSK"
    else:
        carrier = "OTHER CARRIER"
        
    pkgs_match = re.search(r"(\d[\d,]*)\s*(BALES|BAGS|PACKAGES|PKGS)", text)
    pkgs = int(pkgs_match.group(1).replace(",", "")) if pkgs_match else 0
    unit = pkgs_match.group(2) if pkgs_match else "PACKAGES"
    
    gross_match = re.search(r"(?:TOTAL\s+)?GROSS\s*(?:WEIGHT|WT)?\s*[:\-]?\s*([\d,.]+)", text)
    gross = float(gross_match.group(1).replace(",", "")) if gross_match else 0.0
    
    net_match = re.search(r"(?:TOTAL\s+)?NET\s*(?:WEIGHT|WT)?\s*[:\-]?\s*([\d,.]+)", text)
    net = float(net_match.group(1).replace(",", "")) if net_match else 0.0
    
    containers = list(set(re.findall(r"\b([A-Z]{4}\d{7})\b", text)))
    
    gstin = re.search(r"\b(\d{2}[A-Z]{5}\d{4}[A-Z]\dZ[A-Z0-9])\b", text)
    gstin_val = gstin.group(1) if gstin else ""
    
    iec = re.search(r"IEC(?:\s*CODE)?\s*[:\-]?\s*([A-Z0-9]{10})\b", text) or re.search(r"\b(\d{10})\b", text)
    iec_val = iec.group(1) if iec else (gstin_val[2:12] if gstin_val else "")
    
    on_behalf = bool(re.search(r"ON\s+BEHALF\s+OF|O/B|JOINTLY\s+AND\s+SEVERALLY", text))
    
    # Render page images with redline marks
    page_images = []
    for page in doc:
        # Draw green checks on verified statutory references
        for inst in page.search_for("CMA CGM") + page.search_for("ORIGINAL"):
            page.draw_rect(inst, color=(0.07, 0.48, 0.27), width=1.5)
        
        pix = page.get_pixmap(dpi=150)
        img_b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
        page_images.append(f"data:image/png;base64,{img_b64}")

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
        "pages": page_images
    }

@app.post("/validate")
async def validate_document(file: UploadFile = File(...)):
    content = await file.read()
    extracted = extract_pdf_data(content)
    
    discrepancies = []
    
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
            "stated": "Verified Present (Jointly/On Behalf)",
            "status": "match",
            "fix": "No action required"
        })
        
    if extracted["packages"] > 0 and extracted["gross_kg"] > extracted["net_kg"]:
        diff = extracted["gross_kg"] - extracted["net_kg"]
        per_pkg = diff / extracted["packages"]
        is_ok = 0.9 <= per_pkg <= 2.8
        discrepancies.append({
            "field": "Packaging Tare & Arithmetic",
            "req": "Packaging tare reconciles with commodity benchmark",
            "stated": f"{diff:.2f} kg total ({per_pkg:.2f} kg/{extracted['unit'].lower()})",
            "status": "match" if is_ok else "warning",
            "fix": "Verify declared weights against tare allowance" if not is_ok else "No action required"
        })
        
    if extracted["gstin"]:
        discrepancies.append({
            "field": "GSTIN Validation",
            "req": "15-character statutory format",
            "stated": extracted["gstin"],
            "status": "match",
            "fix": "No action required"
        })
        
    if extracted["iec"]:
        is_pan_linked = len(extracted["iec"]) == 10 and extracted["gstin"] and extracted["iec"] == extracted["gstin"][2:12]
        discrepancies.append({
            "field": "DGFT IEC Format",
            "req": "PAN-based 10-character alphanumeric IEC",
            "stated": extracted["iec"],
            "status": "match" if is_pan_linked else "warning",
            "fix": "Update to PAN-linked IEC under DGFT reform" if not is_pan_linked else "No action required"
        })

    blockers = [d for d in discrepancies if d["status"] == "blocker"]
    
    return {
        "success": True,
        "verdict": "APPROVED" if len(blockers) == 0 else "CRITICAL BLOCKERS FOUND",
        "extracted": extracted,
        "discrepancies": discrepancies
    }
