import fitz  # PyMuPDF
import json
import os
import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types

app = FastAPI(title="Nereus AI Trade Document Validation Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

NEREUS_SYSTEM_INSTRUCTION = """
You are Nereus AI, the Senior Trade Document Validation & Compliance Specialist.
Your task is to audit maritime Draft Bills of Lading against international trade compliance standards, statutory regulations, and strict physical arithmetic benchmarks.

You must return a single, valid JSON object following this exact schema:
{
  "bl_no": "Extracted B/L number",
  "carrier": "Carrier legal entity name",
  "commodity": "Commodity description (e.g., Raw Cotton in Compressed Bales, Raw Cashew Nuts in Shell)",
  "commodity_type": "cotton | cashew | general",
  "packages": 880,
  "unit": "Bales | Bags | Packages",
  "gross_kg": 204950.0,
  "gross_mts": 204.950,
  "net_kg": 203190.0,
  "net_mts": 203.190,
  "container_count": 8,
  "containers": ["TCKU7165641", "SEGU5438621"],
  "avg_unit_weight": "230.9 kg/bale",
  "gstin": "15-character GSTIN",
  "iec": "10-character IEC",
  "header_vessel": "Vessel stated in top header",
  "sob_vessel": "Vessel stated in Shipped-on-Board execution box",
  "verdict": "APPROVED | CRITICAL BLOCKERS FOUND",
  "discrepancies": [
    {
      "field": "Field name",
      "req": "Benchmark / statutory requirement",
      "stated": "Value stated on B/L",
      "status": "match | warning | blocker",
      "fix": "Required editorial correction"
    }
  ],
  "redline_actions": [
    {
      "action": "check",
      "search_text": "Exact text on page to place a discrete green checkmark next to"
    },
    {
      "action": "strike",
      "search_text": "Exact text on page to draw a horizontal red line through",
      "correction_text": "Correct value to write in open whitespace to the right"
    },
    {
      "action": "callout",
      "search_text": "Anchor text to place an outline box and callout",
      "callout_text": "Required insertion or correction note"
    }
  ]
}

CRITICAL RULES TO APPLY:
1. Shipper Authority: If the Shipper block lacks an 'On Behalf Of [Principal]' clause (or O/B, Pour le compte de, Jointly and Severally), flag as BLOCKER.
2. Vessel Consistency: If Header Vessel != Shipped-on-Board Vessel, flag as BLOCKER.
3. Weight Decimal Format: If weights have missing decimals (e.g. '204 060 MTS'), flag as BLOCKER.
4. Packaging Tare Arithmetic:
   - Packaging Tare = Gross KG - Net KG.
   - For Cotton: Benchmark is 1.80 - 2.50 kg/bale.
   - For Raw Cashew Nuts (RCN): Benchmark is 0.90 - 1.30 kg/bag.
   - Container equipment tare (empty steel boxes: ~3,700-3,900 kg) is separate and must NEVER be subtracted from cargo weights.
5. Statutory IDs:
   - GSTIN must be 15 alphanumeric characters.
   - DGFT IEC must be 10 characters (PAN-based alphanumeric is MATCH; legacy 10-digit numeric is WARNING).
6. Redline Actions:
   - Provide clean 'check' actions for verified matches (Carrier, B/L No, SOB stamp, GSTIN, IEC, Packages).
   - Provide 'strike' or 'callout' actions for any identified blocker or warning.
"""

def annotate_pdf_pages(doc, redline_actions):
    """Draws discrete green checks and redline strikethroughs directly onto PDF pixels."""
    page_images = []
    
    for page in doc:
        # Apply checks
        for action in redline_actions:
            search = action.get("search_text", "")
            if not search:
                continue
            
            if action.get("action") == "check":
                for rect in page.search_for(search)[:2]:
                    page.insert_text((rect.x1 + 4, rect.y1 - 1), "✓", fontsize=11, color=(0.07, 0.48, 0.27))
                    
            elif action.get("action") == "strike":
                for rect in page.search_for(search):
                    mid_y = (rect.y0 + rect.y1) / 2
                    page.draw_line(fitz.Point(rect.x0, mid_y), fitz.Point(rect.x1, mid_y), color=(0.78, 0.14, 0.10), width=1.5)
                    corr = action.get("correction_text")
                    if corr:
                        page.insert_text((rect.x1 + 8, rect.y1 - 1), corr, fontsize=9.5, color=(0.78, 0.14, 0.10))
                        
            elif action.get("action") == "callout" and page.number == 0:
                for rect in page.search_for(search)[:1]:
                    box = fitz.Rect(rect.x0 - 2, rect.y0 - 2, rect.x0 + 260, rect.y0 + 50)
                    page.draw_rect(box, color=(0.78, 0.14, 0.10), width=1.2)
                    call = action.get("callout_text")
                    if call:
                        page.insert_text((box.x1 + 8, box.y0 + 14), call, fontsize=8.5, color=(0.78, 0.14, 0.10))

        pix = page.get_pixmap(dpi=150)
        img_b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
        page_images.append(f"data:image/png;base64,{img_b64}")

    return page_images

@app.post("/validate")
async def validate_document(file: UploadFile = File(...)):
    content = await file.read()
    doc = fitz.open(stream=content, filetype="pdf")
    
    full_text = ""
    for idx, page in enumerate(doc):
        full_text += f"\n--- PAGE {idx + 1} ---\n" + page.get_text()

    # Call Gemini API with Nereus AI system instructions
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    prompt = f"Perform a complete trade compliance and redline audit on this Bill of Lading text:\n\n{full_text}"
    
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=NEREUS_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            temperature=0.1
        )
    )
    
    audit_data = json.loads(response.text)
    
    # Render PDF pages with exact redlines and green checks
    redline_actions = audit_data.get("redline_actions", [])
    page_images = annotate_pdf_pages(doc, redline_actions)
    audit_data["pages"] = page_images
    
    # Calculate counts
    blockers = [d for d in audit_data.get("discrepancies", []) if d.get("status") == "blocker"]
    warnings = [d for d in audit_data.get("discrepancies", []) if d.get("status") == "warning"]
    matches = [d for d in audit_data.get("discrepancies", []) if d.get("status") == "match"]
    
    audit_data["blockers_count"] = len(blockers)
    audit_data["warnings_count"] = len(warnings)
    audit_data["matches_count"] = len(matches)
    
    return {
        "success": True,
        "verdict": audit_data.get("verdict", "APPROVED"),
        "extracted": audit_data,
        "discrepancies": audit_data.get("discrepancies", [])
    }
