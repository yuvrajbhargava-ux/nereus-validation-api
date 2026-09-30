import fitz  # PyMuPDF
import json
import os
import traceback
import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from google import genai
from google.genai import types

app = FastAPI(title="Nereus AI Autonomous Trade Validation Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

NEREUS_SYSTEM_INSTRUCTION = """
You are Nereus AI, the Senior Trade Document Validation & Compliance Specialist.
Perform an exhaustive, multi-point statutory and commercial audit on maritime Draft Bills of Lading.

You must return a single, valid JSON object following this exact schema:
{
  "bl_no": "Extracted B/L number",
  "carrier": "Carrier legal entity name",
  "commodity": "Commodity description (e.g. Raw Cotton in Compressed Bales, Raw Cashew Nuts in Shell)",
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
  "verdict": "APPROVED | CRITICAL BLOCKERS FOUND",
  "discrepancies": [
    {
      "field": "Exact Field Name (e.g. On Behalf Of, Shipper, Consignee, Notify Party, GSTIN, IEC Code, GSTIN-IEC Consistency, Commodity, HS Code, Origin, Crop Year, Contract Ref, Total Bales, Total Cargo Gross, Total Cargo Net, Average Bale Weight, Packaging Tare, Port of Loading, Port of Discharge, Vessel Name, Ocean Voyage, Container Inventory, Container Tare Tolerance, Freight Terms, SOB Stamp, Originals Count, Carrier Entity)",
      "req": "Benchmark / statutory / contractual requirement",
      "stated": "Exact stated value on draft B/L",
      "status": "match | warning | blocker",
      "fix": "Required editorial correction or 'No action required'"
    }
  ],
  "redline_actions": [
    {
      "action": "check | strike | callout",
      "search_text": "Text anchor on document",
      "correction_text": "Correction value (if strike)",
      "callout_text": "Callout message (if callout)"
    }
  ]
}

MANDATORY AUDIT DIMENSIONS TO EVALUATE:
1. On Behalf Of Clause (Carrier template requirement: Shipper must have 'On Behalf Of [Principal]' or O/B. Flag BLOCKER if absent.)
2. Shipper Legal Entity & Address
3. Consignee Negotiability (Standard 'TO ORDER')
4. Notify Party Address & PIN Code Alignment
5. GSTIN Format (15-character statutory format; state code matches destination)
6. DGFT IEC Code (10-character DGFT compliance; PAN-based is MATCH, numeric is WARNING)
7. GSTIN-IEC Consistency (Characters 3-12 of GSTIN must equal IEC)
8. Commercial Commodity Description
9. HS Code (8-digit Indian Customs Tariff classification)
10. Country of Origin
11. Crop Year
12. Contract Reference
13. Total Package Sum (Sum across all containers)
14. Total Cargo Gross Weight (Tonnage and kg reconciliation)
15. Total Cargo Net Weight (Decimal and numeric format verification; flag BLOCKER if decimal missing)
16. Average Unit Weight (Benchmark: Cotton 210-240 kg/bale; Cashew ~80 kg/bag)
17. Packaging Tare Arithmetic (Gross KG - Net KG / packages; Cotton: 1.80-2.50 kg/bale; Cashew: 0.90-1.30 kg/bag)
18. Port of Loading
19. Port of Discharge
20. Vessel Name Consistency (Header vessel must match Shipped on Board vessel; flag BLOCKER if mismatched)
21. Ocean Voyage Number
22. Container Count & Type (e.g. 40HC, 20FT)
23. Container Inventory & Seal Numbers
24. Container Equipment Tare Tolerances (Benchmark: 3,700-3,900 kg per 40HC steel container)
25. Freight Payment Terms (Prepaid vs Collect)
26. Shipped on Board Execution (Date, port, and signature)
27. Originals Set Count (Standard THREE (3))
28. Carrier Legal Entity Name & Registration
"""

def annotate_pdf_pages(doc, redline_actions):
    page_images = []
    baseline_checks = [
        "ORIGINAL", "BILL OF LADING", "FREIGHT PREPAID", "CONTAINER", "SEAL",
        "TO ORDER", "MANGALORE", "TUTICORIN", "ABIDJAN", "SINGAPORE", "CMA CGM",
        "MEDITERRANEAN", "MSC", "MAERSK", "SHIPPED ON BOARD"
    ]

    for page in doc:
        for term in baseline_checks:
            for rect in page.search_for(term)[:1]:
                page.insert_text((rect.x1 + 4, rect.y1 - 1), "✓", fontsize=11, color=(0.07, 0.48, 0.27))

        for action in redline_actions:
            search = action.get("search_text", "")
            if not search:
                continue
            act = action.get("action")
            if act == "check":
                for rect in page.search_for(search)[:2]:
                    page.insert_text((rect.x1 + 4, rect.y1 - 1), "✓", fontsize=11, color=(0.07, 0.48, 0.27))
            elif act == "strike":
                for rect in page.search_for(search):
                    mid_y = (rect.y0 + rect.y1) / 2
                    page.draw_line(fitz.Point(rect.x0, mid_y), fitz.Point(rect.x1, mid_y), color=(0.78, 0.14, 0.10), width=1.5)
                    corr = action.get("correction_text")
                    if corr:
                        page.insert_text((rect.x1 + 8, rect.y1 - 1), corr, fontsize=9.5, color=(0.78, 0.14, 0.10))
            elif act == "callout" and page.number == 0:
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

    audit_data = None
    prompt = f"Perform the complete 28-point trade compliance and redline audit on this Bill of Lading text:\n\n{full_text}"
    diagnostics = []

    # 1. Primary: Groq API
    if GROQ_API_KEY:
        try:
            print("Executing Groq request...")
            groq_client = Groq(api_key=GROQ_API_KEY)
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": NEREUS_SYSTEM_INSTRUCTION},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",
                response_format={"type": "json_object"},
                temperature=0.1
            )
            raw_out = chat_completion.choices[0].message.content
            print("Groq response received successfully")
            if raw_out:
                audit_data = json.loads(raw_out)
        except Exception as e:
            err = f"Groq execution failed: {type(e).__name__} - {str(e)}"
            print(err)
            traceback.print_exc()
            diagnostics.append(err)

    # 2. Failover: Gemini API
    if not audit_data and GEMINI_API_KEY:
        try:
            print("Failing over to Gemini request...")
            gemini_client = genai.Client(api_key=GEMINI_API_KEY)
            resp = gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=NEREUS_SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    temperature=0.1
                )
            )
            print("Gemini response received successfully")
            if resp and resp.text:
                audit_data = json.loads(resp.text)
        except Exception as e:
            err = f"Gemini execution failed: {type(e).__name__} - {str(e)}"
            print(err)
            traceback.print_exc()
            diagnostics.append(err)

    if not audit_data:
        diag_str = " | ".join(diagnostics) if diagnostics else "No API keys configured"
        raise RuntimeError(f"Audit Engine Failure: {diag_str}")

    # Annotate pages
    redline_actions = audit_data.get("redline_actions", [])
    page_images = annotate_pdf_pages(doc, redline_actions)
    audit_data["pages"] = page_images
    
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
