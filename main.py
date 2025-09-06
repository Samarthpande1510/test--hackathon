# main.py - FastAPI entrypoint
import io
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import json
import torch
from utils import parse_fasta_bytes, load_config
from models import initialize_models
from pipeline import process_sequences
import os

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deepsea-ai")

cfg = load_config('config.yaml')

# device
device = 'cuda' if (cfg.get('device') in (None, 'auto') and torch.cuda.is_available()) or cfg.get('device') == 'cuda' else 'cpu'
logger.info(f"Using device: {device}")

app = FastAPI(title="DeepSea-AI eDNA Pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models
dna_encoder = None
tax_classifier = None
novelty_vae = None

class EnvMetadata(BaseModel):
    depth: Optional[float] = None
    temperature: Optional[float] = None
    pressure: Optional[float] = None
    oxygen: Optional[float] = None

class AnalyzeResponse(BaseModel):
    taxonomic_predictions: List[Dict[str, Any]]
    novel_species: List[Dict[str, Any]]
    biodiversity_metrics: Dict[str, Any]
    abundance_estimates: List[Dict[str, Any]]
    processing_time: float
    environmental_correlations: Dict[str, Any]

@app.on_event("startup")
def startup_event():
    global dna_encoder, tax_classifier, novelty_vae
    dna_encoder, tax_classifier, novelty_vae = initialize_models(cfg, device)
    logger.info("Models initialized.")

@app.post("/analyze_edna", response_model=AnalyzeResponse)
async def analyze_edna(
    file: UploadFile = File(...),
    metadata: Optional[str] = None
):
    """
    Upload a FASTA file (or newline-separated sequences) and analyze.
    Optionally include `metadata` JSON string of list of per-sequence env metadata.
    """
    contents = await file.read()
    if len(contents) > cfg.get('max_upload_size_bytes', 50_000_000):
        raise HTTPException(status_code=413, detail="Uploaded file is too large")

    sequences = parse_fasta_bytes(contents)
    if not sequences:
        raise HTTPException(status_code=400, detail="No valid sequences found in upload")

    # limit to safe number
    max_seq = int(cfg.get('max_sequences_per_request', 1000))
    if len(sequences) > max_seq:
        sequences = sequences[:max_seq]

    env_meta = None
    if metadata:
        try:
            env_meta = json.loads(metadata)
            if not isinstance(env_meta, list):
                env_meta = None
        except Exception:
            env_meta = None

    try:
        results = process_sequences(
            sequences=sequences,
            dna_encoder=dna_encoder,
            tax_classifier=tax_classifier,
            novelty_vae=novelty_vae,
            cfg=cfg,
            device=device,
            env_metadata=env_meta
        )
        return JSONResponse(content=results)
    except Exception as e:
        logger.exception("Processing failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "models_loaded": dna_encoder is not None}

@app.post("/demo_analysis", response_model=AnalyzeResponse)
async def demo_analysis():
    sample_sequences = [
        "ATGCGATCGTAGCTAGCATGCTAGCTAGCATGCTAGCTAGCATGC",
        "GCTAGCTAGCATGCGATCGTAGCTAGCATGCTAGCTAGCATGCTA",
        "CGATCGTAGCTAGCATGCTAGCTAGCATGCTAGCTAGCATGCGAT",
        "TAGCTAGCATGCGATCGTAGCTAGCATGCTAGCTAGCATGCGATC",
        "ATGCTAGCTAGCATGCGATCGTAGCTAGCATGCTAGCTAGCATGC"
    ]
    results = process_sequences(
        sequences=sample_sequences,
        dna_encoder=dna_encoder,
        tax_classifier=tax_classifier,
        novelty_vae=novelty_vae,
        cfg=cfg,
        device=device
    )
    return JSONResponse(content=results)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
