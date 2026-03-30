from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.legalbert_service import analyze_legal_text

router = APIRouter(prefix="/legalbert", tags=["Legal Analysis"])


class LegalTextRequest(BaseModel):
    text: str
    language: str = "eu"


@router.post("/analyze")
async def analyze(request: LegalTextRequest):
    """Analyse un texte légal EU avec LEGAL-BERT EU (FP16, zéro coût)."""
    result = await analyze_legal_text(request.text)

    if "error_type" in result:
        if result["error_type"] == "validation_error":
            raise HTTPException(status_code=400, detail=result["error"])
        raise HTTPException(status_code=500, detail=result["error"])

    return result
