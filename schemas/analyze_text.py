# schemas/analyze_text.py — Modèles Pydantic pour l'analyse de texte légal avec LegalBert

from typing import List, Optional

from pydantic import BaseModel


class AnalyzeTextRequest(BaseModel):
    text: str
    search_jurisprudence: bool = True


class AnalyzeTextResponse(BaseModel):
    text: str
    classification: str
    legal_score: float
    latency_ms: float
    jurisprudence: Optional[List[dict]] = None
    textes: Optional[List[dict]] = None
