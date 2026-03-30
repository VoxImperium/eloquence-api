# routers/chat.py — Endpoint de chat juridique avec RAG OpenLégi

import logging
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.chat_service import chat_service
from services.legalbert_service import analyze_legal_text
from schemas.chat import Message, Source

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)

_DOMAINES_VALIDES = {"civil", "pénal", "social", "commercial", "administratif"}

_WARNING_LOW_LEGAL_SCORE = "Question peu légale"


class ChatRequest(BaseModel):
    messages: List[Message]
    domaine: str = "civil"


class ChatResponse(BaseModel):
    response: str
    sources: List[Source]
    question_legal_score: float | None = None
    response_legal_score: float | None = None
    warning: str | None = None


def _extract_legal_score(bert_result: dict) -> float | None:
    """Extrait le score de légalité du résultat LegalBert.

    Retourne la probabilité de LABEL_1 (classe légale positive),
    ou None si le résultat est invalide / le modèle est indisponible.
    """
    results = bert_result.get("results")
    if not results:
        return None
    label = results[0].get("label", "")
    score = results[0].get("score")
    if score is None:
        return None
    # LABEL_1 = classe légale positive ; LABEL_0 = négative
    return float(score) if label == "LABEL_1" else float(1.0 - score)


@router.post("/juridique", response_model=ChatResponse)
async def chat_juridique(req: ChatRequest):
    """
    Chat juridique avec RAG : recherche OpenLégi + génération Groq.

    - Score LegalBert de la question (avant OpenLégi)
    - Avertissement si score < 0.3 (question peu juridique)
    - Recherche simultanée jurisprudence + textes de loi (max 3 chacun)
    - Réponse contextualisée avec citations officielles
    - Score LegalBert de la réponse Groq (après génération)
    - Fallback gracieux si OpenLégi est indisponible
    """
    if not req.messages:
        raise HTTPException(status_code=422, detail="La liste de messages est vide.")

    domaine = req.domaine if req.domaine in _DOMAINES_VALIDES else "civil"
    user_question = req.messages[-1].content.strip()

    if not user_question:
        raise HTTPException(status_code=422, detail="Le message utilisateur est vide.")

    # ── Scoring LegalBert de la question ─────────────────────────────────────
    question_legal_score: float | None = None
    try:
        question_bert = await analyze_legal_text(user_question)
        question_legal_score = _extract_legal_score(question_bert)
    except Exception as exc:
        logger.warning("LegalBert indisponible pour scoring question : %s", exc)

    # Avertissement si la question est peu liée au droit
    if question_legal_score is not None and question_legal_score < 0.3:
        return ChatResponse(
            response=(
                "Cette question ne semble pas directement liée au droit. "
                "Reformulez avec un contexte juridique."
            ),
            sources=[],
            question_legal_score=question_legal_score,
            response_legal_score=None,
            warning=_WARNING_LOW_LEGAL_SCORE,
        )

    # Historique de conversation (tout sauf le dernier message)
    history = [m.model_dump() for m in req.messages[:-1]]

    try:
        response_text, sources = await chat_service.answer(
            question=user_question,
            domaine=domaine,
            history=history,
        )
    except Exception as exc:
        logger.error("Erreur inattendue dans /chat/juridique : %s", exc)
        raise HTTPException(status_code=500, detail="Erreur interne du serveur.") from exc

    # ── Scoring LegalBert de la réponse ──────────────────────────────────────
    response_legal_score: float | None = None
    try:
        response_bert = await analyze_legal_text(response_text[:512])  # BERT max 512 tokens
        response_legal_score = _extract_legal_score(response_bert)
    except Exception as exc:
        logger.warning("LegalBert indisponible pour scoring réponse : %s", exc)

    return ChatResponse(
        response=response_text,
        sources=sources,
        question_legal_score=question_legal_score,
        response_legal_score=response_legal_score,
        warning=None,
    )
