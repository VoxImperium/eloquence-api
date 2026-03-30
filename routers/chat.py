# routers/chat.py — Endpoint de chat juridique avec RAG OpenLégi

import logging
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.chat_service import chat_service
from schemas.chat import Message, Source

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)

_DOMAINES_VALIDES = {"civil", "pénal", "social", "commercial", "administratif"}


class ChatRequest(BaseModel):
    messages: List[Message]
    domaine: str = "civil"


class ChatResponse(BaseModel):
    response: str
    sources: List[Source]


@router.post("/juridique", response_model=ChatResponse)
async def chat_juridique(req: ChatRequest):
    """
    Chat juridique avec RAG : recherche OpenLégi + génération Groq.

    - Recherche simultanée jurisprudence + textes de loi (max 3 chacun)
    - Réponse contextualisée avec citations officielles
    - Fallback gracieux si OpenLégi est indisponible
    """
    if not req.messages:
        raise HTTPException(status_code=422, detail="La liste de messages est vide.")

    domaine = req.domaine if req.domaine in _DOMAINES_VALIDES else "civil"
    user_question = req.messages[-1].content.strip()

    if not user_question:
        raise HTTPException(status_code=422, detail="Le message utilisateur est vide.")

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

    return ChatResponse(response=response_text, sources=sources)
