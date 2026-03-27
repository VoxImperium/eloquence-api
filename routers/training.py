# api/routers/training.py
# Mode entraînement — dialogue socratique sur 500 sujets

import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from groq import Groq
from services.topics import get_all_topics, get_categories, get_random_topic
import os
import json

router = APIRouter(prefix="/training", tags=["training"])
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SOCRATE_SYSTEM = """Tu es Socrate, le philosophe grec. Tu pratiques la maïeutique :
tu poses des questions pour aider l'interlocuteur à approfondir sa pensée.
Tu ne donnes jamais directement ton avis, tu questionnes, tu challenges, tu creuses.
Tu parles un français soutenu mais accessible. Tu poses UNE seule question à la fois.
Maximum 3 phrases par réponse. Tu peux parfois citer des philosophes pertinents."""


class TrainingMessage(BaseModel):
    topic:      str
    category:   str
    messages:   list
    user_input: str


class TrainingDebrief(BaseModel):
    topic:    str
    category: str
    messages: list


@router.get("/topics")
async def get_topics():
    """Retourne tous les sujets par catégorie."""
    return {
        "categories": get_categories(),
        "topics":     get_all_topics(),
        "total":      len(get_all_topics())
    }


@router.get("/random")
async def random_topic():
    """Retourne un sujet aléatoire."""
    return get_random_topic()


@router.post("/message")
async def training_message(req: TrainingMessage):
    """
    Dialogue socratique — Socrate répond et questionne.
    """
    history = []
    for msg in req.messages[-12:]:
        history.append({"role": msg["role"], "content": msg["content"]})

    history.append({"role": "user", "content": req.user_input})

    system = f"""{SOCRATE_SYSTEM}

Sujet du débat : "{req.topic}" (catégorie : {req.category})

L'utilisateur développe sa position sur ce sujet. Aide-le à approfondir sa pensée
en posant des questions pertinentes, en pointant les contradictions, en explorant
les implications de ses arguments."""

    try:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system},
                    *history
                ],
                max_tokens=250,
                temperature=0.7,
            )
        )
        return {"response": response.choices[0].message.content.strip()}
    except Exception as e:
        raise HTTPException(500, f"Erreur: {str(e)}")


@router.post("/debrief")
async def training_debrief(req: TrainingDebrief):
    """Génère un bilan de la session d'entraînement."""
    conversation = "\n".join([
        f"{'Toi' if m['role'] == 'user' else 'Socrate'}: {m['content']}"
        for m in req.messages
    ])

    prompt = f"""Sujet : "{req.topic}"

Conversation :
{conversation}

Génère un bilan JSON de la qualité argumentative :
{{
  "note_argumentation": <0-10>,
  "note_richesse": <0-10>,
  "note_coherence": <0-10>,
  "note_globale": <moyenne>,
  "points_forts": ["point 1", "point 2"],
  "axes_amelioration": ["axe 1", "axe 2"],
  "meilleur_argument": "citation de l'argument le plus fort",
  "philosophes_a_lire": ["philosophe 1", "philosophe 2"],
  "conseil": "UN conseil pour progresser en argumentation",
  "resume": "2 phrases résumant la qualité du débat"
}}"""

    try:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "Coach en éloquence et philosophie. JSON uniquement, sans backticks."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.3,
            )
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(raw)
    except Exception as e:
        raise HTTPException(500, f"Erreur debrief: {str(e)}")
