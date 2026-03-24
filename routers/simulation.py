# api/routers/simulation.py
# Simulation IA — 5 scénarios de prise de parole

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from groq import Groq
import os
import json

router = APIRouter(prefix="/simulate", tags=["simulation"])
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── Personas IA par scénario ─────────────────────────────────────────
PERSONAS = {
    "pitch_investisseur": {
        "name": "Marc Dubois",
        "role": "Investisseur VC senior",
        "description": "pitch devant un investisseur venture capital",
        "system": """Tu es Marc Dubois, un investisseur VC senior avec 20 ans d'expérience.
Tu poses des questions difficiles sur le modèle économique, la concurrence, la traction.
Tu es sceptique mais juste. Tu poses UNE seule question courte à la fois.
Tu parles français, tu es direct et exigeant. Maximum 2 phrases par réponse."""
    },
    "entretien_embauche": {
        "name": "Sophie Martin",
        "role": "DRH Cabinet de conseil",
        "description": "entretien d'embauche dans un cabinet de conseil",
        "system": """Tu es Sophie Martin, DRH d'un cabinet de conseil top tier.
Tu évalues les compétences comportementales, la clarté d'expression, la logique.
Tu poses des questions STAR (Situation, Tâche, Action, Résultat).
Tu poses UNE seule question à la fois. Maximum 2 phrases. Tu parles français."""
    },
    "debat_contradictoire": {
        "name": "Thomas Laurent",
        "role": "Débatteur contradicteur",
        "description": "débat contradictoire sur un sujet de société",
        "system": """Tu es Thomas Laurent, débatteur expérimenté.
Tu prends systématiquement la position OPPOSÉE à celle de l'utilisateur.
Tu argues avec logique, tu cites des faits, tu poses des questions rhétoriques.
Tu es incisif mais respectueux. Maximum 3 phrases. Tu parles français."""
    },
    "reunion_client": {
        "name": "Isabelle Morel",
        "role": "Directrice Grands Comptes",
        "description": "réunion client avec une directrice grands comptes",
        "system": """Tu es Isabelle Morel, directrice grands comptes d'une grande entreprise.
Tu as peu de temps, tu veux des réponses précises et concrètes.
Tu poses des objections sur le prix, les délais, les garanties.
Tu poses UNE objection ou question à la fois. Maximum 2 phrases. Tu parles français."""
    },
    "plaidoirie": {
        "name": "Le Président",
        "role": "Président du jury",
        "description": "plaidoirie devant un jury",
        "system": """Tu es le Président d'un jury de concours d'éloquence.
L'utilisateur plaide pour ou contre une proposition. Tu écoutes, puis tu poses
des questions pour tester la solidité des arguments, les contradictions, la rhétorique.
Tu es solennel et précis. Maximum 2 phrases. Tu parles français."""
    }
}


class SimulationMessage(BaseModel):
    scenario:   str
    messages:   list
    user_input: str
    topic:      str = ""


class SimulationDebrief(BaseModel):
    scenario:  str
    messages:  list
    topic:     str = ""


@router.post("/message")
async def simulate_message(req: SimulationMessage):
    """
    Envoie un message utilisateur et reçoit la réponse de l'IA.
    """
    persona = PERSONAS.get(req.scenario)
    if not persona:
        raise HTTPException(400, f"Scénario inconnu: {req.scenario}")

    # Construire l'historique
    history = []
    for msg in req.messages[-10:]:  # Garder les 10 derniers messages
        history.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    # Ajouter le message utilisateur
    history.append({"role": "user", "content": req.user_input})

    # Contexte du sujet si fourni
    system = persona["system"]
    if req.topic:
        system += f"\n\nSujet de la session : {req.topic}"

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system},
                *history
            ],
            max_tokens=200,
            temperature=0.7,
        )

        ai_response = response.choices[0].message.content.strip()

        return {
            "response":    ai_response,
            "persona_name": persona["name"],
            "persona_role": persona["role"],
        }

    except Exception as e:
        raise HTTPException(500, f"Erreur simulation: {str(e)}")


@router.post("/debrief")
async def simulation_debrief(req: SimulationDebrief):
    """
    Génère un debrief complet de la simulation.
    """
    persona = PERSONAS.get(req.scenario)
    if not persona:
        raise HTTPException(400, f"Scénario inconnu: {req.scenario}")

    # Reconstruire la conversation
    conversation = "\n".join([
        f"{'Utilisateur' if m['role'] == 'user' else persona['name']}: {m['content']}"
        for m in req.messages
    ])

    prompt = f"""Tu as joué le rôle de {persona['name']} ({persona['role']}) dans une simulation de {persona['description']}.

Voici la conversation complète :
{conversation}

Génère un debrief JSON de la performance de l'utilisateur :
{{
  "note_globale": <0-10>,
  "points_forts": ["point 1", "point 2"],
  "points_faibles": ["point 1", "point 2"],
  "meilleure_replique": "citation exacte de la meilleure réplique de l'utilisateur",
  "conseil_principal": "UN conseil actionnable pour progresser",
  "resume": "2 phrases résumant la performance"
}}"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Tu es un coach expert en éloquence. Réponds UNIQUEMENT en JSON valide sans backticks."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.3,
        )

        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        return json.loads(raw)

    except Exception as e:
        raise HTTPException(500, f"Erreur debrief: {str(e)}")


@router.get("/scenarios")
async def get_scenarios():
    """Liste tous les scénarios disponibles."""
    return [
        {
            "id":          key,
            "name":        val["name"],
            "role":        val["role"],
            "description": val["description"],
        }
        for key, val in PERSONAS.items()
    ]
