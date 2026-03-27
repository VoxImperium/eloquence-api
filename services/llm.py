# api/services/llm.py
# Feedback coach éloquence via Groq (Llama 3.3 70B) — 100% gratuit

import asyncio
import json
import os
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

CONTEXT_DESCRIPTIONS = {
    "general":   "prise de parole générale (présentation, exposé)",
    "pitch":     "pitch d'entrepreneur devant des investisseurs",
    "entretien": "entretien professionnel ou réunion d'affaires",
}


async def generate_feedback(
    transcript: str,
    metrics:    dict,
    fillers:    dict,
    context:    str = "general"
) -> dict:
    """
    Génère un feedback de coach éloquence via Groq Llama 3.3 70B.
    Gratuit, rapide, et très bon en français.
    """
    context_desc = CONTEXT_DESCRIPTIONS.get(context, CONTEXT_DESCRIPTIONS["general"])

    debit_note = {
        "trop_lent":   "très lent (moins de 80 mots/min)",
        "lent":        "lent (80-120 mots/min)",
        "optimal":     "optimal (120-160 mots/min)",
        "rapide":      "rapide (160-200 mots/min)",
        "trop_rapide": "très rapide (plus de 200 mots/min)",
    }.get(metrics.get("speech_rate_rating", ""), "inconnu")

    pitch_note = {
        "monotone":       "très monotone, peu d'expressivité",
        "peu_expressif":  "peu expressif",
        "expressif":      "bien expressif",
        "très_expressif": "très expressif, voix vivante",
    }.get(metrics.get("pitch_rating", ""), "inconnu")

    filler_list = ", ".join(
        [f'"{w}" ({n} fois)' for w, n in fillers.get("details", {}).items()]
    ) or "aucun mot parasite détecté"

    prompt = f"""Analyse ce discours dans le contexte : {context_desc}

DONNÉES OBJECTIVES :
- Durée : {metrics.get('duration_seconds', 0)} secondes
- Débit : {metrics.get('speech_rate_wpm', 0)} mots/minute — {debit_note}
- Pauses : {metrics.get('pause_count', 0)} pauses (durée moyenne : {metrics.get('avg_pause_s', 0)}s)
- Expressivité vocale : {pitch_note}
- Mots parasites : {filler_list}

TRANSCRIPTION :
{transcript}

Génère une analyse JSON avec exactement cette structure :
{{
  "scores": {{
    "fluidite":    <note de 0 à 10>,
    "structure":   <note de 0 à 10>,
    "vocabulaire": <note de 0 à 10>,
    "rythme":      <note de 0 à 10>,
    "global":      <moyenne arrondie à 1 décimale>
  }},
  "points_forts": ["point fort 1", "point fort 2"],
  "axes_amelioration": ["axe 1", "axe 2"],
  "conseil_prioritaire": "UN seul conseil actionnable immédiatement",
  "resume": "2 phrases max résumant la prestation"
}}"""

    try:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": "Tu es un coach expert en éloquence et prise de parole publique. Tu analyses des discours en français. Tu réponds UNIQUEMENT en JSON valide, sans texte avant ni après, sans backticks, sans markdown."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=1024,
                temperature=0.3,
            )
        )

        raw = response.choices[0].message.content.strip()

        # Nettoyer si backticks présents
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        return json.loads(raw)

    except json.JSONDecodeError as e:
        print(f"Erreur parsing JSON Groq: {e}")
        return _fallback_feedback()

    except Exception as e:
        print(f"Erreur appel Groq: {e}")
        return _fallback_feedback()


def _fallback_feedback() -> dict:
    """Retourné si Groq échoue — évite de bloquer l'utilisateur."""
    return {
        "scores": {
            "fluidite": 0, "structure": 0,
            "vocabulaire": 0, "rythme": 0, "global": 0
        },
        "points_forts":        ["Analyse temporairement indisponible"],
        "axes_amelioration":   ["Réessaie dans quelques instants"],
        "conseil_prioritaire": "Le service d'analyse est temporairement indisponible.",
        "resume":              "Analyse non disponible pour le moment.",
        "error":               True
    }
