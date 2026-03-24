from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from groq import Groq
from services.transcription import transcribe_audio
import os, json, re

router = APIRouter(prefix="/speech", tags=["speech"])
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM = """Tu es le plus grand expert mondial en art oratoire, rhétorique classique et éloquence judiciaire.
Tu maîtrises les techniques de Cicéron, De Gaulle, MLK, Churchill, Lincoln, Mandela, Démosthène et Maître Vergès.
Tu réponds UNIQUEMENT en JSON valide sans backticks ni markdown."""

STYLE_RULES = """RÈGLES DE STYLE OBLIGATOIRES :
- Anaphores prophétiques (répète 3 à 5 fois une formule clé)
- Phrases courtes souveraines : "Voilà la vérité. Elle ne se discute pas."
- Accumulations en triades : "nous avons souffert, nous avons lutté, nous avons vaincu"
- Antithèses tranchantes : "Non pas la facilité, mais la grandeur."
- Questions rhétoriques suivies de réponses immédiates
- Gradations ascendantes vers le paroxysme
- Images concrètes et universelles
- Sépare chaque paragraphe par une ligne vide"""

class SpeechTextRequest(BaseModel):
    text: str
    context: str = "general"

def parse_json_safe(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
    return {}

def groq_call(messages: list, max_tokens: int = 3000, temperature: float = 0.3) -> str:
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return r.choices[0].message.content.strip()

@router.post("/analyze-text")
async def analyze_text(req: SpeechTextRequest):
    return await _full_analysis(req.text, req.context)

@router.post("/analyze-audio")
async def analyze_audio_ep(audio: UploadFile = File(...), context: str = Form("general")):
    audio_bytes = await audio.read()
    if not audio_bytes: raise HTTPException(400, "Fichier vide")
    transcript = await transcribe_audio(audio_bytes, audio.filename)
    if not transcript["text"]: raise HTTPException(422, "Transcription impossible")
    result = await _full_analysis(transcript["text"], context)
    result["transcript"] = transcript["text"]
    return result

async def _full_analysis(text: str, context: str) -> dict:
    word_count = len(text.split())
    min_ann    = max(6, word_count // 80)

    # Diviser le texte original en sections pour la réécriture
    words      = text.split()
    third      = len(words) // 3
    part1_text = " ".join(words[:third])
    part2_text = " ".join(words[third:2*third])
    part3_text = " ".join(words[2*third:])
    target_per_part = max(word_count // 3, 200)

    # ── APPEL 1 : Analyse ────────────────────────────────────────────
    raw1 = groq_call([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"""Analyse rhétorique experte ({word_count} mots, contexte: {context}).

DISCOURS:
{text}

JSON :
{{
  "analyse_globale": {{
    "note_globale": 7,
    "note_structure": 6,
    "note_style": 7,
    "note_persuasion": 6,
    "note_rythme": 7,
    "structure_detectee": "description précise de la structure actuelle",
    "resume_critique": "paragraphe critique de 5 phrases constructif"
  }},
  "points_forts": ["Point fort 1 détaillé", "Point fort 2 détaillé", "Point fort 3 détaillé"],
  "faiblesses": ["Faiblesse 1 précise", "Faiblesse 2 précise", "Faiblesse 3 précise"],
  "figures_utilisees": [{{"figure": "nom", "extrait": "extrait", "effet": "effet"}}],
  "figures_manquantes": [{{"figure": "nom", "explication": "où et pourquoi"}}],
  "annotations": [
    {{
      "numero": 1,
      "extrait": "phrase exacte",
      "type": "structure",
      "probleme": "problème précis",
      "suggestion": "suggestion concrète",
      "exemple_ameliore": "version améliorée"
    }}
  ],
  "orateurs_reference": ["orateur proche", "orateur à étudier"]
}}
OBLIGATOIRE: points_forts = 3 strings. faiblesses = 3 strings. annotations = minimum {min_ann} éléments."""},
    ], max_tokens=4000, temperature=0.2)

    analysis = parse_json_safe(raw1)
    if not analysis.get("points_forts"):     analysis["points_forts"]     = ["Analyse non disponible"]
    if not analysis.get("faiblesses"):       analysis["faiblesses"]       = ["Analyse non disponible"]
    if not analysis.get("analyse_globale"):  analysis["analyse_globale"]  = {"note_globale":5,"note_structure":5,"note_style":5,"note_persuasion":5,"note_rythme":5,"structure_detectee":"","resume_critique":""}
    if not analysis.get("annotations"):      analysis["annotations"]      = []
    if not analysis.get("figures_utilisees"):analysis["figures_utilisees"]= []
    if not analysis.get("orateurs_reference"):analysis["orateurs_reference"]=[]

    # ── APPEL 2a : Réécriture EXORDE + NARRATION ────────────────────
    raw2a = groq_call([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"""Tu réécris la première partie de ce discours.
Contexte : {context}. Cible : environ {target_per_part} mots pour cette partie.

PREMIÈRE PARTIE DU DISCOURS ORIGINAL :
{part1_text}

{STYLE_RULES}

Réécris en JSON :
{{
  "exorde": "EXORDE\\n\\n[accroche foudroyante — question rhétorique ou fait saisissant — {target_per_part // 3} mots minimum]",
  "narration": "NARRATION\\n\\n[mise en contexte sobre et précise — {target_per_part // 3} mots minimum]"
}}
Chaque section doit faire au minimum {target_per_part // 3} mots."""},
    ], max_tokens=2500, temperature=0.5)

    part_a = parse_json_safe(raw2a)
    exorde   = part_a.get("exorde",   "EXORDE\n\n[Indisponible]")
    narration= part_a.get("narration","NARRATION\n\n[Indisponible]")

    # ── APPEL 2b : Réécriture CONFIRMATION (développement) ──────────
    raw2b = groq_call([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"""Tu réécris la partie centrale (arguments) de ce discours.
Contexte : {context}. Cible : environ {target_per_part} mots.

PARTIE CENTRALE DU DISCOURS ORIGINAL :
{part2_text}

{STYLE_RULES}

Développe chaque argument original en paragraphe distinct (5 à 8 phrases par argument).
Utilise des anaphores, des triades, des antithèses.

Réécris en JSON :
{{
  "confirmation": "CONFIRMATION\\n\\n[premier argument développé 5-8 phrases]\\n\\n[deuxième argument développé 5-8 phrases]\\n\\n[troisième argument si présent]"
}}
La confirmation doit faire au minimum {target_per_part} mots."""},
    ], max_tokens=2500, temperature=0.5)

    part_b = parse_json_safe(raw2b)
    confirmation = part_b.get("confirmation", "CONFIRMATION\n\n[Indisponible]")

    # ── APPEL 2c : Réécriture RÉFUTATION + PÉRORAISON ───────────────
    raw2c = groq_call([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"""Tu réécris la conclusion de ce discours.
Contexte : {context}. Cible : environ {target_per_part} mots.

DERNIÈRE PARTIE DU DISCOURS ORIGINAL :
{part3_text}

{STYLE_RULES}

La péroraison doit être mémorable, grave le message dans les esprits.
Finir sur une formule ou une image inoubliable.

Réécris en JSON :
{{
  "refutation": "RÉFUTATION\\n\\n[anticipe et détruit l objection principale — 4 à 6 phrases]",
  "peroraison": "PÉRORAISON\\n\\n[conclusion mémorable — formule finale inoubliable — {target_per_part // 2} mots minimum]"
}}"""},
    ], max_tokens=2500, temperature=0.5)

    part_c = parse_json_safe(raw2c)
    refutation = part_c.get("refutation", "RÉFUTATION\n\n[Indisponible]")
    peroraison = part_c.get("peroraison", "PÉRORAISON\n\n[Indisponible]")

    # ── ASSEMBLAGE ───────────────────────────────────────────────────
    version_complete = f"{exorde}\n\n{narration}\n\n{confirmation}\n\n{refutation}\n\n{peroraison}"
    final_word_count = len(version_complete.split())
    print(f"Réécriture : {final_word_count} mots (original : {word_count} mots)")

    # ── APPEL 3 : Métadonnées de réécriture ─────────────────────────
    raw3 = groq_call([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"""En une phrase par item, décris les 5 modifications majeures apportées à ce discours.
Discours original : {word_count} mots. Réécriture : {final_word_count} mots.
Contexte : {context}.

JSON :
{{
  "structure": {{
    "exorde": "technique utilisée dans l exorde",
    "narration": "description de la narration",
    "developpement": "techniques dans le développement",
    "peroraison": "technique de la péroraison"
  }},
  "modifications": [
    "LOGOS : ...",
    "PATHOS : ...",
    "ETHOS : ...",
    "STYLE : ...",
    "STRUCTURE : ..."
  ],
  "techniques_ajoutees": ["anaphore", "chiasme", "triade"],
  "inspiration": "orateurs dont les techniques ont été utilisées"
}}"""},
    ], max_tokens=800, temperature=0.3)

    meta = parse_json_safe(raw3)
    if not meta.get("structure"):    meta["structure"]    = {"exorde":"","narration":"","developpement":"","peroraison":""}
    if not meta.get("modifications"):meta["modifications"]= []

    rewrite = {
        "version_amelioree": version_complete,
        **meta,
    }

    return {**analysis, **rewrite}
