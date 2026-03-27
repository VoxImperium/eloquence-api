from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from groq import Groq
from services.transcription import transcribe_audio
import asyncio, os, json, re

router = APIRouter(prefix="/speech", tags=["speech"])
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

THEMIS_SYSTEM = """Tu es "L'Éloquence de Thémis", IA experte en rhétorique classique française, membre de l'Académie française et Avocat à la Cour. Ta spécialité est la "Scansion Juridique" et la transformation de la technique froide en puissance oratoire.

Ta mission : Transformer tout texte oral ou écrit en une plaidoirie ou un discours majestueux, rythmé et prêt pour l'audience.

CONSIGNES DE STYLE IMPÉRATIVES :
1. Lexique Noble : Bannis le jargon moderne ou familier. Remplace les termes vagues par des termes précis et imagés.
   Exemples : "problème" → "achoppement" | "punir" → "frapper du glaive" | "mauvaise foi" → "duplicité" | "expliquer" → "élucider" | "montrer" → "démontrer avec éclat" | "important" → "capital" | "commencer" → "engager"

2. Figures de Style : Intègre OBLIGATOIREMENT au moins TROIS figures classiques parmi :
   - Anaphore : répétition en début de phrase
   - Chiasme : structure en miroir (A-B / B-A)
   - Métaphore filée : image étendue sur plusieurs phrases
   - Parallélisme : construction symétrique
   - Gradation ascendante : montée en puissance
   - Hyperbole noble : amplification majestueuse
   - Périphrase : désignation élégante indirecte

3. Structure de Souffle (Scansion) : Insère des balises de respiration :
   [ / ] pour une respiration courte (virgule de suspension)
   [ // ] pour un silence dramatique de 2 secondes (point d'orgue)

4. Accentuation : Mets en **gras** les mots ou segments sur lesquels l'orateur doit poser sa voix.

Tu réponds UNIQUEMENT en JSON valide sans backticks ni markdown."""

ANALYSIS_SYSTEM = """Tu es un expert mondial en rhétorique classique, analyse oratoire et éloquence.
Tu analyses des discours avec une précision académique et une exigence de haut niveau.
Tu réponds UNIQUEMENT en JSON valide sans backticks ni markdown."""

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

async def groq_async(messages: list, max_tokens: int = 3000, temperature: float = 0.3) -> str:
    loop = asyncio.get_running_loop()
    r = await loop.run_in_executor(
        None,
        lambda: client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
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
    word_count  = len(text.split())
    min_ann     = max(4, word_count // 100)

    # ── APPEL 1 : Analyse rhétorique experte ────────────────────────
    raw1 = await groq_async([
        {"role": "system", "content": ANALYSIS_SYSTEM},
        {"role": "user",   "content": f"""Analyse rhétorique experte et sans concession de ce discours ({word_count} mots, contexte: {context}).

DISCOURS :
{text}

Sois exigeant, précis, académique. Identifie les forces réelles et les faiblesses profondes.

JSON (clés exactes) :
{{
  "analyse_globale": {{
    "note_globale": 7,
    "note_structure": 6,
    "note_style": 7,
    "note_persuasion": 6,
    "note_rythme": 7,
    "structure_detectee": "description précise de la structure actuelle",
    "resume_critique": "analyse critique de 4 à 6 phrases, exigeante et constructive"
  }},
  "points_forts": [
    "Point fort 1 — précis et argumenté avec exemple du texte",
    "Point fort 2 — précis et argumenté avec exemple du texte",
    "Point fort 3 — précis et argumenté avec exemple du texte"
  ],
  "faiblesses": [
    "Faiblesse 1 — précise avec référence au texte et impact oratoire",
    "Faiblesse 2 — précise avec référence au texte et impact oratoire",
    "Faiblesse 3 — précise avec référence au texte et impact oratoire"
  ],
  "figures_utilisees": [
    {{"figure": "nom exact", "extrait": "citation exacte du texte", "effet": "effet rhétorique produit et son impact sur l auditoire"}}
  ],
  "figures_manquantes": [
    {{"figure": "nom", "explication": "pourquoi cette figure manque et où exactement l intégrer"}}
  ],
  "annotations": [
    {{
      "numero": 1,
      "extrait": "phrase exacte du discours",
      "type": "structure|style|rythme|argumentation|figure|lexique",
      "probleme": "problème rhétorique précis et son impact",
      "suggestion": "suggestion concrète et détaillée",
      "exemple_ameliore": "version transformée selon les codes oratoires classiques"
    }}
  ],
  "orateurs_reference": ["orateur historique dont le style est proche", "orateur à étudier pour progresser"]
}}
OBLIGATOIRE: points_forts = 3 strings. faiblesses = 3 strings. annotations = minimum {min_ann} éléments détaillés."""},
    ], max_tokens=4000, temperature=0.2)

    analysis = parse_json_safe(raw1)
    if not analysis.get("points_forts"):      analysis["points_forts"]      = ["Analyse non disponible"]
    if not analysis.get("faiblesses"):        analysis["faiblesses"]        = ["Analyse non disponible"]
    if not analysis.get("analyse_globale"):   analysis["analyse_globale"]   = {"note_globale":5,"note_structure":5,"note_style":5,"note_persuasion":5,"note_rythme":5,"structure_detectee":"","resume_critique":""}
    if not analysis.get("annotations"):       analysis["annotations"]       = []
    if not analysis.get("figures_utilisees"): analysis["figures_utilisees"] = []
    if not analysis.get("orateurs_reference"):analysis["orateurs_reference"]= []

    # ── APPEL 2a : Thémis — Essence du Drame + Tirade ───────────────
    words      = text.split()
    third      = len(words) // 3
    part1_text = " ".join(words[:third])
    part2_text = " ".join(words[third:2*third])
    part3_text = " ".join(words[2*third:])
    target     = max(word_count // 3, 150)

    raw2a = await groq_async([
        {"role": "system", "content": THEMIS_SYSTEM},
        {"role": "user",   "content": f"""Contexte : {context}. Partie 1 du discours ({target} mots cible).

TEXTE ORIGINAL :
{part1_text}

Transforme cette partie en EXORDE et NARRATION oratoires selon les codes de Thémis.
Respecte les balises [ / ] et [ // ] et les **accentuations**.

JSON :
{{
  "essence_du_drame": "Une phrase saisissante expliquant l enjeu humain profond de ce discours",
  "exorde": "EXORDE\\n\\n[texte de l exorde transformé avec balises et accentuations — {target//3} mots minimum]",
  "narration": "NARRATION\\n\\n[texte de la narration transformée — {target//3} mots minimum]"
}}"""},
    ], max_tokens=2500, temperature=0.6)

    part_a = parse_json_safe(raw2a)

    # ── APPEL 2b : Thémis — Confirmation ───────────────────────────
    raw2b = await groq_async([
        {"role": "system", "content": THEMIS_SYSTEM},
        {"role": "user",   "content": f"""Contexte : {context}. Partie centrale du discours.

TEXTE ORIGINAL :
{part2_text}

Transforme en CONFIRMATION oratoire majestueuse.
Utilise OBLIGATOIREMENT : anaphore + gradation ascendante + au moins une métaphore filée.
Chaque argument = un paragraphe distinct avec balises de souffle.
Cible : {target} mots minimum.

JSON :
{{
  "confirmation": "CONFIRMATION\\n\\n[premier argument transformé — 5 à 8 phrases avec balises]\\n\\n[deuxième argument]\\n\\n[troisième argument si présent]"
}}"""},
    ], max_tokens=2500, temperature=0.6)

    part_b = parse_json_safe(raw2b)

    # ── APPEL 2c : Thémis — Réfutation + Péroraison ────────────────
    raw2c = await groq_async([
        {"role": "system", "content": THEMIS_SYSTEM},
        {"role": "user",   "content": f"""Contexte : {context}. Conclusion du discours.

TEXTE ORIGINAL :
{part3_text}

Transforme en RÉFUTATION et PÉRORAISON dignes des plus grands orateurs.
La péroraison doit être mémorable, gravée dans les esprits.
Finir sur une formule ou une image inoubliable.
Utilise le chiasme dans la réfutation. Termine par une anaphore dans la péroraison.

JSON :
{{
  "refutation": "RÉFUTATION\\n\\n[réfutation avec chiasme — 4 à 6 phrases avec balises [ // ] avant chaque point fort]",
  "peroraison": "PÉRORAISON\\n\\n[péroraison avec anaphore finale — formule conclusive inoubliable]"
}}"""},
    ], max_tokens=2500, temperature=0.6)

    part_c = parse_json_safe(raw2c)

    # ── ASSEMBLAGE DE LA TIRADE ─────────────────────────────────────
    exorde      = part_a.get("exorde",      "EXORDE\n\n[Non généré]")
    narration   = part_a.get("narration",   "NARRATION\n\n[Non généré]")
    confirmation= part_b.get("confirmation","CONFIRMATION\n\n[Non généré]")
    refutation  = part_c.get("refutation",  "RÉFUTATION\n\n[Non généré]")
    peroraison  = part_c.get("peroraison",  "PÉRORAISON\n\n[Non généré]")
    essence     = part_a.get("essence_du_drame", "L'enjeu humain de ce discours.")

    tirade_complete = f"{exorde}\n\n{narration}\n\n{confirmation}\n\n{refutation}\n\n{peroraison}"
    final_words = len(tirade_complete.split())

    # ── APPEL 3 : Note de l'Expert ──────────────────────────────────
    raw3 = await groq_async([
        {"role": "system", "content": THEMIS_SYSTEM},
        {"role": "user",   "content": f"""En tant que Thémis, rédige une note d expert sur cette transformation oratoire.
Discours original : {word_count} mots → Tirade : {final_words} mots.
Contexte : {context}.

JSON :
{{
  "note_expert": "Paragraphe de 3 à 5 phrases expliquant les choix rhétoriques majeurs, les figures de style utilisées et leur effet sur l auditoire",
  "structure": {{
    "exorde": "technique et effet de l exorde",
    "narration": "technique de la narration",
    "developpement": "techniques du développement — figures utilisées",
    "peroraison": "technique et impact de la péroraison"
  }},
  "figures_integrees": ["liste des figures de style intégrées dans la tirade"],
  "modifications": [
    "LEXIQUE : transformation lexicale principale",
    "FIGURES : figures de style ajoutées et leurs effets",
    "RYTHME : travail sur le souffle et les silences",
    "STRUCTURE : restructuration opérée",
    "IMPACT : effet attendu sur l auditoire"
  ],
  "inspiration": "Orateurs dont les techniques ont été utilisées (Cicéron, De Gaulle, MLK...)"
}}"""},
    ], max_tokens=1000, temperature=0.4)

    meta = parse_json_safe(raw3)

    rewrite = {
        "essence_du_drame":  essence,
        "tirade_oratoire":   tirade_complete,
        "version_amelioree": tirade_complete,  # compatibilité frontend
        "note_expert":       meta.get("note_expert", ""),
        "structure":         meta.get("structure", {"exorde":"","narration":"","developpement":"","peroraison":""}),
        "figures_integrees": meta.get("figures_integrees", []),
        "modifications":     meta.get("modifications", []),
        "inspiration":       meta.get("inspiration", ""),
        "techniques_ajoutees": meta.get("figures_integrees", []),
    }

    return {**analysis, **rewrite}
