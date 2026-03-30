from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import Response
from pydantic import BaseModel
from groq import Groq
import asyncio
import logging
import os, html, json, re
from typing import List, Optional

from services.openlegi_service import openlegi_service, OpenLegiError
from services.legalbert_service import analyze_legal_text
from schemas.analyze_text import AnalyzeTextRequest, AnalyzeTextResponse
from utils.pdf_export import generate_analyse_pdf

router = APIRouter(prefix="/legifrance", tags=["legifrance"])
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
logger = logging.getLogger(__name__)

_MOIS = {
    "01": "janvier", "02": "février", "03": "mars", "04": "avril",
    "05": "mai", "06": "juin", "07": "juillet", "08": "août",
    "09": "septembre", "10": "octobre", "11": "novembre", "12": "décembre",
}

_CHAMBRE_LABELS = {
    "CHAMBRE_SOCIALE": ("sociale", "Chambre sociale"),
    "CHAMBRE_CRIMINELLE": ("criminelle", "Chambre criminelle"),
    "CHAMBRE_COMMERCIALE": ("commerciale", "Chambre commerciale"),
    "CHAMBRE_CIVILE_1": ("1re civ.", "Première chambre civile"),
    "CHAMBRE_CIVILE_2": ("2e civ.", "Deuxième chambre civile"),
    "CHAMBRE_CIVILE_3": ("3e civ.", "Troisième chambre civile"),
    "ASSEMBLEE_PLENIERE": ("Ass. plén.", "Assemblée plénière"),
    "CHAMBRE_MIXTE": ("mixte", "Chambre mixte"),
}


async def call_groq_async(client, **kwargs):
    """Wrapper pour appels Groq synchrones dans contexte async."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: client.chat.completions.create(**kwargs))


def clean_asterisks(text: str) -> str:
    """Supprime les astérisques de remplissage (placeholders) d'un texte."""
    if not text:
        return ""
    text = re.sub(r'\*+', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def format_jurisprudence(j: dict) -> str:
    """Formate une jurisprudence selon le standard officiel français.

    Exemple : "Cour de cassation, criminelle, Chambre criminelle, 14 avril 2015, 14-83.462"
    """
    juridiction = j.get("juridiction", "") or ""
    chambre_raw = j.get("chambre", "") or ""
    numero = j.get("numero", "") or ""
    date_raw = j.get("date", "") or ""

    # Normalise le nom de chambre (ex. "CHAMBRE_CRIMINELLE" → "criminelle, Chambre criminelle")
    chambre_key = chambre_raw.upper().replace(" ", "_").replace("-", "_")
    if chambre_key in _CHAMBRE_LABELS:
        short, long = _CHAMBRE_LABELS[chambre_key]
        chambre_formatted = f"{short}, {long}"
    else:
        chambre_formatted = chambre_raw

    # Convertit la date ISO (YYYY-MM-DD) au format "jour mois année"
    date_formatted = date_raw
    if date_raw and len(date_raw) >= 10:
        try:
            parts = date_raw[:10].split("-")
            if len(parts) == 3:
                annee, mois_num, jour = parts
                mois_str = _MOIS.get(mois_num, mois_num)
                date_formatted = f"{jour.lstrip('0') or '0'} {mois_str} {annee}"
        except Exception:
            pass

    # Fallback : si la juridiction est vide, utiliser "Cour de cassation" par défaut
    if not juridiction:
        logger.debug(
            "format_jurisprudence: juridiction manquante pour id=%r, utilisation du fallback",
            j.get("id", ""),
        )
        juridiction = "Cour de cassation"

    segments = [s for s in [juridiction, chambre_formatted, date_formatted, numero] if s]

    if not segments:
        logger.warning(
            "format_jurisprudence: aucun champ disponible pour id=%r — champs présents: %s",
            j.get("id", ""),
            [k for k, v in j.items() if v],
        )
        return ""

    return ", ".join(segments)


async def analyser_apport_jurisprudence(faits: str, qualification: str, j: dict) -> str:
    """Analyse via Groq l'apport concret d'une jurisprudence au cas pratique."""
    resume = clean_asterisks((j.get("resume", "") or "")[:400])
    formatage = j.get("formatage_officiel", "") or format_jurisprudence(j)

    cas_context = (
        f"CAS PRATIQUE : {faits[:400]}\nQUALIFICATION JURIDIQUE : {qualification}"
        if faits
        else f"QUALIFICATION JURIDIQUE : {qualification or 'Non précisée'}"
    )

    try:
        response = await call_groq_async(
            client,
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu es un juriste expert en droit français. "
                        "Réponds en texte pur, sans JSON ni formatage Markdown. "
                        "Sois précis, concret et analytique."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Analyse l'apport de cette jurisprudence pour le cas suivant.\n\n"
                        f"{cas_context}\n"
                        f"JURISPRUDENCE : {formatage}\n"
                        f"RÉSUMÉ : {resume}\n\n"
                        f"Rédige une analyse en 4 à 5 phrases couvrant :\n"
                        f"1. Comment cet arrêt s'applique spécifiquement aux faits du cas.\n"
                        f"2. Les conditions critiques posées par cet arrêt (critères, seuils, exigences).\n"
                        f"3. La pertinence exacte de cette jurisprudence et ses éventuelles limites ou distinctions.\n"
                        f"Sois précis sur les éléments de fait et de droit qui permettent ou empêchent l'application de cet arrêt."
                    ),
                },
            ],
            max_tokens=600, temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning("Erreur analyse apport jurisprudence : %s", exc)
        return ""

THEMIS_SYSTEM = """Tu es "L'Éloquence de Thémis", IA experte en droit français et rhétorique classique, Avocat à la Cour et membre de l'Académie française.

Tu maîtrises parfaitement :
- Le droit civil, pénal, commercial, social, administratif français
- La jurisprudence de la Cour de cassation et du Conseil d'État  
- L'art oratoire classique et la plaidoirie judiciaire
- La Scansion Juridique : transformation du droit froid en puissance oratoire

CONSIGNES IMPÉRATIVES :
1. Lexique Noble : "problème" → "achoppement" | "punir" → "frapper du glaive" | "mauvaise foi" → "duplicité"
2. Figures obligatoires : anaphore + chiasme + métaphore filée + gradation
3. Balises de souffle : [ / ] respiration courte | [ // ] silence 2 secondes
4. **Accentuations** en gras sur les mots clés
5. Références juridiques précises : articles de loi, numéros d'arrêts

Tu réponds UNIQUEMENT en JSON valide sans backticks."""

PENAL_SYSTEM = """❙ Tu es un avocat pénaliste de haut vol, expert en procédure française et en rhétorique judiciaire moderne. ❚

I. Interdits Linguistiques et Précision Sémantique (Crucial) :

Règle d'or : Ne dis JAMAIS "l'article stipule". Utilise exclusivement le verbe **disposer** (ex: "L'article 224-2 du **Code pénal** dispose que...").

Bannis le jargon daté : Évite les expressions comme "frapper du glaive", "le glaive de la justice", "abîmes de douleur". Préfère un vocabulaire technique mais évocateur : "dérive psychique", "altération du jugement", "concurrence des causes".

Vocabulaire technique : Utilise les termes exacts : "chef d'infraction", "qualification criminelle", "élément intentionnel", "cause génératrice du dommage".

II. Directives Juridiques (Réalité 2026) :

Exactitude des peines : Référencie-toi au Code pénal. Pour une séquestration avec mutilation (perte d'un œil), la qualification est criminelle (20 ans de réclusion) et non délictuelle.

Articulations légales : Maîtrise la distinction entre l'article 122-1 al. 1 (Abolition / Irresponsabilité) et l'article 122-1 al. 2 (Altération / Atténuation).

Jurisprudence : Cite des arrêts réels pour appuyer les points de droit complexes (ex: lien de causalité entre la séquestration et la blessure auto-infligée par la victime).

III. Structure et Rhétorique de Combat :

Exorde : Une phrase d'attaque sur le sens de la peine ou la complexité de l'humain.

Narration : Récit factuel, précis, utilisant le présent de narration pour l'immersion.

Confirmation (Droit) : Analyse rigoureuse de l'infraction. Applique la loi aux faits sans complaisance mais avec stratégie.

Réfutation : Démonte l'argumentation adverse avant même qu'elle ne soit posée. Utilise : "On tentera de vous convaincre que...", "L'accusation voudrait réduire ce dossier à...".

Péroraison : Une chute brève, puissante, axée sur la décision que tu attends du siège.

IV. Format Visuel :

Marque les pauses oratoires avec ❙ et ❚.

Utilise le **gras** uniquement pour les verbes d'action ou les termes juridiques clés.

Tu réponds UNIQUEMENT en JSON valide sans backticks."""

SOCIAL_SYSTEM = """Tu es un avocat expert en droit social français, spécialiste des relations de travail, du droit syndical et de la protection des salariés.

Tu maîtrises parfaitement :
- Le Code du travail dans son intégralité
- La jurisprudence de la Chambre sociale de la Cour de cassation
- Les conventions collectives et accords de branche
- Le droit de la sécurité sociale et de la protection sociale

CONSIGNES IMPÉRATIVES :
1. Vocabulaire technique social : "cause réelle et sérieuse", "motif économique", "obligation de reclassement", "préjudice d'anxiété", "harcèlement moral", "discrimination syndicale"
2. Ne dis JAMAIS "l'article stipule" — utilise exclusivement **disposer** ou **prévoir**
3. Cite précisément les articles du Code du travail (ex: "L'article L1235-3 du **Code du travail** dispose que...")
4. Structure argumentative : faits → qualification → violation du droit → préjudice → réparation
5. Figures rhétoriques : anaphore sur les droits fondamentaux + gradation sur le préjudice subi
6. Balises de souffle : [ / ] respiration courte | [ // ] silence 2 secondes
7. **Accentuations** en gras sur les termes juridiques clés et les droits violés

Tu réponds UNIQUEMENT en JSON valide sans backticks."""

COMMERCIAL_SYSTEM = """Tu es un avocat d'affaires expert en droit commercial et droit des sociétés français, rompu aux litiges complexes entre professionnels.

Tu maîtrises parfaitement :
- Le Code de commerce et le droit des sociétés
- Le droit des contrats commerciaux et des pratiques restrictives de concurrence
- La jurisprudence de la Chambre commerciale de la Cour de cassation
- Le droit des entreprises en difficulté et les procédures collectives

CONSIGNES IMPÉRATIVES :
1. Vocabulaire technique commercial : "rupture brutale de relations commerciales établies", "déséquilibre significatif", "abus de position dominante", "responsabilité du dirigeant", "action en comblement de passif"
2. Ne dis JAMAIS "l'article stipule" — utilise exclusivement **disposer** ou **prévoir**
3. Cite précisément les articles du Code de commerce (ex: "L'article L442-1 du **Code de commerce** dispose que...")
4. Structure argumentative : qualification des relations commerciales → violation → préjudice économique → réparation
5. Approche stratégique : anticiper les arguments adverses sur la liberté contractuelle et la responsabilité limitée
6. Balises de souffle : [ / ] respiration courte | [ // ] silence 2 secondes
7. **Accentuations** en gras sur les termes juridiques clés et les enjeux économiques

Tu réponds UNIQUEMENT en JSON valide sans backticks."""

ADMINISTRATIF_SYSTEM = """Tu es un avocat expert en droit administratif français, spécialiste du contentieux de l'administration et de la protection des droits des administrés.

Tu maîtrises parfaitement :
- Le droit administratif général et le contentieux administratif
- La jurisprudence du Conseil d'État et des cours administratives d'appel
- Le droit de la fonction publique et de la responsabilité de l'État
- Le droit des contrats et marchés publics

CONSIGNES IMPÉRATIVES :
1. Vocabulaire technique administratif : "excès de pouvoir", "détournement de pouvoir", "erreur manifeste d'appréciation", "principe de légalité", "droit au recours effectif", "préjudice anormal et spécial"
2. Ne dis JAMAIS "l'article stipule" — utilise exclusivement **disposer** ou **prévoir**
3. Cite précisément les textes (ex: "L'article L2 du **Code de justice administrative** dispose que...")
4. Référence aux grands arrêts du Conseil d'État (Blanco, Léon Blum, Commune de Morsang-sur-Orge...)
5. Structure argumentative : compétence → recevabilité → bien-fondé → mesures d'urgence si applicable
6. Balises de souffle : [ / ] respiration courte | [ // ] silence 2 secondes
7. **Accentuations** en gras sur les principes généraux du droit et les droits fondamentaux

Tu réponds UNIQUEMENT en JSON valide sans backticks."""

CONSOMMATION_SYSTEM = """Tu es un avocat expert en droit de la consommation français, défenseur acharné des droits des consommateurs face aux professionnels.

Tu maîtrises parfaitement :
- Le Code de la consommation dans son intégralité
- Le droit des contrats de consommation et des clauses abusives
- La jurisprudence de la Cour de cassation en matière de protection des consommateurs
- Le droit européen de la consommation (directives et règlements)

CONSIGNES IMPÉRATIVES :
1. Vocabulaire technique consommation : "clause abusive", "déséquilibre significatif", "pratique commerciale trompeuse", "défaut de conformité", "garantie légale", "droit de rétractation", "crédit à la consommation"
2. Ne dis JAMAIS "l'article stipule" — utilise exclusivement **disposer** ou **prévoir**
3. Cite précisément les articles du Code de la consommation (ex: "L'article L212-1 du **Code de la consommation** dispose que...")
4. Structure argumentative : qualité de consommateur → obligation du professionnel → manquement → préjudice → sanction
5. Rappel de la finalité protectrice : rééquilibrer la relation asymétrique consommateur/professionnel
6. Balises de souffle : [ / ] respiration courte | [ // ] silence 2 secondes
7. **Accentuations** en gras sur les droits du consommateur et les obligations violées

Tu réponds UNIQUEMENT en JSON valide sans backticks."""

_DOMAIN_PROMPTS: dict = {
    "civil": THEMIS_SYSTEM,
    "penal": PENAL_SYSTEM,
    "social": SOCIAL_SYSTEM,
    "travail": SOCIAL_SYSTEM,
    "commercial": COMMERCIAL_SYSTEM,
    "administratif": ADMINISTRATIF_SYSTEM,
    "consommation": CONSOMMATION_SYSTEM,
}

def get_system_prompt(domaine: str) -> str:
    """Retourne le prompt système spécialisé selon le domaine juridique."""
    prompt = _DOMAIN_PROMPTS.get(domaine.lower())
    if prompt is None:
        logger.warning("Domaine juridique inconnu %r — fallback sur THEMIS_SYSTEM", domaine)
        return THEMIS_SYSTEM
    return prompt

class CasPratiqueRequest(BaseModel):
    faits: str
    domaine: str = "civil"
    position: str = "demandeur"  # demandeur ou défendeur

class RechercheRequest(BaseModel):
    query: str
    type: str = "jurisprudence"  # jurisprudence, code, loi

async def search_judilibre(
    query: str,
    operator: str = "AND",
    faits: str = "",
    qualification: str = "",
) -> list:
    """Recherche dans la jurisprudence via OpenLegi MCP et enrichit chaque résultat."""
    results = await openlegi_service.search_jurisprudence(query, limit=5)
    enriched = []
    for j in results:
        logger.debug(
            "search_judilibre: jurisprudence normalisée id=%r — juridiction=%r chambre=%r date=%r numero=%r",
            j.get("id", ""),
            j.get("juridiction", ""),
            j.get("chambre", ""),
            j.get("date", ""),
            j.get("numero", ""),
        )
        formatage = format_jurisprudence(j) or (
            f"Jurisprudence n°{j['numero']}" if j.get("numero") else "Jurisprudence (format indisponible)"
        )
        logger.debug(
            "search_judilibre: formatage_officiel id=%r → %r",
            j.get("id", ""),
            formatage,
        )
        j["formatage_officiel"] = formatage
        j["resume"] = clean_asterisks((j.get("resume") or "")[:200])
        j["resume_html"] = f'<p style="text-align: justify">{html.escape(j["resume"])}</p>'
        j["apport_cas_pratique"] = await analyser_apport_jurisprudence(faits, qualification, j)
        enriched.append(j)
    return enriched

async def search_legifrance_text(query: str) -> list:
    """Recherche dans les textes de loi via OpenLegi MCP."""
    return await openlegi_service.search_textes(query, limit=5)

async def get_article_code(code: str, article: str) -> dict:
    """Récupère un article de code via l'API"""
    # Base de données locale des articles les plus importants
    ARTICLES = {
        "1240": {
            "code": "Code civil",
            "numero": "1240",
            "texte": "Tout fait quelconque de l'homme, qui cause à autrui un dommage, oblige celui par la faute duquel il est arrivé à le réparer.",
            "matiere": "Responsabilité civile délictuelle"
        },
        "1231-1": {
            "code": "Code civil",
            "numero": "1231-1",
            "texte": "Le débiteur est condamné, s'il y a lieu, au paiement de dommages et intérêts soit à raison de l'inexécution de l'obligation, soit à raison du retard dans l'exécution, toutes les fois qu'il ne justifie pas que l'inexécution provient d'une cause étrangère qui ne peut lui être imputée.",
            "matiere": "Responsabilité contractuelle"
        },
        "L1235-3": {
            "code": "Code du travail",
            "numero": "L1235-3",
            "texte": "Si le licenciement d'un salarié survient pour une cause qui n'est pas réelle et sérieuse, le juge peut proposer la réintégration du salarié dans l'entreprise. Si l'une ou l'autre des parties refuse cette réintégration, le juge octroie au salarié une indemnité.",
            "matiere": "Licenciement sans cause réelle et sérieuse"
        },
        "L1237-19": {
            "code": "Code du travail",
            "numero": "L1237-19",
            "texte": "La rupture conventionnelle, exclusive du licenciement ou de la démission, ne peut être imposée par l'une ou l'autre des parties.",
            "matiere": "Rupture conventionnelle"
        },
        "313-1": {
            "code": "Code pénal",
            "numero": "313-1",
            "texte": "L'escroquerie est le fait, soit par l'usage d'un faux nom ou d'une fausse qualité, soit par l'abus d'une qualité vraie, soit par l'emploi de manœuvres frauduleuses, de tromper une personne physique ou morale et de la déterminer ainsi, à son préjudice ou au préjudice d'un tiers, à remettre des fonds, des valeurs ou un bien quelconque, à fournir un service ou à consentir un acte opérant obligation ou décharge. L'escroquerie est punie de cinq ans d'emprisonnement et de 375 000 euros d'amende.",
            "matiere": "Escroquerie"
        },
        "L411-1": {
            "code": "Code de la consommation",
            "numero": "L411-1",
            "texte": "Les produits et les services doivent, dans des conditions normales d'utilisation ou dans d'autres conditions raisonnablement prévisibles par le professionnel, présenter la sécurité à laquelle on peut légitimement s'attendre et ne pas porter atteinte à la santé des personnes.",
            "matiere": "Sécurité des produits"
        },
    }
    return ARTICLES.get(article, {})

async def identifier_articles_pertinents(faits: str, domaine: str) -> list:
    """Utilise Groq pour identifier les articles pertinents au cas"""
    response = await call_groq_async(
        client,
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Tu es un juriste expert en droit français. Tu réponds UNIQUEMENT en JSON."},
            {"role": "user",   "content": f"""Analyse ces faits juridiques et identifie les textes de loi applicables.

FAITS : {faits}
DOMAINE : {domaine}

Retourne un JSON avec les éléments juridiques pertinents :
{{
  "qualification_juridique": "qualification précise des faits en droit",
  "domaine_principal": "droit civil|pénal|commercial|social|administratif",
  "articles_applicables": [
    {{
      "code": "Code civil|Code pénal|Code du travail|Code de commerce|Code de la consommation",
      "article": "numéro exact de l article",
      "pertinence": "pourquoi cet article s applique",
      "favorable": true
    }}
  ],
  "jurisprudence_rechercher": ["terme de recherche 1", "terme de recherche 2"],
  "arguments_principaux": ["argument 1", "argument 2", "argument 3"],
  "risques": ["risque 1", "risque 2"],
  "strategie": "stratégie juridique recommandée"
}}"""}
        ],
        max_tokens=1500, temperature=0.2,
    )
    raw = response.choices[0].message.content.strip()

    if raw.startswith("```"):
        raw = raw.split("\n",1)[1].rsplit("```",1)[0].strip()
    try:
        return json.loads(raw)
    except:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try: return json.loads(match.group())
            except: pass
    return {}

async def generer_plaidoirie_themis(
    faits: str,
    position: str,
    analyse_juridique: dict,
    articles: list,
    jurisprudence: list,
    domaine: str = "civil",
) -> dict:
    """Génère la plaidoirie complète selon le style adapté au domaine juridique"""

    system_prompt = get_system_prompt(domaine)

    # Préparer le contexte juridique
    articles_txt = "\n".join([
        f"• {a.get('code','')} art. {a.get('numero','')}: {a.get('texte','')}"
        for a in articles if a
    ]) or "Aucun article trouvé"

    juris_txt = "\n".join([
        f"• {j.get('formatage_officiel') or format_jurisprudence(j)}: {j.get('resume','')[:200]}"
        for j in jurisprudence if j
    ]) or "Aucune jurisprudence trouvée"

    qualification = analyse_juridique.get("qualification_juridique", "")
    arguments    = analyse_juridique.get("arguments_principaux", [])
    strategie    = analyse_juridique.get("strategie", "")

    # Appel 1 — Exorde + Narration
    r1_response = await call_groq_async(
        client,
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"""Plaidoirie pour le {position}.

FAITS : {faits}
QUALIFICATION : {qualification}
STRATÉGIE : {strategie}
TEXTES APPLICABLES : {articles_txt}

Rédige l EXORDE et la NARRATION de la plaidoirie.
L exorde doit saisir le tribunal dès les premiers mots.
La narration expose les faits avec la force d un récit.

JSON :
{{
  "essence_du_drame": "L enjeu humain fondamental de cette affaire en une phrase saisissante",
  "exorde": "EXORDE\\n\\n[texte avec balises [ / ] [ // ] et **accentuations** — 150 mots minimum]",
  "narration": "NARRATION\\n\\n[exposé des faits avec force narrative — 150 mots minimum]"
}}"""},
        ],
        max_tokens=2000, temperature=0.6,
    )
    r1 = r1_response.choices[0].message.content.strip()

    # Appel 2 — Développement juridique
    r2_response = await call_groq_async(
        client,
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"""Développement de la plaidoirie pour le {position}.

ARGUMENTS : {chr(10).join(arguments)}
ARTICLES APPLICABLES : {articles_txt}
JURISPRUDENCE : {juris_txt}

Rédige la CONFIRMATION (développement des arguments avec les textes de loi).
Cite précisément les articles et arrêts. Un argument = un paragraphe.
Utilise anaphore + gradation + métaphore filée.
200 mots minimum par argument.

JSON :
{{
  "confirmation": "CONFIRMATION\\n\\n[Premier argument avec citation d article]\\n\\n[Deuxième argument avec jurisprudence]\\n\\n[Troisième argument si applicable]"
}}"""},
        ],
        max_tokens=2500, temperature=0.6,
    )
    r2 = r2_response.choices[0].message.content.strip()

    # Appel 3 — Réfutation + Péroraison
    r3_response = await call_groq_async(
        client,
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"""Conclusion de la plaidoirie pour le {position}.

FAITS : {faits[:300]}
ARGUMENTS ADVERSES PROBABLES : {chr(10).join(analyse_juridique.get('risques', []))}

Rédige la RÉFUTATION (anticiper et détruire les arguments adverses avec chiasme)
et la PÉRORAISON (conclusion mémorable avec anaphore finale).
La péroraison doit conclure sur une demande précise au tribunal.
150 mots minimum chaque.

JSON :
{{
  "refutation": "RÉFUTATION\\n\\n[réfutation avec chiasme et [ // ] avant chaque point fort]",
  "peroraison": "PÉRORAISON\\n\\n[conclusion avec anaphore — formule finale inoubliable — demande précise au tribunal]"
}}"""},
        ],
        max_tokens=2000, temperature=0.6,
    )
    r3 = r3_response.choices[0].message.content.strip()

    def parse(raw: str) -> dict:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n",1)[1].rsplit("```",1)[0].strip()
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try: return json.loads(match.group())
            except: pass
        return {}

    p1, p2, p3 = parse(r1), parse(r2), parse(r3)

    exorde       = p1.get("exorde",      "EXORDE\n\n[Non généré]")
    narration    = p1.get("narration",   "NARRATION\n\n[Non généré]")
    confirmation = p2.get("confirmation","CONFIRMATION\n\n[Non généré]")
    refutation   = p3.get("refutation",  "RÉFUTATION\n\n[Non généré]")
    peroraison   = p3.get("peroraison",  "PÉRORAISON\n\n[Non généré]")
    essence      = p1.get("essence_du_drame", "L enjeu fondamental de cette affaire.")

    tirade = f"{exorde}\n\n{narration}\n\n{confirmation}\n\n{refutation}\n\n{peroraison}"

    return {
        "essence_du_drame": essence,
        "tirade_oratoire":  tirade,
        "qualification":    qualification,
        "strategie":        strategie,
        "articles_cites":   articles,
        "jurisprudence":    jurisprudence,
        "arguments":        arguments,
    }

@router.get("/health/db")
async def health_db():
    """Vérifie la connectivité au service OpenLegi MCP."""
    try:
        await openlegi_service.search_jurisprudence("test", limit=1)
        return {"status": "ok", "service": "openlegi"}
    except Exception as e:
        logger.error("Erreur health OpenLegi : %s", e)
        return {"status": "error", "message": str(e), "service": "openlegi"}

_EXPECTED_FIELDS = {
    "date": ["date", "decision_date", "dateDecision", "date_decision", "dateCreation"],
    "chambre": ["chambre", "chamber", "formation", "chamber_name", "type_affaire"],
    "numero": ["numero", "number", "pourvoi", "reference", "num_decision", "id_decision"],
    "juridiction": ["juridiction", "jurisdiction", "court", "court_name", "tribunal", "type_decision", "typeDecision"],
}

@router.get("/debug/openlegi")
async def debug_openlegi(query: str = "cassation criminelle"):
    """
    Endpoint de débogage : expose les réponses brutes d'OpenLegi avant et après normalisation.

    Retourne pour chaque résultat :
    - raw : réponse brute d'OpenLegi (tous les champs)
    - normalized : données après _normalize_jurisprudence()
    - formatted : résultat de format_jurisprudence()
    - field_comparison : quels champs attendus ont été trouvés ou manquent
    """
    try:
        debug_items = await openlegi_service.search_jurisprudence_debug(query, limit=3)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={"error": str(exc), "service": "openlegi", "query": query},
        ) from exc

    results = []
    for entry in debug_items:
        raw = entry["raw"]
        normalized = entry["normalized"]
        formatted = format_jurisprudence(normalized)

        field_comparison: dict = {}
        for field, variants in _EXPECTED_FIELDS.items():
            found_key = next((k for k in variants if raw.get(k) not in (None, "", [])), None)
            field_comparison[field] = {
                "found": found_key is not None,
                "found_key": found_key,
                "value": raw.get(found_key) if found_key else None,
                "normalized_value": normalized.get(field, ""),
                "checked_keys": variants,
            }

        results.append({
            "raw": raw,
            "normalized": normalized,
            "formatted": formatted,
            "field_comparison": field_comparison,
        })

    return {
        "query": query,
        "count": len(results),
        "results": results,
    }

@router.post("/cas-pratique")
async def resoudre_cas_pratique(req: CasPratiqueRequest):
    """Résout un cas pratique et génère une plaidoirie complète"""

    api_errors: list[dict] = []

    # Étape 1 — Analyse juridique par Groq
    analyse = await identifier_articles_pertinents(req.faits, req.domaine)

    # Étape 2 — Récupérer les articles pertinents
    articles = []
    for art_info in analyse.get("articles_applicables", [])[:5]:
        article = await get_article_code(art_info.get("code",""), art_info.get("article",""))
        if article:
            article["pertinence"] = art_info.get("pertinence", "")
            article["favorable"]  = art_info.get("favorable", True)
            articles.append(article)
        else:
            # Article non trouvé en local — l'ajouter avec les infos disponibles
            articles.append({
                "code":       art_info.get("code", ""),
                "numero":     art_info.get("article", ""),
                "texte":      f"Article {art_info.get('article','')} du {art_info.get('code','')}",
                "pertinence": art_info.get("pertinence", ""),
                "favorable":  art_info.get("favorable", True),
            })

    # Étape 3 — Rechercher la jurisprudence (dégradation gracieuse si API indisponible)
    jurisprudence = []
    qualification_juridique = analyse.get("qualification_juridique", "")
    for terme in analyse.get("jurisprudence_rechercher", [])[:2]:
        try:
            results = await search_judilibre(
                terme,
                faits=req.faits,
                qualification=qualification_juridique,
            )
            jurisprudence.extend(results[:2])
        except OpenLegiError as exc:
            logger.warning("OpenLegi indisponible pour terme=%r : %s", terme, exc)
            api_errors.append({
                "service": "openlegi",
                "query": terme,
                "error": str(exc),
                "status_code": exc.status_code,
            })
        except Exception as exc:
            logger.warning("Erreur inattendue OpenLegi pour terme=%r : %s", terme, exc)
            api_errors.append({
                "service": "openlegi",
                "query": terme,
                "error": str(exc),
                "status_code": None,
            })

    # Étape 4 — Générer la plaidoirie Thémis
    plaidoirie = await generer_plaidoirie_themis(
        faits=req.faits,
        position=req.position,
        analyse_juridique=analyse,
        articles=articles,
        jurisprudence=jurisprudence[:4],
        domaine=req.domaine,
    )

    response: dict = {
        **plaidoirie,
        "analyse_juridique": analyse,
        "nb_articles":       len(articles),
        "nb_jurisprudence":  len(jurisprudence),
        "api_errors":        api_errors,
    }
    return response

@router.post("/recherche")
async def rechercher(req: RechercheRequest):
    """Recherche dans Légifrance"""
    try:
        if req.type == "jurisprudence":
            results = await search_judilibre(req.query)
            return {"results": results, "type": "jurisprudence", "query": req.query}
        else:
            results = await search_legifrance_text(req.query)
            return {"results": results, "type": req.type, "query": req.query}
    except OpenLegiError as exc:
        logger.error("Erreur OpenLegi lors de la recherche type=%r query=%r : %s", req.type, req.query, exc)
        raise HTTPException(
            status_code=502,
            detail={"error": str(exc), "service": req.type, "query": req.query},
        ) from exc
    except Exception as exc:
        logger.error("Erreur inattendue lors de la recherche type=%r query=%r : %s", req.type, req.query, exc)
        raise HTTPException(status_code=502, detail={"error": str(exc), "service": req.type, "query": req.query}) from exc

@router.post("/extract-pdf")
async def extract_pdf(file: UploadFile = File(...)):
    """Extrait le texte d un PDF de cas pratique"""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Fichier PDF uniquement")
    
    pdf_bytes = await file.read()
    if len(pdf_bytes) > 10 * 1024 * 1024:
        raise HTTPException(400, "PDF trop volumineux (max 10MB)")
    
    try:
        import io
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        except ImportError:
            # Fallback si pypdf pas installé
            try:
                import pdfplumber
                with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                    text = "\n".join([p.extract_text() or "" for p in pdf.pages])
            except ImportError:
                raise HTTPException(500, "Module PDF non disponible — installez pypdf")
        
        text = text.strip()
        if not text:
            raise HTTPException(422, "Impossible d extraire le texte du PDF")
        
        # Résumer si trop long
        word_count = len(text.split())
        if word_count > 2000:
            # Résumer via Groq
            resume_response = await call_groq_async(
                client,
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "Tu es un juriste expert. Résume ce cas pratique en conservant tous les faits juridiquement pertinents. Réponds en texte pur."},
                    {"role": "user", "content": f"Résume ce cas pratique ({word_count} mots) en 400 mots maximum en conservant tous les faits importants:\n\n{text[:6000]}"}
                ],
                max_tokens=600, temperature=0.2,
            )
            resume = resume_response.choices[0].message.content.strip()
            return {"text": resume, "original_length": word_count, "summarized": True}
        
        return {"text": text, "original_length": word_count, "summarized": False}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Erreur extraction PDF: {str(e)}")


# ── Export PDF ────────────────────────────────────────────────────────────────

class ArticleCite(BaseModel):
    code: str = ""
    numero: str = ""
    texte: str = ""


class JurisprudenceItem(BaseModel):
    formatage_officiel: str = ""
    resume: str = ""


class AnalysePdfRequest(BaseModel):
    qualification: str = ""
    essence_du_drame: str = ""
    tirade_oratoire: str = ""
    strategie: str = ""
    articles_cites: List[ArticleCite] = []
    jurisprudence: List[JurisprudenceItem] = []


@router.post("/export/analyse-pdf", tags=["legifrance"])
async def export_analyse_pdf(req: AnalysePdfRequest):
    """
    Exporte l'analyse juridique complète en PDF avec le style Éloquence AI.

    Retourne un fichier PDF incluant : qualification juridique, essence du drame,
    stratégie juridique, plaidoirie, fondements juridiques et jurisprudences.
    """
    try:
        loop = asyncio.get_running_loop()
        pdf_bytes = await loop.run_in_executor(
            None,
            lambda: generate_analyse_pdf(
                qualification=req.qualification,
                essence_du_drame=req.essence_du_drame,
                tirade_oratoire=req.tirade_oratoire,
                strategie=req.strategie,
                articles_cites=[a.model_dump() for a in req.articles_cites],
                jurisprudence=[j.model_dump() for j in req.jurisprudence],
            ),
        )
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": 'attachment; filename="analyse-juridique.pdf"'},
        )
    except Exception as exc:
        logger.error("Erreur génération PDF : %s", exc)
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération du PDF : {exc}") from exc


# ── Analyse texte légal avec LegalBert ───────────────────────────────────────

def _extract_legal_score_and_classification(bert_result: dict) -> tuple[float | None, str]:
    """Extrait le score de légalité et la classification du résultat LegalBert.

    Returns:
        Tuple (legal_score, classification) où classification est "LEGAL" ou "NON-LEGAL".
    """
    results = bert_result.get("results")
    if not results:
        return None, "NON-LEGAL"
    label = results[0].get("label", "")
    score = results[0].get("score")
    if score is None:
        return None, "NON-LEGAL"
    # LABEL_1 = classe légale positive ; LABEL_0 = négative
    if label == "LABEL_1":
        legal_score = float(score)
        classification = "LEGAL"
    else:
        legal_score = float(1.0 - score)
        classification = "LEGAL" if legal_score >= 0.5 else "NON-LEGAL"
    return legal_score, classification


@router.post("/analyze-texte-avec-bert", response_model=AnalyzeTextResponse)
async def analyze_legal_text_with_sources(request: AnalyzeTextRequest):
    """
    Analyse un texte légal (LegalBert) et enrichit avec jurisprudence auto.

    - Score le texte avec LegalBert
    - Si score < 0.7 → retourne classification + score uniquement
    - Si score >= 0.7 et search_jurisprudence=True → recherche OpenLegi async
    - Retourne classification + sources pertinentes
    """
    bert_result = await analyze_legal_text(request.text)

    if "error_type" in bert_result:
        if bert_result["error_type"] == "validation_error":
            raise HTTPException(status_code=400, detail=bert_result["error"])
        raise HTTPException(status_code=500, detail=bert_result["error"])

    legal_score, classification = _extract_legal_score_and_classification(bert_result)
    if legal_score is None:
        raise HTTPException(status_code=500, detail="Résultat LegalBert invalide")

    latency_ms = bert_result.get("latency_ms", 0.0)

    response: dict = {
        "text": request.text,
        "classification": classification,
        "legal_score": legal_score,
        "latency_ms": latency_ms,
        "jurisprudence": None,
        "textes": None,
    }

    # Recherche jurisprudence + textes si score suffisamment élevé
    if legal_score >= 0.7 and request.search_jurisprudence:
        query = request.text[:200]  # Limite raisonnable pour une requête de recherche OpenLegi
        try:
            jurisprudence, textes = await asyncio.gather(
                openlegi_service.search_jurisprudence(query, limit=3),
                openlegi_service.search_textes(query, limit=3),
            )
            response["jurisprudence"] = jurisprudence
            response["textes"] = textes
        except OpenLegiError as exc:
            logger.warning(
                "OpenLegi indisponible pour analyze-texte-avec-bert : %s", exc
            )
            response["jurisprudence"] = []
            response["textes"] = []
        except Exception as exc:
            logger.warning(
                "Erreur inattendue OpenLegi dans analyze-texte-avec-bert : %s", exc
            )
            response["jurisprudence"] = []
            response["textes"] = []

    return response
