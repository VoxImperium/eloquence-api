from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from groq import Groq
import asyncio
import logging
import os, json, re

from services.openlegi_service import openlegi_service, OpenLegiError

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
) -> dict:
    """Génère la plaidoirie complète selon le style Thémis"""

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
            {"role": "system", "content": THEMIS_SYSTEM},
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
            {"role": "system", "content": THEMIS_SYSTEM},
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
            {"role": "system", "content": THEMIS_SYSTEM},
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
