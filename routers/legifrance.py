from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from groq import Groq
import httpx, os, json, re

router = APIRouter(prefix="/legifrance", tags=["legifrance"])
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# APIs publiques Légifrance
LEGI_SEARCH_URL = "https://recherche.data.gouv.fr/api/1/datasets/"
JUDILIBRE_URL   = "https://api.piste.gouv.fr/cassation/judilibre/v1"
CODES_URL       = "https://api.legifrance.gouv.fr/consult/code"

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

async def search_judilibre(query: str, operator: str = "AND") -> list:
    """Recherche dans la jurisprudence via Judilibre (API publique)"""
    try:
        async with httpx.AsyncClient(timeout=15) as http:
            # API Judilibre - accès public pour la recherche
            params = {
                "query": query,
                "operator": operator,
                "field": ["summary", "themes"],
                "resolve_references": "true",
                "page_size": 5,
            }
            r = await http.get(
                "https://api.piste.gouv.fr/cassation/judilibre/v1/search",
                params=params,
                headers={"Accept": "application/json"}
            )
            if r.status_code == 200:
                data = r.json()
                results = data.get("results", [])
                return [{
                    "id":       res.get("id", ""),
                    "date":     res.get("decision_date", ""),
                    "chambre":  res.get("chamber", ""),
                    "solution": res.get("solution", ""),
                    "resume":   res.get("summary", "")[:500] if res.get("summary") else "",
                    "themes":   res.get("themes", []),
                    "numero":   res.get("number", ""),
                } for res in results[:5]]
    except Exception as e:
        print(f"Judilibre error: {e}")
    return []

async def search_legifrance_text(query: str) -> list:
    """Recherche dans les textes de loi via l'API publique"""
    try:
        async with httpx.AsyncClient(timeout=15) as http:
            # Utiliser l'API de recherche Légifrance publique
            r = await http.get(
                "https://api.piste.gouv.fr/dila/legifrance/lf-engine-app/search",
                params={
                    "query": query,
                    "searchedPage": 1,
                    "pageSize": 5,
                    "sort": "PERTINENCE",
                    "typePagination": "DEFAUT",
                },
                headers={"Accept": "application/json"}
            )
            if r.status_code == 200:
                return r.json().get("results", [])[:5]
    except Exception as e:
        print(f"Légifrance search error: {e}")
    return []

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
    raw = client.chat.completions.create(
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
    ).choices[0].message.content.strip()

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
        f"• Cass. {j.get('chambre','')} {j.get('date','')} n°{j.get('numero','')}: {j.get('resume','')[:200]}"
        for j in jurisprudence if j
    ]) or "Aucune jurisprudence trouvée"

    qualification = analyse_juridique.get("qualification_juridique", "")
    arguments    = analyse_juridique.get("arguments_principaux", [])
    strategie    = analyse_juridique.get("strategie", "")

    # Appel 1 — Exorde + Narration
    r1 = client.chat.completions.create(
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
    ).choices[0].message.content.strip()

    # Appel 2 — Développement juridique
    r2 = client.chat.completions.create(
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
    ).choices[0].message.content.strip()

    # Appel 3 — Réfutation + Péroraison
    r3 = client.chat.completions.create(
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
    ).choices[0].message.content.strip()

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

@router.post("/cas-pratique")
async def resoudre_cas_pratique(req: CasPratiqueRequest):
    """Résout un cas pratique et génère une plaidoirie complète"""

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

    # Étape 3 — Rechercher la jurisprudence
    jurisprudence = []
    for terme in analyse.get("jurisprudence_rechercher", [])[:2]:
        results = await search_judilibre(terme)
        jurisprudence.extend(results[:2])

    # Étape 4 — Générer la plaidoirie Thémis
    plaidoirie = await generer_plaidoirie_themis(
        faits=req.faits,
        position=req.position,
        analyse_juridique=analyse,
        articles=articles,
        jurisprudence=jurisprudence[:4],
    )

    return {
        **plaidoirie,
        "analyse_juridique": analyse,
        "nb_articles":       len(articles),
        "nb_jurisprudence":  len(jurisprudence),
    }

@router.post("/recherche")
async def rechercher(req: RechercheRequest):
    """Recherche dans Légifrance"""
    if req.type == "jurisprudence":
        results = await search_judilibre(req.query)
        return {"results": results, "type": "jurisprudence", "query": req.query}
    else:
        results = await search_legifrance_text(req.query)
        return {"results": results, "type": req.type, "query": req.query}

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
            resume = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "Tu es un juriste expert. Résume ce cas pratique en conservant tous les faits juridiquement pertinents. Réponds en texte pur."},
                    {"role": "user", "content": f"Résume ce cas pratique ({word_count} mots) en 400 mots maximum en conservant tous les faits importants:\n\n{text[:6000]}"}
                ],
                max_tokens=600, temperature=0.2,
            ).choices[0].message.content.strip()
            return {"text": resume, "original_length": word_count, "summarized": True}
        
        return {"text": text, "original_length": word_count, "summarized": False}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Erreur extraction PDF: {str(e)}")
