"""
services/openlegi_service.py — Service OpenLegi via MCP (Model Context Protocol)

Remplace Judilibre et Légifrance avec un service unifié.
Endpoint : https://mcp.openlegi.fr/legifrance/mcp?token=<OPENLEGI_TOKEN>
"""

import asyncio
import html
import json
import logging
import os
import re

from groq import Groq
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

logger = logging.getLogger(__name__)

_OPENLEGI_BASE_URL = "https://mcp.openlegi.fr/legifrance/mcp"
_GROQ_MODEL = "llama-3.3-70b-versatile"
_groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


class OpenLegiError(Exception):
    """Erreur levée lors d'un échec du service OpenLegi."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def _build_url() -> str:
    """Construit l'URL MCP OpenLegi avec le token d'authentification."""
    token = os.getenv("OPENLEGI_TOKEN", "")
    if not token:
        raise OpenLegiError("Variable d'environnement OPENLEGI_TOKEN manquante")
    return f"{_OPENLEGI_BASE_URL}?token={token}"


def _extract_text(result) -> str:
    """Extrait le texte brut du résultat MCP."""
    for item in result.content:
        if hasattr(item, "text") and item.text:
            return item.text
    return ""


_RE_OPENLEGI_HEADER = re.compile(r"RÉSULTATS JURISPRUDENCE")
_RE_OPENLEGI_AFFICHAGE = re.compile(r"Affichage\s*:")
_RE_OPENLEGI_SEPARATOR = re.compile(r"^[=\-]{10,}\s*$")


def _clean_openlegi_text(text: str) -> str:
    """Supprime les en-têtes et séparateurs techniques d'OpenLegi du texte brut."""
    lines = text.split("\n")
    cleaned = [
        line for line in lines
        if not _RE_OPENLEGI_HEADER.match(line)
        and not _RE_OPENLEGI_AFFICHAGE.match(line)
        and not _RE_OPENLEGI_SEPARATOR.match(line)
    ]
    return "\n".join(cleaned).strip()


def _parse_result(text: str) -> list[dict]:
    """
    Tente de parser le texte retourné par OpenLegi.
    Retourne une liste de dicts ou une liste contenant le texte brut.
    """
    text = text.strip()
    if not text:
        return []

    # Tentative de parsing JSON direct
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # Certains outils retournent {"results": [...]} ou {"items": [...]}
            for key in ("results", "items", "decisions", "textes"):
                if isinstance(data.get(key), list):
                    return data[key]
            return [data]
    except json.JSONDecodeError:
        pass

    # Nettoie les en-têtes OpenLegi avant de retourner le texte brut
    text = _clean_openlegi_text(text)

    # Retourne le texte brut emballé dans un dict pour dégradation gracieuse
    return [{"resume": text, "source": "openlegi"}]


async def extract_metadata_from_text(text: str) -> dict:
    """
    Utilise Groq pour extraire les métadonnées de jurisprudence depuis un texte brut.

    Args:
        text: Texte brut retourné par OpenLegi.

    Returns:
        Dict contenant juridiction, chambre, date (ISO) et numero, ou {} en cas d'échec.
    """
    if not text or not text.strip():
        return {}

    try:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: _groq_client.chat.completions.create(
                model=_GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Tu es un assistant juridique expert en droit français. "
                            "Extrais les informations suivantes du texte juridique fourni "
                            "et retourne UNIQUEMENT un objet JSON valide, sans markdown ni backticks : "
                            '{"juridiction": "...", "chambre": "...", "date": "YYYY-MM-DD", "numero": "..."} '
                            "Si une information est absente, utilise une chaîne vide. "
                            "La date doit être au format ISO 8601 (YYYY-MM-DD). "
                            "Le numéro de pourvoi ressemble à '14-83.462' ou '12-34.567'."
                        ),
                    },
                    {
                        "role": "user",
                        "content": text[:2000],
                    },
                ],
                max_tokens=200,
                temperature=0.1,
            ),
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(raw)
    except Exception as exc:
        logger.warning("extract_metadata_from_text error: %s", exc)
        return {}


class OpenLegiService:
    """Accès unifié à Légifrance via le serveur MCP OpenLegi."""

    async def search_jurisprudence(self, query: str, limit: int = 5) -> list[dict]:
        """
        Recherche dans la jurisprudence judiciaire via OpenLegi MCP.

        Args:
            query: Termes de recherche.
            limit: Nombre de résultats souhaité.

        Returns:
            Liste de décisions normalisées.
        """
        logger.debug("OpenLegi search_jurisprudence: query=%r", query)
        try:
            url = _build_url()
            async with streamablehttp_client(url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(
                        "rechercher_jurisprudence_judiciaire",
                        {"search": query, "page_size": limit},
                    )
            text = _extract_text(result)
            items = _parse_result(text)
            return list(
                await asyncio.gather(
                    *[self._normalize_jurisprudence(item) for item in items[:limit]]
                )
            )
        except OpenLegiError:
            raise
        except Exception as exc:
            logger.error(
                "OpenLegi search_jurisprudence error for query=%r : %s", query, exc
            )
            raise OpenLegiError(
                f"Erreur lors de la recherche de jurisprudence : {exc}"
            ) from exc

    async def search_jurisprudence_debug(self, query: str, limit: int = 3) -> list[dict]:
        """
        Recherche dans la jurisprudence et retourne les données brutes ET normalisées.

        Args:
            query: Termes de recherche.
            limit: Nombre de résultats souhaité.

        Returns:
            Liste de dicts contenant les données brutes et normalisées pour chaque résultat.
        """
        logger.debug("OpenLegi search_jurisprudence_debug: query=%r", query)
        try:
            url = _build_url()
            async with streamablehttp_client(url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(
                        "rechercher_jurisprudence_judiciaire",
                        {"search": query, "page_size": limit},
                    )
            text = _extract_text(result)
            items = _parse_result(text)
            normalized_items = await asyncio.gather(
                *[self._normalize_jurisprudence(item) for item in items[:limit]]
            )
            debug_results = [
                {"raw": item, "normalized": normalized}
                for item, normalized in zip(items[:limit], normalized_items)
            ]
            return debug_results
        except OpenLegiError:
            raise
        except Exception as exc:
            logger.error(
                "OpenLegi search_jurisprudence_debug error for query=%r : %s", query, exc
            )
            raise OpenLegiError(
                f"Erreur lors de la recherche de jurisprudence (debug) : {exc}"
            ) from exc

    async def search_textes(self, query: str, limit: int = 5) -> list[dict]:
        """
        Recherche dans les codes juridiques via OpenLegi MCP.

        Args:
            query: Termes de recherche.
            limit: Nombre de résultats souhaité.

        Returns:
            Liste de textes normalisés.
        """
        logger.debug("OpenLegi search_textes: query=%r", query)
        try:
            url = _build_url()
            async with streamablehttp_client(url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(
                        "rechercher_code",
                        {"codeQuery": query, "pageSize": limit},
                    )
            text = _extract_text(result)
            items = _parse_result(text)
            return [self._normalize_texte(item) for item in items[:limit]]
        except OpenLegiError:
            raise
        except Exception as exc:
            logger.error(
                "OpenLegi search_textes error for query=%r : %s", query, exc
            )
            raise OpenLegiError(
                f"Erreur lors de la recherche de textes : {exc}"
            ) from exc

    async def _normalize_jurisprudence(self, item: dict) -> dict:
        """Transforme une entrée brute en format uniforme (jurisprudence).

        Quand OpenLegi retourne du texte brut (champ `resume` uniquement),
        utilise Groq pour en extraire les métadonnées structurées.
        """
        logger.debug(
            "OpenLegi raw item keys: %s",
            list(item.keys()) if isinstance(item, dict) else type(item).__name__,
        )

        # Date — plusieurs noms possibles selon la version de l'API
        date = (
            item.get("date")
            or item.get("decision_date")
            or item.get("dateDecision")
            or item.get("date_decision")
            or item.get("dateCreation")
            or ""
        )

        # Chambre — plusieurs noms possibles
        chambre = (
            item.get("chambre")
            or item.get("chamber")
            or item.get("formation")
            or item.get("chamber_name")
            or item.get("type_affaire")
            or ""
        )

        # Numéro de pourvoi — plusieurs noms possibles
        numero = (
            item.get("numero")
            or item.get("number")
            or item.get("pourvoi")
            or item.get("reference")
            or item.get("num_decision")
            or item.get("id_decision")
            or ""
        )

        # Juridiction — plusieurs noms possibles ; fallback depuis type_decision
        juridiction = (
            item.get("juridiction")
            or item.get("jurisdiction")
            or item.get("court")
            or item.get("court_name")
            or item.get("tribunal")
            or ""
        )
        if not juridiction:
            # Certains résultats encodent la juridiction dans le type de décision
            type_decision = item.get("type_decision") or item.get("typeDecision") or ""
            if type_decision:
                juridiction = type_decision

        # Si les champs structurés sont vides, utiliser Groq pour extraire
        # les métadonnées depuis le texte brut du résumé
        if not any([date, chambre, numero, juridiction]):
            resume_text = str(item.get("resume", item.get("summary", "")))
            if resume_text:
                extracted = await extract_metadata_from_text(resume_text)
                date = date or extracted.get("date", "")
                chambre = chambre or extracted.get("chambre", "")
                numero = numero or extracted.get("numero", "")
                juridiction = juridiction or extracted.get("juridiction", "")

        resume_text = str(item.get("resume", item.get("summary", "")))[:500]
        resume_justified = f'<p style="text-align: justify">{html.escape(resume_text)}</p>'
        return {
            "id": item.get("id", ""),
            "date": date,
            "chambre": chambre,
            "solution": item.get("solution", ""),
            "resume": resume_text,
            "resume_html": resume_justified,
            "resume_styled": resume_justified,
            "themes": item.get("themes", []),
            "numero": numero,
            "juridiction": juridiction,
            "source": "openlegi",
        }

    @staticmethod
    def _normalize_texte(item: dict) -> dict:
        """Transforme une entrée brute en format uniforme (texte de loi)."""
        return {
            "id": item.get("id", ""),
            "titre": item.get("titre", item.get("title", item.get("titreCode", ""))),
            "nature": item.get("nature", ""),
            "date": item.get("date", item.get("dateTexte", "")),
            "nor": item.get("nor", ""),
            "resume": item.get("resume", item.get("texte", "")),
            "url": item.get("url", ""),
            "source": "openlegi",
        }


# Instance partagée
openlegi_service = OpenLegiService()
