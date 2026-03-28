"""
services/openlegi_service.py — Service OpenLegi via MCP (Model Context Protocol)

Remplace Judilibre et Légifrance avec un service unifié.
Endpoint : https://mcp.openlegi.fr/legifrance/mcp?token=<OPENLEGI_TOKEN>
"""

import json
import logging
import os

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

logger = logging.getLogger(__name__)

_OPENLEGI_BASE_URL = "https://mcp.openlegi.fr/legifrance/mcp"


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

    # Retourne le texte brut emballé dans un dict pour dégradation gracieuse
    return [{"resume": text, "source": "openlegi"}]


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
            return [self._normalize_jurisprudence(item) for item in items[:limit]]
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
            debug_results = []
            for item in items[:limit]:
                normalized = self._normalize_jurisprudence(item)
                debug_results.append({"raw": item, "normalized": normalized})
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
                        {"search": query, "page_size": limit},
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

    @staticmethod
    def _normalize_jurisprudence(item: dict) -> dict:
        """Transforme une entrée brute en format uniforme (jurisprudence)."""
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

        return {
            "id": item.get("id", ""),
            "date": date,
            "chambre": chambre,
            "solution": item.get("solution", ""),
            "resume": str(item.get("resume", item.get("summary", "")))[:500],
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
