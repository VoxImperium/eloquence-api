"""
services/legifrance_service.py — Service Légifrance via l'API PISTE authentifiée
Endpoints : POST /search, GET /consult/texte/{id}
"""

import logging
import httpx
from .piste_auth import piste_auth

logger = logging.getLogger(__name__)

LEGI_BASE_URL = "https://api.piste.gouv.fr/dila/legifrance/lf-engine-app"


class LegifranceAPIError(Exception):
    """Erreur levée lors d'un échec de l'API Légifrance PISTE."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class LegifranceService:
    """Accès aux textes de loi Légifrance via l'API PISTE (OAuth 2.0)."""

    async def search_textes(
        self,
        query: str,
        type_texte: str | None = None,
        page: int = 1,
        limit: int = 20,
    ) -> list[dict]:
        """
        Recherche de textes de loi via l'API Légifrance PISTE.

        Args:
            query: Termes de recherche.
            type_texte: Filtre optionnel (LOI, DECRET, ARRETE, CODE, etc.).
            page: Numéro de page (commence à 1).
            limit: Nombre de résultats par page.

        Returns:
            Liste de textes normalisés.
        """
        payload: dict = {
            "recherche": {
                "champs": [
                    {"typeChamp": "ALL", "criteres": [{"typeRecherche": "TOUS_LES_MOTS_DANS_UN_CHAMP", "valeur": query}], "operateur": "ET"}
                ],
                "pageNumber": page,
                "pageSize": limit,
                "sort": "PERTINENCE",
                "typePagination": "DEFAUT",
            }
        }

        if type_texte:
            payload["recherche"]["filtres"] = [
                {"facette": "NATURE", "valeur": type_texte.upper()}
            ]

        logger.debug("Légifrance search: query=%r type=%r page=%d limit=%d", query, type_texte, page, limit)

        try:
            headers = await piste_auth.get_headers()
            async with httpx.AsyncClient(timeout=30) as http:
                r = await http.post(
                    f"{LEGI_BASE_URL}/search",
                    json=payload,
                    headers=headers,
                )
                if r.status_code == 401:
                    # Token expiré — forcer le refresh et réessayer
                    logger.warning("Légifrance search: token expiré (401), tentative de refresh")
                    await piste_auth.refresh_token()
                    headers = await piste_auth.get_headers()
                    r = await http.post(
                        f"{LEGI_BASE_URL}/search",
                        json=payload,
                        headers=headers,
                    )
                logger.debug("Légifrance search response: HTTP %d for query=%r", r.status_code, query)
                r.raise_for_status()
                data = r.json()

        except httpx.TimeoutException as exc:
            logger.error("Légifrance search timeout for query=%r : %s", query, exc)
            raise LegifranceAPIError(
                f"Timeout lors de la recherche Légifrance (query={query!r})"
            ) from exc
        except httpx.HTTPStatusError as exc:
            logger.error(
                "Légifrance search HTTP %s for query=%r : %s",
                exc.response.status_code, query, exc.response.text[:300],
            )
            raise LegifranceAPIError(
                f"Erreur HTTP {exc.response.status_code} de l'API Légifrance",
                status_code=exc.response.status_code,
            ) from exc
        except Exception as exc:
            logger.error("Légifrance search error for query=%r : %s", query, exc)
            raise LegifranceAPIError(
                f"Erreur lors de la recherche Légifrance : {exc}"
            ) from exc

        results = data.get("results", [])
        return [self._normalize(item) for item in results[:limit]]

    async def get_texte(self, texte_id: str) -> dict | None:
        """
        Récupère le texte complet d'une loi, d'un décret ou d'un arrêté.

        Args:
            texte_id: Identifiant PISTE du texte (ex: LEGITEXT000006070721).

        Returns:
            Dictionnaire avec les données du texte, ou None si non trouvé.
        """
        logger.debug("Légifrance get_texte: id=%r", texte_id)

        try:
            headers = await piste_auth.get_headers()
            async with httpx.AsyncClient(timeout=30) as http:
                r = await http.post(
                    f"{LEGI_BASE_URL}/consult/texte",
                    json={"textId": texte_id},
                    headers=headers,
                )
                if r.status_code == 401:
                    logger.warning("Légifrance get_texte: token expiré (401), tentative de refresh")
                    await piste_auth.refresh_token()
                    headers = await piste_auth.get_headers()
                    r = await http.post(
                        f"{LEGI_BASE_URL}/consult/texte",
                        json={"textId": texte_id},
                        headers=headers,
                    )
                if r.status_code == 404:
                    return None
                r.raise_for_status()
                return r.json()

        except httpx.TimeoutException as exc:
            logger.error("Légifrance get_texte timeout for id=%r : %s", texte_id, exc)
            raise LegifranceAPIError(
                f"Timeout lors de la récupération du texte Légifrance (id={texte_id!r})"
            ) from exc
        except httpx.HTTPStatusError as exc:
            logger.error(
                "Légifrance get_texte HTTP %s for id=%r : %s",
                exc.response.status_code, texte_id, exc.response.text[:300],
            )
            raise LegifranceAPIError(
                f"Erreur HTTP {exc.response.status_code} de l'API Légifrance",
                status_code=exc.response.status_code,
            ) from exc
        except Exception as exc:
            logger.error("Légifrance get_texte error for id=%r : %s", texte_id, exc)
            raise LegifranceAPIError(
                f"Erreur lors de la récupération du texte Légifrance : {exc}"
            ) from exc

    @staticmethod
    def _normalize(item: dict) -> dict:
        """Transforme une entrée brute PISTE en format uniforme."""
        titles = item.get("titles", [{}])
        title_obj = titles[0] if titles else {}
        return {
            "id": item.get("id", ""),
            "titre": title_obj.get("title", item.get("title", "")),
            "nature": item.get("nature", ""),
            "date": item.get("dateTexte", item.get("date", "")),
            "nor": item.get("nor", ""),
            "resume": item.get("resume", ""),
            "url": item.get("url", ""),
            "source": "legifrance",
        }


# Instance partagée
legifrance_service = LegifranceService()
