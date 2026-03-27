"""
services/legifrance_service.py — Service Légifrance via l'API PISTE authentifiée
Endpoints : POST /search, GET /consult/texte/{id}
"""

import logging
import httpx
from .piste_auth import piste_auth

logger = logging.getLogger(__name__)

LEGI_BASE_URL = "https://api.piste.gouv.fr/dila/legifrance/lf-engine-app"


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
        headers = await piste_auth.get_headers()

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

        try:
            async with httpx.AsyncClient(timeout=20) as http:
                r = await http.post(
                    f"{LEGI_BASE_URL}/search",
                    json=payload,
                    headers=headers,
                )
                if r.status_code == 401:
                    # Token expiré — forcer le refresh et réessayer
                    await piste_auth.refresh_token()
                    headers = await piste_auth.get_headers()
                    r = await http.post(
                        f"{LEGI_BASE_URL}/search",
                        json=payload,
                        headers=headers,
                    )
                r.raise_for_status()
                data = r.json()

        except httpx.HTTPStatusError as exc:
            logger.error("Légifrance search HTTP %s : %s", exc.response.status_code, exc.response.text[:300])
            return []
        except Exception as exc:
            logger.error("Légifrance search error: %s", exc)
            return []

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
        headers = await piste_auth.get_headers()

        try:
            async with httpx.AsyncClient(timeout=20) as http:
                r = await http.post(
                    f"{LEGI_BASE_URL}/consult/texte",
                    json={"textId": texte_id},
                    headers=headers,
                )
                if r.status_code == 401:
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

        except httpx.HTTPStatusError as exc:
            logger.error("Légifrance get texte HTTP %s : %s", exc.response.status_code, exc.response.text[:300])
            return None
        except Exception as exc:
            logger.error("Légifrance get texte error: %s", exc)
            return None

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
