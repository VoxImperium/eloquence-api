"""
services/judilibre_service.py — Service Judilibre via l'API PISTE authentifiée
Endpoints : GET /search, GET /decision/{id}
"""

import logging
import httpx
from .piste_auth import piste_auth

logger = logging.getLogger(__name__)

JUDILIBRE_BASE_URL = "https://api.piste.gouv.fr/cassation/judilibre/v1"


class JudilibreService:
    """Accès aux décisions de justice via l'API Judilibre PISTE (OAuth 2.0)."""

    async def search_decisions(
        self,
        query: str,
        filters: dict | None = None,
        page: int = 1,
        limit: int = 20,
    ) -> list[dict]:
        """
        Recherche plein texte dans la jurisprudence Judilibre.

        Args:
            query: Termes de recherche.
            filters: Dictionnaire optionnel de filtres PISTE :
                     chambre, date_start, date_end, juridiction, type.
            page: Numéro de page (commence à 0 pour Judilibre).
            limit: Nombre de résultats par page.

        Returns:
            Liste de décisions normalisées.
        """
        headers = await piste_auth.get_headers()
        # Judilibre attend une page à base 0
        page_index = max(0, page - 1)

        params: dict = {
            "query": query,
            "operator": "AND",
            "field": ["summary", "themes"],
            "resolve_references": "true",
            "page_size": limit,
            "page_index": page_index,
        }

        if filters:
            if filters.get("chambre"):
                params["chamber"] = filters["chambre"]
            if filters.get("juridiction"):
                params["jurisdiction"] = filters["juridiction"]
            if filters.get("date_start"):
                params["date_start"] = filters["date_start"]
            if filters.get("date_end"):
                params["date_end"] = filters["date_end"]
            if filters.get("type"):
                params["type"] = filters["type"]

        try:
            async with httpx.AsyncClient(timeout=20) as http:
                r = await http.get(
                    f"{JUDILIBRE_BASE_URL}/search",
                    params=params,
                    headers=headers,
                )
                if r.status_code == 401:
                    # Token expiré — forcer le refresh et réessayer
                    await piste_auth.refresh_token()
                    headers = await piste_auth.get_headers()
                    r = await http.get(
                        f"{JUDILIBRE_BASE_URL}/search",
                        params=params,
                        headers=headers,
                    )
                r.raise_for_status()
                data = r.json()

        except httpx.HTTPStatusError as exc:
            logger.error("Judilibre search HTTP %s : %s", exc.response.status_code, exc.response.text[:300])
            return []
        except Exception as exc:
            logger.error("Judilibre search error: %s", exc)
            return []

        results = data.get("results", [])
        return [self._normalize(res) for res in results]

    async def get_decision(self, decision_id: str) -> dict | None:
        """
        Récupère une décision complète avec enrichissements.

        Args:
            decision_id: Identifiant PISTE de la décision.

        Returns:
            Dictionnaire avec les données de la décision, ou None si non trouvée.
        """
        headers = await piste_auth.get_headers()

        try:
            async with httpx.AsyncClient(timeout=20) as http:
                r = await http.get(
                    f"{JUDILIBRE_BASE_URL}/decision",
                    params={"id": decision_id, "resolve_references": "true"},
                    headers=headers,
                )
                if r.status_code == 401:
                    await piste_auth.refresh_token()
                    headers = await piste_auth.get_headers()
                    r = await http.get(
                        f"{JUDILIBRE_BASE_URL}/decision",
                        params={"id": decision_id, "resolve_references": "true"},
                        headers=headers,
                    )
                if r.status_code == 404:
                    return None
                r.raise_for_status()
                data = r.json()

        except httpx.HTTPStatusError as exc:
            logger.error("Judilibre get decision HTTP %s : %s", exc.response.status_code, exc.response.text[:300])
            return None
        except Exception as exc:
            logger.error("Judilibre get decision error: %s", exc)
            return None

        results = data.get("results", [])
        if not results:
            return None
        return self._normalize(results[0], full=True)

    @staticmethod
    def _normalize(res: dict, full: bool = False) -> dict:
        """Transforme une entrée brute Judilibre en format uniforme."""
        normalized = {
            "id": res.get("id", ""),
            "date": res.get("decision_date", ""),
            "chambre": res.get("chamber", ""),
            "solution": res.get("solution", ""),
            "resume": (res.get("summary", "") or "")[:500],
            "themes": res.get("themes", []),
            "numero": res.get("number", ""),
            "juridiction": res.get("jurisdiction", ""),
            "source": "judilibre",
        }
        if full:
            normalized["texte_integral"] = res.get("text", "")
            normalized["references"] = res.get("bulletin", [])
            normalized["sommaire"] = res.get("summary", "")
        return normalized


# Instance partagée
judilibre_service = JudilibreService()
