"""
services/piste_auth.py — Authentification OAuth 2.0 pour le portail PISTE
Gère l'obtention, le cache et le refresh du token client_credentials.
"""

import time
import asyncio
import logging
import os
import httpx

logger = logging.getLogger(__name__)

class PISTEAuth:
    """Gestion du token OAuth 2.0 PISTE avec cache mémoire et refresh automatique."""

    def __init__(self):
        self._token: str | None = None
        self._expires_at: float = 0.0
        self._lock = asyncio.Lock()

    async def get_token(self) -> str:
        """Retourne un token valide, depuis le cache ou en en obtenant un nouveau."""
        async with self._lock:
            if self._token and time.time() < self._expires_at - 60:
                return self._token
            return await self._fetch_token()

    async def refresh_token(self) -> str:
        """Force l'obtention d'un nouveau token en ignorant le cache."""
        async with self._lock:
            self._token = None
            self._expires_at = 0.0
            return await self._fetch_token()

    async def _fetch_token(self) -> str:
        """Appelle l'endpoint OAuth PISTE et met à jour le cache."""
        client_id = os.getenv("PISTE_CLIENT_ID")
        client_secret = os.getenv("PISTE_CLIENT_SECRET")
        oauth_url = os.getenv("PISTE_OAUTH_URL", "https://piste.gouv.fr/api/oauth/token")
        token_cache_ttl = int(os.getenv("TOKEN_CACHE_TTL", "3600"))

        if not client_id or not client_secret:
            raise RuntimeError(
                "PISTE_CLIENT_ID et PISTE_CLIENT_SECRET doivent être définis "
                "dans les variables d'environnement."
            )

        try:
            async with httpx.AsyncClient(timeout=15) as http:
                r = await http.post(
                    oauth_url,
                    data={
                        "grant_type": "client_credentials",
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "scope": "openid",
                    },
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                r.raise_for_status()
                data = r.json()

        except httpx.HTTPStatusError as exc:
            logger.error("Erreur OAuth PISTE %s : %s", exc.response.status_code, exc.response.text)
            raise RuntimeError(f"Authentification PISTE échouée ({exc.response.status_code})") from exc
        except Exception as exc:
            logger.error("Erreur réseau OAuth PISTE : %s", exc)
            raise RuntimeError(f"Impossible de contacter le serveur OAuth PISTE : {exc}") from exc

        self._token = data.get("access_token") or data.get("token")
        if not self._token:
            raise RuntimeError(f"Réponse OAuth PISTE invalide : {data}")

        expires_in = int(data.get("expires_in", token_cache_ttl))
        self._expires_at = time.time() + expires_in
        logger.info("Nouveau token PISTE obtenu, expire dans %ds", expires_in)
        return self._token

    async def get_headers(self) -> dict:
        """Retourne les headers HTTP avec le Bearer token."""
        token = await self.get_token()
        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }


# Instance partagée (singleton)
piste_auth = PISTEAuth()
