"""
services/piste_auth.py — Service OAuth PISTE pour l'authentification aux APIs gouvernementales.

Gère le flux client_credentials OAuth 2.0 avec cache du token et refresh automatique.
Compatible avec Judilibre (v1) et Légifrance (v2).
"""

import asyncio
import logging
import os
import time

import httpx

logger = logging.getLogger(__name__)

# URL du endpoint OAuth PISTE (production)
_PISTE_TOKEN_URL = "https://oauth.piste.gouv.fr/api/oauth/token"

# Marge de sécurité avant expiration du token (en secondes)
_TOKEN_REFRESH_MARGIN = 60

# Durée d'expiration par défaut du token (en secondes) si non fournie par le serveur
_DEFAULT_TOKEN_EXPIRATION = 3600


class PisteAuthError(Exception):
    """Erreur levée lors d'un échec d'authentification PISTE."""


class PisteAuth:
    """
    Gestion du token OAuth PISTE avec cache et refresh automatique.

    Utilise le flux client_credentials (machine-to-machine).
    Le token est mis en cache jusqu'à son expiration moins une marge de sécurité.
    """

    def __init__(self) -> None:
        self._token: str | None = None
        self._expires_at: float = 0.0
        self._lock = asyncio.Lock()

    async def get_token(self) -> str:
        """
        Retourne un token OAuth PISTE valide.

        Utilise le cache si le token n'est pas expiré.
        Renouvelle automatiquement avant l'expiration.

        Returns:
            Token Bearer valide.

        Raises:
            PisteAuthError: Si les credentials sont manquants ou si l'authentification échoue.
        """
        async with self._lock:
            if self._token and time.monotonic() < self._expires_at - _TOKEN_REFRESH_MARGIN:
                return self._token
            return await self._fetch_token()

    async def get_headers(self) -> dict:
        """
        Retourne les headers HTTP avec le token Bearer PISTE.

        Returns:
            Dictionnaire de headers prêts à l'emploi.
        """
        token = await self.get_token()
        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }

    async def _fetch_token(self) -> str:
        """Obtient un nouveau token depuis PISTE via le flux client_credentials."""
        client_id = os.getenv("PISTE_CLIENT_ID")
        client_secret = os.getenv("PISTE_CLIENT_SECRET")

        if not client_id or not client_secret:
            raise PisteAuthError(
                "Credentials PISTE manquants — configurez PISTE_CLIENT_ID et PISTE_CLIENT_SECRET"
            )

        logger.debug("Obtention d'un nouveau token PISTE…")
        try:
            async with httpx.AsyncClient(timeout=15) as http:
                r = await http.post(
                    _PISTE_TOKEN_URL,
                    data={
                        "grant_type": "client_credentials",
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "scope": "openid",
                    },
                )
                r.raise_for_status()
                data = r.json()

        except httpx.TimeoutException as exc:
            raise PisteAuthError(
                f"Timeout lors de l'authentification PISTE : {exc}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise PisteAuthError(
                f"Erreur HTTP {exc.response.status_code} lors de l'authentification PISTE"
                f" : {exc.response.text[:200]}"
            ) from exc
        except Exception as exc:
            raise PisteAuthError(
                f"Erreur lors de l'authentification PISTE : {exc}"
            ) from exc

        token = data.get("access_token")
        if not token:
            raise PisteAuthError(
                f"Pas de access_token dans la réponse PISTE : {list(data.keys())}"
            )

        expires_in = int(data.get("expires_in", _DEFAULT_TOKEN_EXPIRATION))
        self._token = token
        self._expires_at = time.monotonic() + expires_in
        logger.info("Token PISTE obtenu (expires_in=%ds)", expires_in)
        return token


# Instance partagée
piste_auth = PisteAuth()
