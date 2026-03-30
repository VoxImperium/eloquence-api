# services/chat_service.py — Service de Chat Juridique avec RAG (OpenLégi + Groq)

import asyncio
import logging
import os

from groq import Groq

from services.openlegi_service import openlegi_service, OpenLegiError
from schemas.chat import Source

logger = logging.getLogger(__name__)

_GROQ_MODEL = "llama-3.3-70b-versatile"
_DOMAINES = {
    "civil": "droit civil",
    "pénal": "droit pénal",
    "social": "droit social et du travail",
    "commercial": "droit commercial",
    "administratif": "droit administratif",
}


class ChatService:
    """Génère des réponses juridiques avec RAG (OpenLégi + Groq)."""

    def __init__(self) -> None:
        self._groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def answer(
        self,
        question: str,
        domaine: str = "civil",
        history: list[dict] | None = None,
    ) -> tuple[str, list[Source]]:
        """
        Recherche les sources juridiques et génère une réponse contextualisée.

        Returns:
            Tuple (texte_réponse, liste_sources).
        """
        jurisprudence, textes = await self._fetch_sources(question)
        sources = self._build_sources(jurisprudence, textes)
        context = self._build_context(jurisprudence, textes)
        response_text = await self._generate_response(
            question, context, domaine, history or []
        )
        return response_text, sources

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _fetch_sources(
        self, query: str
    ) -> tuple[list[dict], list[dict]]:
        """Recherche indépendante jurisprudence + textes (3 résultats chacun).

        Chaque source est traitée indépendamment : une erreur sur la jurisprudence
        ne bloque pas la récupération des textes, et vice versa.
        """
        jurisprudence: list[dict] = []
        textes: list[dict] = []

        try:
            jurisprudence = await openlegi_service.search_jurisprudence(query, limit=3)
        except OpenLegiError:
            logger.warning(
                "OpenLégi indisponible (jurisprudence) pour query=%r — fallback vide",
                query,
            )
        except Exception as exc:
            logger.error(
                "Erreur inattendue lors de la recherche jurisprudence pour query=%r : %s",
                query,
                exc,
            )

        try:
            textes = await openlegi_service.search_textes(query, limit=3)
        except OpenLegiError:
            logger.warning(
                "OpenLégi indisponible (textes) pour query=%r — fallback vide",
                query,
            )
        except Exception as exc:
            logger.error(
                "Erreur inattendue lors de la recherche textes pour query=%r : %s",
                query,
                exc,
            )

        return jurisprudence, textes

    @staticmethod
    def _jurisprudence_titre(j: dict) -> str:
        """Construit le titre officiel d'une décision de jurisprudence."""
        parts = [
            j.get("juridiction", ""),
            j.get("chambre", ""),
            j.get("date", ""),
            j.get("numero", ""),
        ]
        return ", ".join(p for p in parts if p) or "Décision"

    @staticmethod
    def _build_sources(jurisprudence: list[dict], textes: list[dict]) -> list[Source]:
        """Convertit les données normalisées OpenLégi en objets Source."""
        sources: list[Source] = []
        for j in jurisprudence:
            sources.append(
                Source(
                    type="jurisprudence",
                    titre=ChatService._jurisprudence_titre(j),
                    contenu=j.get("resume", ""),
                    url=j.get("url", ""),
                )
            )
        for t in textes:
            sources.append(
                Source(
                    type="loi",
                    titre=t.get("titre", "Texte de loi"),
                    contenu=t.get("resume", ""),
                    url=t.get("url", ""),
                )
            )
        return sources

    @staticmethod
    def _build_context(jurisprudence: list[dict], textes: list[dict]) -> str:
        """Construit le contexte RAG à injecter dans le prompt."""
        lines: list[str] = []

        if jurisprudence:
            lines.append("## Jurisprudence pertinente")
            for j in jurisprudence:
                titre = ChatService._jurisprudence_titre(j)
                resume = j.get("resume", "")
                lines.append(f"- {titre} : {resume}")

        if textes:
            lines.append("\n## Textes de loi")
            for t in textes:
                titre = t.get("titre", "Texte de loi")
                resume = t.get("resume", "")
                lines.append(f"- {titre} : {resume}")

        return "\n".join(lines)

    async def _generate_response(
        self,
        question: str,
        context: str,
        domaine: str,
        history: list[dict],
    ) -> str:
        """Appelle Groq pour générer la réponse avec contexte RAG."""
        domaine_label = _DOMAINES.get(domaine, domaine)

        system_prompt = (
            f"Tu es un assistant juridique expert en {domaine_label} français. "
            "Réponds aux questions en t'appuyant sur les sources juridiques fournies. "
            "Cite précisément les articles et arrêts. "
            "Sois concis, clair et précis (500-800 tokens maximum). "
            "Si aucune source n'est disponible, indique-le et réponds à partir de tes connaissances générales."
        )

        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        messages.extend(history)

        user_content = question
        if context:
            user_content = (
                f"Question : {question}\n\n"
                f"Sources juridiques disponibles :\n{context}\n\n"
                "Réponds en citant ces sources lorsque c'est pertinent."
            )

        messages.append({"role": "user", "content": user_content})

        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._groq.chat.completions.create(
                model=_GROQ_MODEL,
                messages=messages,
                max_tokens=800,
                temperature=0.2,
            ),
        )
        return response.choices[0].message.content.strip()


# Instance partagée
chat_service = ChatService()
