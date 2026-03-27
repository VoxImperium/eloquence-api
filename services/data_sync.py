"""
services/data_sync.py — Synchronisation du dump Judilibre depuis data.gouv.fr
vers la base de données PostgreSQL locale.

Flux :
  1. Interroge l'API data.gouv.fr pour trouver l'URL du dernier dump JSON/ZIP
  2. Télécharge le fichier
  3. Extrait et normalise les décisions
  4. Importe en base via jurisprudence_db.bulk_import()
"""

import io
import json
import logging
import os
import zipfile

import httpx

from .jurisprudence_db import bulk_import

logger = logging.getLogger(__name__)

# URL de l'API data.gouv.fr pour le dataset Judilibre
_DATAGOUV_DATASET_API = (
    "https://www.data.gouv.fr/api/1/datasets/"
    "judilibre-jurisprudence-des-cours-supremes-francaises/"
)

# Fallback : URL directe configurable via variable d'environnement
_ENV_DUMP_URL = "JUDILIBRE_DUMP_URL"

# Timeout HTTP pour le téléchargement du dump Judilibre (peut être volumineux)
_DUMP_DOWNLOAD_TIMEOUT = httpx.Timeout(connect=30, read=300, write=30, pool=10)


async def _discover_dump_url() -> str | None:
    """
    Interroge l'API data.gouv.fr pour obtenir l'URL du dernier dump Judilibre.
    Cherche en priorité un fichier ZIP ou JSON parmi les ressources du dataset.
    """
    env_url = os.getenv(_ENV_DUMP_URL)
    if env_url:
        logger.info("Utilisation de l'URL de dump configurée : %s", env_url)
        return env_url

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30)) as client:
            r = await client.get(_DATAGOUV_DATASET_API)
            r.raise_for_status()
            data = r.json()
    except Exception as exc:
        logger.error("Impossible d'interroger l'API data.gouv.fr : %s", exc)
        return None

    resources = data.get("resources", [])
    if not resources:
        logger.error("Aucune ressource trouvée dans le dataset Judilibre sur data.gouv.fr")
        return None

    # Priorité : ZIP > JSON, trié par date de mise à jour décroissante
    def _priority(res: dict) -> tuple:
        fmt = (res.get("format") or res.get("mime") or "").lower()
        is_zip = "zip" in fmt
        is_json = "json" in fmt
        last_modified = res.get("last_modified") or res.get("created_at") or ""
        return (is_zip, is_json, last_modified)

    resources_sorted = sorted(resources, key=_priority, reverse=True)
    best = resources_sorted[0]
    url = best.get("url") or best.get("latest")
    if url:
        logger.info(
            "Dump Judilibre trouvé : %s (format=%s, modifié=%s)",
            url,
            best.get("format", "?"),
            best.get("last_modified", "?"),
        )
    else:
        logger.error("Ressource sans URL dans le dataset Judilibre")
    return url


def _extract_decisions_from_zip(raw_bytes: bytes) -> list[dict]:
    """Extrait toutes les décisions depuis un fichier ZIP contenant des JSON."""
    decisions = []
    try:
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
            json_files = [n for n in zf.namelist() if n.lower().endswith(".json")]
            logger.info("ZIP Judilibre : %d fichier(s) JSON trouvé(s)", len(json_files))
            for name in json_files:
                try:
                    with zf.open(name) as f:
                        content = json.load(f)
                    if isinstance(content, list):
                        decisions.extend(content)
                    elif isinstance(content, dict):
                        # Format possible : {"results": [...]}
                        if "results" in content:
                            decisions.extend(content["results"])
                        else:
                            decisions.append(content)
                    logger.debug("Fichier %s : %d décisions", name, len(decisions))
                except Exception as exc:
                    logger.warning("Erreur lecture %s dans le ZIP : %s", name, exc)
    except zipfile.BadZipFile as exc:
        logger.error("Fichier ZIP invalide : %s", exc)
    return decisions


def _extract_decisions_from_json(raw_bytes: bytes) -> list[dict]:
    """Extrait les décisions depuis un fichier JSON brut."""
    try:
        content = json.loads(raw_bytes)
        if isinstance(content, list):
            return content
        if isinstance(content, dict):
            if "results" in content:
                return content["results"]
            return [content]
    except Exception as exc:
        logger.error("Erreur de parsing JSON : %s", exc)
    return []


def _normalize_decision(raw: dict) -> dict:
    """Normalise un enregistrement brut Judilibre en format uniforme."""
    themes = raw.get("themes", [])
    if isinstance(themes, str):
        themes = [themes]

    return {
        "id":          str(raw.get("id", "")),
        "date":        str(raw.get("decision_date", "") or raw.get("date", "")),
        "chambre":     str(raw.get("chamber", "") or raw.get("chambre", "")),
        "solution":    str(raw.get("solution", "")),
        "resume":      str(raw.get("summary", "") or raw.get("resume", ""))[:2000],
        "themes":      themes,
        "numero":      str(raw.get("number", "") or raw.get("numero", "")),
        "juridiction": str(raw.get("jurisdiction", "") or raw.get("juridiction", "")),
        "texte":       str(raw.get("text", "") or raw.get("texte", ""))[:10000],
        "source":      "judilibre",
    }


async def run_sync() -> dict:
    """
    Télécharge et importe le dernier dump Judilibre dans PostgreSQL.

    Returns:
        Dictionnaire avec les statistiques de synchronisation :
        {success, downloaded_bytes, decisions_found, decisions_imported, error}
    """
    stats: dict = {
        "success": False,
        "downloaded_bytes": 0,
        "decisions_found": 0,
        "decisions_imported": 0,
        "error": None,
    }

    # 1. Découvrir l'URL du dump
    dump_url = await _discover_dump_url()
    if not dump_url:
        stats["error"] = "Impossible de localiser le dump Judilibre sur data.gouv.fr"
        logger.error(stats["error"])
        return stats

    # 2. Télécharger le dump
    logger.info("Téléchargement du dump Judilibre : %s", dump_url)
    try:
        async with httpx.AsyncClient(timeout=_DUMP_DOWNLOAD_TIMEOUT, follow_redirects=True) as client:
            r = await client.get(dump_url)
            r.raise_for_status()
            raw_bytes = r.content
            stats["downloaded_bytes"] = len(raw_bytes)
        logger.info("Dump téléchargé : %.1f Mo", len(raw_bytes) / 1_048_576)
    except httpx.TimeoutException as exc:
        stats["error"] = f"Timeout téléchargement : {exc}"
        logger.error(stats["error"])
        return stats
    except Exception as exc:
        stats["error"] = f"Erreur téléchargement : {exc}"
        logger.error(stats["error"])
        return stats

    # 3. Extraire les décisions
    content_type = ""
    if hasattr(r, "headers"):
        content_type = r.headers.get("content-type", "")

    is_zip = (
        dump_url.lower().endswith(".zip")
        or "zip" in content_type.lower()
        or raw_bytes[:4] == b"PK\x03\x04"
    )

    if is_zip:
        raw_decisions = _extract_decisions_from_zip(raw_bytes)
    else:
        raw_decisions = _extract_decisions_from_json(raw_bytes)

    stats["decisions_found"] = len(raw_decisions)
    logger.info("%d décisions extraites du dump", stats["decisions_found"])

    if not raw_decisions:
        stats["error"] = "Aucune décision trouvée dans le dump"
        logger.warning(stats["error"])
        return stats

    # 4. Normaliser et importer
    normalized = [_normalize_decision(d) for d in raw_decisions if d.get("id")]
    logger.info("Import en base de %d décisions valides…", len(normalized))

    imported = await bulk_import(normalized)
    stats["decisions_imported"] = imported
    stats["success"] = imported > 0

    if stats["success"]:
        logger.info("Synchronisation Judilibre terminée : %d/%d décisions importées", imported, len(normalized))
    else:
        stats["error"] = "Aucune décision importée (vérifier la connexion DB)"
        logger.error(stats["error"])

    return stats
