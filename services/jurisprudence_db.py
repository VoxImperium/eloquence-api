"""
services/jurisprudence_db.py — Gestion de la base de données PostgreSQL locale
pour la jurisprudence Judilibre avec recherche full-text en français.
"""

import logging
import os
from typing import Any

import asyncpg

logger = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None

# DDL ──────────────────────────────────────────────────────────────────────────

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS decisions (
    id           TEXT PRIMARY KEY,
    date         TEXT,
    chambre      TEXT,
    solution     TEXT,
    resume       TEXT,
    themes       TEXT,
    numero       TEXT,
    juridiction  TEXT,
    texte        TEXT,
    source       TEXT DEFAULT 'judilibre',
    tsv          TSVECTOR
);
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS decisions_tsv_idx ON decisions USING GIN(tsv);
"""

_UPSERT_DECISION = """
INSERT INTO decisions (id, date, chambre, solution, resume, themes, numero, juridiction, texte, source, tsv)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
        to_tsvector('french',
            coalesce($4, '') || ' ' ||
            coalesce($5, '') || ' ' ||
            coalesce($6, '') || ' ' ||
            coalesce($9, '')
        ))
ON CONFLICT (id) DO UPDATE SET
    date        = EXCLUDED.date,
    chambre     = EXCLUDED.chambre,
    solution    = EXCLUDED.solution,
    resume      = EXCLUDED.resume,
    themes      = EXCLUDED.themes,
    numero      = EXCLUDED.numero,
    juridiction = EXCLUDED.juridiction,
    texte       = EXCLUDED.texte,
    source      = EXCLUDED.source,
    tsv         = EXCLUDED.tsv;
"""

_SEARCH_DECISIONS = """
SELECT id, date, chambre, solution, resume, themes, numero, juridiction,
       ts_rank(tsv, query) AS rank
FROM decisions, to_tsquery('french', $1) AS query
WHERE tsv @@ query
ORDER BY rank DESC
LIMIT $2;
"""

_COUNT_DECISIONS = "SELECT COUNT(*) FROM decisions;"

_HEALTH_CHECK = "SELECT 1;"


# Pool lifecycle ───────────────────────────────────────────────────────────────

async def init_pool() -> None:
    """Initialise le pool de connexions AsyncPG."""
    global _pool
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        logger.warning("DATABASE_URL non configuré — base de données jurisprudence désactivée")
        return
    try:
        _pool = await asyncpg.create_pool(dsn, min_size=2, max_size=10)
        await _ensure_schema()
        count = await get_decision_count()
        logger.info("Pool PostgreSQL initialisé — %d décisions en base", count)
    except Exception as exc:
        logger.error("Impossible d'initialiser le pool PostgreSQL : %s", exc)
        _pool = None


async def close_pool() -> None:
    """Ferme proprement le pool de connexions."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("Pool PostgreSQL fermé")


def get_pool() -> asyncpg.Pool | None:
    """Retourne le pool courant (None si non initialisé)."""
    return _pool


# Schema ───────────────────────────────────────────────────────────────────────

async def _ensure_schema() -> None:
    """Crée la table et l'index si nécessaire."""
    async with _pool.acquire() as conn:
        await conn.execute(_CREATE_TABLE)
        await conn.execute(_CREATE_INDEX)
    logger.debug("Schéma PostgreSQL vérifié/créé")


# Public API ───────────────────────────────────────────────────────────────────

async def search_decisions(query: str, limit: int = 5) -> list[dict]:
    """
    Recherche full-text en français dans la base locale.

    Args:
        query: Termes de recherche (ex. "responsabilité civile accident").
        limit: Nombre maximum de résultats.

    Returns:
        Liste de décisions triées par pertinence décroissante.
    """
    if _pool is None:
        logger.warning("DB non initialisée — search_decisions renvoie []")
        return []

    # Transformer la requête en tsquery (AND implicite entre les termes)
    tsquery = _build_tsquery(query)
    if not tsquery:
        return []

    try:
        async with _pool.acquire() as conn:
            rows = await conn.fetch(_SEARCH_DECISIONS, tsquery, limit)
        return [_row_to_dict(r) for r in rows]
    except Exception as exc:
        logger.error("Erreur lors de la recherche dans la DB : %s", exc)
        return []


async def bulk_import(decisions: list[dict]) -> int:
    """
    Importe en masse des décisions dans PostgreSQL.

    Args:
        decisions: Liste de dictionnaires normalisés (format Judilibre).

    Returns:
        Nombre de décisions importées avec succès.
    """
    if _pool is None:
        logger.warning("DB non initialisée — bulk_import ignoré")
        return 0

    imported = 0
    BATCH = 500

    for i in range(0, len(decisions), BATCH):
        batch = decisions[i: i + BATCH]
        records = [_decision_to_record(d) for d in batch]
        try:
            async with _pool.acquire() as conn:
                await conn.executemany(_UPSERT_DECISION, records)
            imported += len(batch)
            logger.debug("Importé %d/%d décisions", imported, len(decisions))
        except Exception as exc:
            logger.error("Erreur import batch %d-%d : %s", i, i + BATCH, exc)

    return imported


async def get_decision_count() -> int:
    """Retourne le nombre total de décisions en base."""
    if _pool is None:
        return 0
    try:
        async with _pool.acquire() as conn:
            row = await conn.fetchrow(_COUNT_DECISIONS)
        return int(row[0])
    except Exception:
        return 0


async def healthcheck() -> dict[str, Any]:
    """Vérifie la connexion et retourne les statistiques de la DB."""
    if _pool is None:
        dsn_set = bool(os.getenv("DATABASE_URL"))
        return {
            "status": "unavailable",
            "message": "Pool non initialisé" + (" — DATABASE_URL manquant" if not dsn_set else ""),
            "decisions_count": 0,
        }
    try:
        async with _pool.acquire() as conn:
            await conn.fetchrow(_HEALTH_CHECK)
        count = await get_decision_count()
        return {
            "status": "ok",
            "message": "Connexion PostgreSQL active",
            "decisions_count": count,
        }
    except Exception as exc:
        return {
            "status": "error",
            "message": str(exc),
            "decisions_count": 0,
        }


# Helpers ──────────────────────────────────────────────────────────────────────

def _build_tsquery(query: str) -> str:
    """
    Convertit une chaîne de recherche libre en expression tsquery PostgreSQL.
    Les termes sont combinés avec OR pour maximiser le rappel.
    """
    tokens = [t.strip() for t in query.split() if len(t.strip()) >= 2]
    if not tokens:
        return ""
    return " | ".join(tokens)


def _decision_to_record(d: dict) -> tuple:
    """Transforme un dict décision en tuple pour executemany."""
    themes_str = ""
    themes = d.get("themes", [])
    if isinstance(themes, list):
        themes_str = " ".join(str(t) for t in themes)
    elif isinstance(themes, str):
        themes_str = themes

    return (
        str(d.get("id", "")),
        str(d.get("date", "") or d.get("decision_date", "")),
        str(d.get("chambre", "") or d.get("chamber", "")),
        str(d.get("solution", "")),
        str(d.get("resume", "") or d.get("summary", ""))[:2000],
        themes_str,
        str(d.get("numero", "") or d.get("number", "")),
        str(d.get("juridiction", "") or d.get("jurisdiction", "")),
        str(d.get("texte", "") or d.get("text", ""))[:10000],
        str(d.get("source", "judilibre")),
    )


def _row_to_dict(row: asyncpg.Record) -> dict:
    """Transforme un enregistrement DB en dict normalisé."""
    return {
        "id":          row["id"],
        "date":        row["date"],
        "chambre":     row["chambre"],
        "solution":    row["solution"],
        "resume":      row["resume"],
        "themes":      row["themes"].split() if row["themes"] else [],
        "numero":      row["numero"],
        "juridiction": row["juridiction"],
        "source":      "judilibre_local",
        "rank":        float(row["rank"]),
    }
