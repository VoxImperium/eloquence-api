#!/usr/bin/env python3
"""
scripts/init_db.py — Initialisation manuelle de la base de données PostgreSQL
et synchronisation initiale du dump Judilibre.

Usage :
    python scripts/init_db.py [--sync]

Options :
    --sync    Lance immédiatement le téléchargement et l'import du dump Judilibre
              (peut prendre plusieurs minutes selon la taille du dump)

Variables d'environnement requises :
    DATABASE_URL    ex. postgresql://eloquence:password@localhost:5432/eloquence_db

Variables optionnelles :
    JUDILIBRE_DUMP_URL   URL directe vers le dump ZIP/JSON (sinon découverte auto)
"""

import asyncio
import logging
import sys
import os

# Remonter d'un niveau pour accéder aux services
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import services.jurisprudence_db as jurisprudence_db
from services.data_sync import run_sync

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("init_db")


async def main() -> None:
    run_sync_flag = "--sync" in sys.argv

    logger.info("=== Initialisation de la base de données Éloquence ===")

    # Vérifier DATABASE_URL
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        logger.error("La variable DATABASE_URL n'est pas définie.")
        logger.error("Définissez-la dans .env ou en variable d'environnement :")
        logger.error("  DATABASE_URL=postgresql://user:password@host:5432/dbname")
        sys.exit(1)

    logger.info("DATABASE_URL détectée (host masqué pour la sécurité)")

    # Initialiser le pool et créer le schéma
    await jurisprudence_db.init_pool()

    if jurisprudence_db.get_pool() is None:
        logger.error("Impossible de se connecter à PostgreSQL — vérifiez DATABASE_URL et que le serveur est démarré")
        sys.exit(1)

    health = await jurisprudence_db.healthcheck()
    logger.info("Connexion OK — %d décisions en base", health["decisions_count"])

    if run_sync_flag:
        logger.info("Lancement de la synchronisation du dump Judilibre…")
        logger.info("(Cela peut prendre plusieurs minutes selon la taille du dump)")
        stats = await run_sync()

        if stats["success"]:
            logger.info(
                "✅ Synchronisation réussie : %d décisions importées (%.1f Mo téléchargés)",
                stats["decisions_imported"],
                stats["downloaded_bytes"] / 1_048_576,
            )
        else:
            logger.error("❌ Synchronisation échouée : %s", stats.get("error"))
            logger.info(
                "   Vous pouvez spécifier l'URL du dump manuellement :"
                "\n   JUDILIBRE_DUMP_URL=<url> python scripts/init_db.py --sync"
            )
    else:
        count = health["decisions_count"]
        if count == 0:
            logger.info(
                "Base vide. Lancez la sync avec :\n"
                "  python scripts/init_db.py --sync"
            )
        else:
            logger.info(
                "Base prête avec %d décisions. Aucune action supplémentaire requise.", count
            )

    await jurisprudence_db.close_pool()
    logger.info("=== Initialisation terminée ===")


if __name__ == "__main__":
    asyncio.run(main())
