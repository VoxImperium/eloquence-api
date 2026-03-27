# api/main.py — VERSION SEMAINE 6
from dotenv import load_dotenv
import os

load_dotenv()

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from routers import analysis, emails as emails_router, legifrance as legifrance_router, payments, simulation, training, speech_analysis
import services.jurisprudence_db as jurisprudence_db
from services.data_sync import run_sync

logger = logging.getLogger(__name__)

scheduler = AsyncIOScheduler(timezone="Europe/Paris")
_initial_sync_task: asyncio.Task | None = None


async def _scheduled_sync() -> None:
    """Tâche planifiée : synchronisation quotidienne du dump Judilibre."""
    logger.info("Démarrage de la synchronisation quotidienne Judilibre…")
    try:
        stats = await run_sync()
        if stats["success"]:
            logger.info(
                "Sync Judilibre OK — %d décisions importées",
                stats["decisions_imported"],
            )
        else:
            logger.warning("Sync Judilibre terminée avec erreur : %s", stats.get("error"))
    except Exception as exc:
        logger.error("Erreur inattendue lors de la sync Judilibre : %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ────────────────────────────────────────────────────────────────
    logging.basicConfig(level=logging.INFO)

    # Initialiser le pool PostgreSQL
    await jurisprudence_db.init_pool()

    # Synchronisation initiale en arrière-plan (ne bloque pas le démarrage)
    if jurisprudence_db.get_pool() is not None:
        count = await jurisprudence_db.get_decision_count()
        if count == 0:
            logger.info("Base vide — lancement de la sync initiale Judilibre en arrière-plan")

            async def _initial_sync_with_logging() -> None:
                try:
                    await _scheduled_sync()
                except Exception as exc:
                    logger.error("Sync initiale Judilibre échouée : %s", exc)

            global _initial_sync_task
            _initial_sync_task = asyncio.create_task(_initial_sync_with_logging())
        else:
            logger.info("Base existante : %d décisions — sync initiale ignorée", count)
    else:
        logger.warning("DB non disponible — sync Judilibre désactivée")

    # Planifier la synchronisation quotidienne à 3h00
    scheduler.add_job(
        _scheduled_sync,
        trigger="cron",
        hour=3,
        minute=0,
        id="judilibre_daily_sync",
        replace_existing=True,
    )
    scheduler.start()
    logger.info("Scheduler démarré — sync Judilibre planifiée à 03h00 Europe/Paris")

    yield

    # ── Shutdown ───────────────────────────────────────────────────────────────
    scheduler.shutdown(wait=False)
    if _initial_sync_task is not None and not _initial_sync_task.done():
        logger.info("Annulation de la sync initiale en cours…")
        _initial_sync_task.cancel()
    await jurisprudence_db.close_pool()
    logger.info("Arrêt propre — scheduler et pool PostgreSQL fermés")


app = FastAPI(title="Éloquence API", version="0.6.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analysis.router)
app.include_router(emails_router.router)
app.include_router(legifrance_router.router)
app.include_router(payments.router)
app.include_router(simulation.router)
app.include_router(training.router)
app.include_router(speech_analysis.router)

@app.get("/")
def root():
    return {"status": "ok", "version": "0.6.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}
