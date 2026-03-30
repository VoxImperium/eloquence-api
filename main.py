# api/main.py — VERSION SEMAINE 7
from dotenv import load_dotenv
import os

load_dotenv()

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import analysis, chat as chat_router, emails as emails_router, legifrance as legifrance_router, payments, simulation, training, speech_analysis, legalbert as legalbert_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ────────────────────────────────────────────────────────────────
    logging.basicConfig(level=logging.INFO)
    logger.info("Démarrage de l'API Éloquence")
    from services.legalbert_service import classifier as _legalbert_classifier
    if _legalbert_classifier is not None:
        logger.info("✅ LEGAL-BERT EU chargé en FP16")
    else:
        logger.warning("⚠️ LEGAL-BERT EU non disponible")

    yield

    # ── Shutdown ───────────────────────────────────────────────────────────────
    logger.info("Arrêt de l'API Éloquence")


app = FastAPI(title="Éloquence API", version="0.7.1", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analysis.router)
app.include_router(chat_router.router)
app.include_router(emails_router.router)
app.include_router(legifrance_router.router)
app.include_router(payments.router)
app.include_router(simulation.router)
app.include_router(training.router)
app.include_router(speech_analysis.router)
app.include_router(legalbert_router.router)

@app.get("/")
def root():
    return {"status": "ok", "version": "0.7.1"}

@app.get("/health")
def health():
    return {"status": "healthy"}

