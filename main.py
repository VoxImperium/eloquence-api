from dotenv import load_dotenv
load_dotenv()

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import analysis, emails as emails_router, legifrance as legifrance_router, payments, simulation, training, speech_analysis

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO)
    logger.info("Éloquence API démarrée — OpenLegi MCP activé")
    yield
    logger.info("Arrêt propre")

app = FastAPI(title="Éloquence API", version="0.7.0", lifespan=lifespan)

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
    return {"status": "ok", "version": "0.7.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}

