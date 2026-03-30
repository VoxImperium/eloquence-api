import os
import time
import logging
import psutil

logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("LEGALBERT_MODEL", "nlpaueb/legal-bert-base-uncased")

classifier = None

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        ignore_mismatched_sizes=True,
    )

    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=-1,
    )
    logger.info("✅ LEGAL-BERT EU chargé en FP32 (~420MB RAM) — modèle : %s", MODEL_NAME)
except Exception as exc:
    logger.error("❌ Erreur chargement LEGAL-BERT EU : %s", exc)
    classifier = None


async def analyze_legal_text(text: str) -> dict:
    """Analyse un texte légal EU avec LEGAL-BERT EU (FP32, sans coût d'API)."""
    if not text or len(text.strip()) < 10:
        return {
            "error": "Texte trop court (minimum 10 caractères)",
            "error_type": "validation_error",
            "text": text,
        }

    if classifier is None:
        return {
            "error": "Modèle LEGAL-BERT non disponible",
            "error_type": "model_unavailable",
            "text": text,
        }

    try:
        process = psutil.Process(os.getpid())

        start = time.perf_counter()
        result = classifier(text[:512])
        latency_ms = round((time.perf_counter() - start) * 1000, 1)

        # Report total RSS so the value is always meaningful (not a delta)
        memory_rss_mb = round(process.memory_info().rss / (1024 * 1024), 1)

        return {
            "text": text[:512],
            "results": result,
            "source": "legal-bert-eu",
            "model": MODEL_NAME,
            "latency_ms": latency_ms,
            "memory_usage_mb": memory_rss_mb,
        }
    except Exception as exc:
        logger.error("Erreur analyse LEGAL-BERT : %s", exc)
        return {
            "error": str(exc),
            "error_type": "inference_error",
            "text": text,
        }
