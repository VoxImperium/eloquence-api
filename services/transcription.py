# api/services/transcription.py
# Transcription audio → texte avec Groq Whisper (gratuit)

import os
import tempfile
from pathlib import Path
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


async def transcribe_audio(audio_bytes: bytes, filename: str = "audio.wav") -> dict:
    """
    Transcrit un fichier audio en texte avec Groq Whisper.
    Gratuit, très rapide (~2-3 secondes).
    """
    suffix = Path(filename).suffix or ".wav"
    if suffix == ".webm":
        suffix = ".webm"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=(Path(tmp_path).name, audio_file, "audio/webm"),
                language="fr",
                response_format="verbose_json",
            )

        # Extraire les segments si disponibles
        segments = []
        if hasattr(response, "segments") and response.segments:
            for seg in response.segments:
                segments.append({
                    "start": getattr(seg, "start", 0),
                    "end":   getattr(seg, "end", 0),
                    "text":  getattr(seg, "text", "").strip()
                })

        text = getattr(response, "text", "").strip()

        return {
            "text":     text,
            "segments": segments,
            "language": getattr(response, "language", "fr"),
            "duration": getattr(response, "duration", 0)
        }

    finally:
        os.unlink(tmp_path)


def detect_filler_words(text: str) -> dict:
    """
    Détecte les mots parasites dans le texte transcrit.
    """
    FILLER_WORDS = [
        "euh", "heu", "bah", "ben", "voilà", "donc",
        "en fait", "du coup", "genre", "c'est-à-dire",
        "quoi", "hein", "bon", "alors", "eh bien"
    ]

    text_lower = text.lower()
    found = {}

    for word in FILLER_WORDS:
        count = text_lower.count(word)
        if count > 0:
            found[word] = count

    return {
        "total":   sum(found.values()),
        "details": found
    }
