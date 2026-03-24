# api/routers/analysis.py — VERSION SEMAINE 3
# Pipeline complet : Whisper → Librosa → Claude → Supabase

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from supabase import create_client
from services.transcription import transcribe_audio, detect_filler_words
from services.audio_analysis import analyze_audio
from services.llm import generate_feedback
import os

router = APIRouter(prefix="/analyze", tags=["analysis"])

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)


@router.post("/")
async def analyze_speech(
    audio:   UploadFile = File(...),
    context: str        = Form("general"),
    user_id: str        = Form(None),
):
    # ── Validation ──────────────────────────────────────────
    audio_bytes = await audio.read()
    if len(audio_bytes) == 0:
        raise HTTPException(400, "Fichier audio vide")
    if len(audio_bytes) > 25 * 1024 * 1024:
        raise HTTPException(400, "Fichier trop volumineux (max 25MB)")

    # ── Étape 1 : Transcription Whisper ─────────────────────
    transcript_data = await transcribe_audio(audio_bytes, audio.filename)
    if not transcript_data["text"]:
        raise HTTPException(422, "Impossible de transcrire. Parle plus près du micro.")

    # ── Étape 2 : Métriques audio Librosa ───────────────────
    total_words   = len(transcript_data["text"].split())
    audio_metrics = analyze_audio(audio_bytes, total_words)

    # ── Étape 3 : Mots parasites ────────────────────────────
    filler_data = detect_filler_words(transcript_data["text"])

    # ── Étape 4 : Feedback Claude IA ────────────────────────
    llm_feedback = await generate_feedback(
        transcript = transcript_data["text"],
        metrics    = audio_metrics,
        fillers    = filler_data,
        context    = context
    )

    # ── Étape 5 : Sauvegarde Supabase ───────────────────────
    session_id = None
    if user_id:
        try:
            row = supabase.table("analysis_sessions").insert({
                "user_id":    user_id,
                "context":    context,
                "transcript": transcript_data["text"],
                "duration_s": audio_metrics["duration_seconds"],
                "metrics":    audio_metrics,
                "feedback":   llm_feedback,
            }).execute()
            session_id = row.data[0]["id"]
        except Exception as e:
            print(f"Erreur sauvegarde: {e}")

    return {
        "session_id":    session_id,
        "transcript":    transcript_data["text"],
        "segments":      transcript_data["segments"],
        "word_count":    total_words,
        "audio_metrics": audio_metrics,
        "filler_words":  filler_data,
        "feedback":      llm_feedback,
        "context":       context,
    }
