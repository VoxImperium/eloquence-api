import librosa
import numpy as np
import tempfile
import os
import subprocess


def analyze_audio(audio_bytes: bytes, total_words: int) -> dict:
    # Sauvegarder le fichier webm temporairement
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp_in:
        tmp_in.write(audio_bytes)
        tmp_webm = tmp_in.name

    # Convertir webm → wav avec ffmpeg
    tmp_wav = tmp_webm.replace(".webm", ".wav")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_webm, "-ar", "16000", "-ac", "1", tmp_wav],
            capture_output=True, check=True
        )
    except Exception as e:
        os.unlink(tmp_webm)
        # Retourner des métriques vides si conversion échoue
        return _empty_metrics()

    try:
        y, sr = librosa.load(tmp_wav, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)

        speech_rate = (total_words / duration) * 60 if duration > 0 else 0

        intervals = librosa.effects.split(y, top_db=30)
        pauses = []
        for i in range(1, len(intervals)):
            gap = (intervals[i][0] - intervals[i-1][1]) / sr
            if gap > 0.3:
                pauses.append(round(gap, 2))

        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_vals = pitches[magnitudes > np.percentile(magnitudes, 75)]
        pitch_vals = pitch_vals[pitch_vals > 50]
        pitch_std  = float(np.std(pitch_vals))  if len(pitch_vals) > 0 else 0
        pitch_mean = float(np.mean(pitch_vals)) if len(pitch_vals) > 0 else 0

        rms = librosa.feature.rms(y=y)[0]

        return {
            "duration_seconds":   round(duration, 1),
            "speech_rate_wpm":    round(speech_rate, 1),
            "speech_rate_rating": _rate_speech_rate(speech_rate),
            "pause_count":        len(pauses),
            "pauses_seconds":     pauses,
            "avg_pause_s":        round(np.mean(pauses), 2) if pauses else 0,
            "pitch_variation":    round(pitch_std, 1),
            "pitch_mean_hz":      round(pitch_mean, 1),
            "pitch_rating":       _rate_pitch(pitch_std),
            "energy_mean":        round(float(np.mean(rms)), 4),
            "energy_variation":   round(float(np.std(rms)), 4),
        }

    finally:
        os.unlink(tmp_webm)
        if os.path.exists(tmp_wav):
            os.unlink(tmp_wav)


def _empty_metrics() -> dict:
    return {
        "duration_seconds": 0, "speech_rate_wpm": 0,
        "speech_rate_rating": "inconnu", "pause_count": 0,
        "pauses_seconds": [], "avg_pause_s": 0,
        "pitch_variation": 0, "pitch_mean_hz": 0,
        "pitch_rating": "inconnu", "energy_mean": 0, "energy_variation": 0,
    }

def _rate_speech_rate(wpm):
    if wpm < 80:  return "trop_lent"
    if wpm < 120: return "lent"
    if wpm < 160: return "optimal"
    if wpm < 200: return "rapide"
    return "trop_rapide"

def _rate_pitch(std):
    if std < 20:  return "monotone"
    if std < 50:  return "peu_expressif"
    if std < 100: return "expressif"
    return "très_expressif"
