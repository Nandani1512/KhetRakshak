"""Voice + language tool (Sarvam AI): translate, text-to-speech, speech-to-text.

Audio in any container (Telegram OGG/Opus, browser WebM) is normalised to 16 kHz mono WAV
with ffmpeg first. Long text is split into sentence chunks to stay under per-request limits.
"""
from __future__ import annotations

import asyncio
import base64
import io
import re
import subprocess
import wave

import imageio_ffmpeg

from app import config
from app.tools.base import http

SARVAM = "https://api.sarvam.ai"
LANGUAGES = {"hi-IN": "Hindi", "ta-IN": "Tamil", "pa-IN": "Punjabi", "en-IN": "English"}
TRANSLATE_CHUNK = 900
TTS_CHUNK = 450


def _headers() -> dict:
    return {"api-subscription-key": config.SARVAM_API_KEY}


def _chunks(text: str, limit: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?।])\s+", text.strip())
    out, cur = [], ""
    for s in sentences:
        while len(s) > limit:  # pathological long sentence
            out.append(s[:limit])
            s = s[limit:]
        if len(cur) + len(s) + 1 > limit and cur:
            out.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        out.append(cur)
    return out


async def translate(text: str, target: str, source: str = "en-IN") -> str:
    if target == source:
        return text
    async def one(chunk: str) -> str:
        resp = await http("POST", f"{SARVAM}/translate", source="Sarvam", headers=_headers(), json={
            "input": chunk, "source_language_code": source, "target_language_code": target,
            "model": config.SARVAM_TRANSLATE_MODEL})
        return resp.json()["translated_text"]

    async def line(text_line: str) -> str:
        if not text_line.strip():
            return ""
        return " ".join(await asyncio.gather(*(one(c) for c in _chunks(text_line, TRANSLATE_CHUNK))))

    # Line by line keeps the alert's structure; lines run concurrently so it stays fast.
    return "\n".join(await asyncio.gather(*(line(t) for t in text.split("\n"))))


def _join_wavs(blobs: list[bytes]) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as dst:
        for i, blob in enumerate(blobs):
            with wave.open(io.BytesIO(blob), "rb") as src:
                if i == 0:
                    dst.setparams(src.getparams())
                dst.writeframes(src.readframes(src.getnframes()))
    return out.getvalue()


async def tts(text: str, language: str) -> bytes:
    """Speech for the whole text as one WAV."""
    blobs = []
    for chunk in _chunks(text, TTS_CHUNK):
        body = {"text": chunk, "target_language_code": language, "model": config.SARVAM_TTS_MODEL}
        if config.SARVAM_TTS_SPEAKER:
            body["speaker"] = config.SARVAM_TTS_SPEAKER
        resp = await http("POST", f"{SARVAM}/text-to-speech", source="Sarvam", headers=_headers(),
                          json=body, timeout=60.0)
        blobs.append(base64.b64decode(resp.json()["audios"][0]))
    return _join_wavs(blobs)


def _ffmpeg(audio: bytes, args: list[str]) -> bytes:
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-i", "pipe:0", *args, "pipe:1"]
    proc = subprocess.run(cmd, input=audio, capture_output=True, timeout=60, check=True)
    return proc.stdout


def to_wav16k(audio: bytes) -> bytes:
    return _ffmpeg(audio, ["-ac", "1", "-ar", "16000", "-f", "wav"])


def wav_to_ogg_opus(audio: bytes) -> bytes:
    """Telegram shows a voice-note bubble only for OGG/Opus."""
    return _ffmpeg(audio, ["-c:a", "libopus", "-b:a", "32k", "-f", "ogg"])


async def stt(audio: bytes, language: str | None = None) -> dict:
    """Transcribe any audio. language=None lets Sarvam auto-detect (handles Hinglish/code-mix)."""
    wav = to_wav16k(audio)
    data = {"model": config.SARVAM_STT_MODEL}
    data["language_code"] = language or "unknown"
    resp = await http("POST", f"{SARVAM}/speech-to-text", source="Sarvam", headers=_headers(),
                      files={"file": ("audio.wav", wav, "audio/wav")}, data=data, timeout=60.0)
    body = resp.json()
    return {"transcript": (body.get("transcript") or "").strip(),
            "language": body.get("language_code") or language}
