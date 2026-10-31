"""Central settings, read once from the environment (.env locally, Space secrets in production)."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


def _get(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


GEMINI_API_KEY = _get("GEMINI_API_KEY")
GEMINI_MODEL = _get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_FALLBACK_MODEL = _get("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash-lite")
GEMINI_EMBED_MODEL = _get("GEMINI_EMBED_MODEL", "gemini-embedding-001")
GEMINI_BASE_URL = _get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai").rstrip("/")

DATA_GOV_IN_API_KEY = _get("DATA_GOV_IN_API_KEY")
AGMARKNET_RESOURCE_ID = _get("AGMARKNET_RESOURCE_ID", "9ef84268-d588-465a-a308-a864a43d0070")
AGMARKNET_HISTORY_RESOURCE_ID = _get("AGMARKNET_HISTORY_RESOURCE_ID", "35985678-0d79-46b4-9ed6-6f13308a1d24")

SARVAM_API_KEY = _get("SARVAM_API_KEY")
SARVAM_TTS_MODEL = _get("SARVAM_TTS_MODEL", "bulbul:v3")
SARVAM_TTS_SPEAKER = _get("SARVAM_TTS_SPEAKER")
SARVAM_STT_MODEL = _get("SARVAM_STT_MODEL", "saarika:v2.5")
SARVAM_TRANSLATE_MODEL = _get("SARVAM_TRANSLATE_MODEL", "mayura:v1")

OSRM_BASE_URL = _get("OSRM_BASE_URL", "https://router.project-osrm.org").rstrip("/")
ORS_API_KEY = _get("ORS_API_KEY")
NOMINATIM_USER_AGENT = _get("NOMINATIM_USER_AGENT", "KhetRakshak/0.1")

DATABASE_URL = _get("DATABASE_URL")
TELEGRAM_BOT_TOKEN = _get("TELEGRAM_BOT_TOKEN")
TELEGRAM_WEBHOOK_SECRET = _get("TELEGRAM_WEBHOOK_SECRET")
PUBLIC_BASE_URL = _get("PUBLIC_BASE_URL").rstrip("/")
ADMIN_TOKEN = _get("ADMIN_TOKEN")

INDEX_DIR = ROOT / "data" / "index"
STATIC_DIR = ROOT / "backend" / "static"

USER_AGENT = "KhetRakshak/0.1 (+https://github.com/khetrakshak)"

SECRETS = [GEMINI_API_KEY, DATA_GOV_IN_API_KEY, SARVAM_API_KEY, ORS_API_KEY,
           TELEGRAM_BOT_TOKEN, DATABASE_URL, ADMIN_TOKEN]
