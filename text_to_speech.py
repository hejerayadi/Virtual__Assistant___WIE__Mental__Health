from gtts import gTTS
from gtts.lang import tts_langs
import os

# Preload supported language codes from gTTS
GTTS_LANGS = tts_langs()

def get_gtts_lang(whisper_code):
    """
    Normalize Whisper's ISO‑639‑1 code to a lang supported by gTTS.
    Whisper returns simple codes like 'es', 'fr', 'ar', etc.; gTTS can handle these directly.
    If Whisper returns an unsupported code, fallback to 'en'.
    """
    code = whisper_code.lower()
    # Exact match
    if code in GTTS_LANGS:
        return code
    # Some languages Whisper returns variants like 'es-mx'
    if "-" in code:
        prefix = code.split("-", 1)[0]
        if prefix in GTTS_LANGS:
            return prefix
    return "en"

def synthesize_speech(text, lang_code="en", output_path="static/llm_response.mp3"):
    lang = get_gtts_lang(lang_code)
    print(f"[TTS] gTTS will speak in language: {lang}")  # ✅ Add this
    tts = gTTS(text=text, lang=lang)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tts.save(output_path)
    return os.path.basename(output_path)
