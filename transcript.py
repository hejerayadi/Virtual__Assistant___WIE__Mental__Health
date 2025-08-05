import whisper

# Load Whisper model once globally (e.g. base, small, etc.)
model = whisper.load_model("base")

def transcribe_with_language(filepath):
    result = model.transcribe(filepath)  # Automatically detects language

    transcription = result["text"]
    language_code = result["language"]

    return {
        "transcription": transcription,
        "language_code": language_code,
        "language_name": whisper.tokenizer.LANGUAGES.get(language_code, "Unknown")
    }
