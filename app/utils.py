
from langdetect import detect
import os

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

def save_output_file(transcription, translation, whisper_lang, detected_lang):
    content = (
        "=== TRANSCRIPTION ===\n" + transcription + "\n\n"
        "=== TRANSLATION (FR) ===\n" + translation + "\n\n"
        "=== LANGUAGE DETECTION ===\n"
        f"Whisper: {whisper_lang}\nLangdetect: {detected_lang}\n"
    )
    file_path = "whisper_output.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return file_path
