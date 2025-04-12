
import gradio as gr
import whisper
from utils import detect_language, save_output_file

model = whisper.load_model("medium")

def transcribe_and_translate(audio_path):
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    whisper_probs = model.detect_language(mel)[1]
    top_lang = max(whisper_probs, key=whisper_probs.get)

    result = model.transcribe(audio_path, fp16=False)
    transcription = result["text"]

    fallback_lang = detect_language(transcription)

    translation = model.transcribe(audio_path, language="fr", fp16=False)["text"]

    file_path = save_output_file(transcription, translation, top_lang, fallback_lang)

    return f"Whisper: {top_lang} | langdetect: {fallback_lang}", transcription, translation, file_path

app = gr.Interface(
    fn=transcribe_and_translate,
    inputs=gr.Audio(type="filepath", label="🎧 Upload Audio File"),
    outputs=[
        gr.Text(label="🌐 Language Detection"),
        gr.Textbox(label="📝 Transcription"),
        gr.Textbox(label="🇫🇷 Translation (French)"),
        gr.File(label="📄 Download Result (.txt)"),
    ],
    title="🗣️ Whisper Transcriber & Translator",
    description="Upload audio to get transcription, French translation, and language detection.",
)

app.launch(share=True)