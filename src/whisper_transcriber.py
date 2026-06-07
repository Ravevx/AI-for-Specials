"""
Whisper Transcriber
-------------------
Transcribes audio files or live microphone input using OpenAI Whisper.
"""

import whisper
import numpy as np
import tempfile
import os
import config


class WhisperTranscriber:
    def __init__(self):
        print(f"Loading Whisper '{config.WHISPER_MODEL}' model...")
        self.model = whisper.load_model(config.WHISPER_MODEL)
        print("Whisper loaded ✅")

    def transcribe_file(self, audio_path: str) -> str:
        """Transcribe an audio file (mp3, wav, m4a, webm, etc.)"""
        result = self.model.transcribe(
            audio_path,
            language=config.WHISPER_LANGUAGE,
            fp16=False
        )
        return result["text"].strip()

    def record_and_transcribe(self, duration: int = 5) -> str:
        """Record from microphone then transcribe."""
        try:
            import sounddevice as sd
            import soundfile as sf
        except ImportError:
            raise ImportError(
                "Install sounddevice + soundfile: pip install sounddevice soundfile"
            )

        print(f"🎤 Recording for {duration} seconds...")
        sample_rate = 16000
        audio_data  = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32"
        )
        sd.wait()
        print("Recording done. Transcribing...")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio_data, sample_rate)
            text = self.transcribe_file(tmp.name)
        os.unlink(tmp.name)
        return text