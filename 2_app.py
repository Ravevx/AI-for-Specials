"""
Audio / Text → ASL Sign Language Avatar
Run: python 2_app.py → http://localhost:7860
"""

import gradio as gr
import os
import sys

# Add src to path if running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from whisper_transcriber import WhisperTranscriber
from text_simplifier     import TextSimplifier
from pose_generator      import PoseGenerator

print("Loading models...")
transcriber = WhisperTranscriber()
simplifier  = TextSimplifier()
generator   = PoseGenerator()
print("All models ready ✅")

lm_available = simplifier.is_available()
print(f"LM Studio: {'✅ connected' if lm_available else '⚠️ not available (rule-based fallback active)'}")

os.makedirs("./output", exist_ok=True)


def process_audio(audio_input, use_lm_studio: bool):
    if audio_input is None:
        return None, "⚠️ Please upload or record audio first."

    transcript = transcriber.transcribe_file(audio_input)
    if not transcript.strip():
        return None, "⚠️ Could not transcribe. Please speak clearly or check your audio."

    simplified = simplifier.simplify(transcript) if use_lm_studio else \
                 simplifier._rule_based_simplify(transcript)

    output_path = "./output/asl_audio_output.gif"
    video_path, glosses, n = generator.text_to_video(simplified, output_path=output_path)

    info = _format_info(transcript, simplified, glosses, n)
    return video_path, info


def process_text(text_input: str, use_lm_studio: bool):
    if not text_input.strip():
        return None, "⚠️ Please enter some text."

    simplified = simplifier.simplify(text_input) if use_lm_studio else \
                 simplifier._rule_based_simplify(text_input)

    output_path = "./output/asl_text_output.gif"
    video_path, glosses, n = generator.text_to_video(simplified, output_path=output_path)

    info = _format_info(text_input, simplified, glosses, n)
    return video_path, info


def _format_info(original: str, simplified: str,
                 glosses: list, n_frames: int) -> str:
    return (
        f"**Original:** {original}\n\n"
        f"**ASL gloss:** {simplified}\n\n"
        f"**Signs ({len(glosses)}):** {' → '.join(glosses)}\n\n"
        f"**Frames:** {n_frames}"
    )


# ── UI ─────────────────────────────────────────────────────

lm_status = "🟢 LM Studio connected" if lm_available else \
            "🟡 LM Studio not running — using rule-based ASL simplification"

with gr.Blocks(title="Audio → ASL Sign Language") as demo:

    gr.Markdown(f"""
    # 🤟 Audio → ASL Sign Language Avatar
    Convert spoken audio or text into **American Sign Language** avatar animation.
    Uses **Whisper** for transcription + **LM Studio** for ASL grammar.

    **{lm_status}**
    > ♿ *Assistive technology for the deaf and hard-of-hearing community.*
    """)

    use_lm = gr.Checkbox(
        label="🧠 Use LM Studio to simplify text for ASL grammar",
        value=lm_available,
        interactive=lm_available
    )

    with gr.Tabs():

        with gr.Tab("🎤 Audio → ASL"):
            with gr.Row():
                with gr.Column():
                    audio_input = gr.Audio(
                        label="Upload or Record Audio",
                        type="filepath",
                        sources=["microphone", "upload"]
                    )
                    audio_btn = gr.Button("🤟 Convert to ASL", variant="primary", size="lg")

                with gr.Column():
                    audio_video = gr.Image(label="🤟 ASL Avatar Output", type="filepath")
                    audio_info  = gr.Markdown()

            audio_btn.click(
                fn=process_audio,
                inputs=[audio_input, use_lm],
                outputs=[audio_video, audio_info]
            )

        with gr.Tab("⌨️ Text → ASL"):
            with gr.Row():
                with gr.Column():
                    text_input = gr.Textbox(
                        label="Enter Text",
                        placeholder="Type any sentence...",
                        lines=3
                    )
                    text_btn = gr.Button("🤟 Convert to ASL", variant="primary", size="lg")
                    gr.Examples(
                        examples=[
                            ["Hello nice to meet you"],
                            ["Please help me I am lost"],
                            ["Thank you very much"],
                            ["Where is the hospital"],
                            ["My name is John I am deaf"],
                            ["Yes no please sorry help"],
                            ["I need water please"],
                            ["Call the police"],
                            ["I feel sick need doctor"],
                        ],
                        inputs=[text_input],
                    )

                with gr.Column():
                    text_video = gr.Image(label="🤟 ASL Avatar Output", type="filepath")
                    text_info  = gr.Markdown()

            text_btn.click(
                fn=process_text,
                inputs=[text_input, use_lm],
                outputs=[text_video, text_info]
            )

demo.launch(server_port=7860, share=False)