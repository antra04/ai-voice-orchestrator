import gradio as gr
import requests
import base64
import soundfile as sf
import numpy as np
import os
import time

# ── Orchestrator URL — update if using ngrok
ORCHESTRATOR_URL = "http://localhost:8004"

HEADERS = {
    "ngrok-skip-browser-warning": "true",
    "User-Agent": "Mozilla/5.0"
}

# ── Intent badge color map
INTENT_COLORS = {
    "greeting":      "🟢",
    "sales":         "🔥",
    "booking":       "📅",
    "order_track":   "📦",
    "refund_return": "↩️",
    "login_crash":   "🔧",
    "farewell":      "👋",
    "unknown":       "❓",
}

LANGUAGE_LABELS = {
    "en":       "🇬🇧 English",
    "hi":       "🇮🇳 Hindi",
    "hinglish": "🔀 Hinglish",
    "auto":     "🌐 Auto",
}


def check_health():
    """Check all services health"""
    try:
        r = requests.get(
            f"{ORCHESTRATOR_URL}/health",
            headers=HEADERS,
            timeout=5
        )
        if r.status_code == 200:
            data = r.json()
            stt = data.get("stt_status", "unknown")
            llm = data.get("llm_status", "unknown")
            tts = data.get("tts_status", "unknown")

            def badge(s):
                return "✅" if s == "healthy" else ("⚠️" if s == "unreachable" else "❌")

            return (
                f"**Pipeline Status**\n\n"
                f"| Service | Status |\n"
                f"|---------|--------|\n"
                f"| STT     | {badge(stt)} {stt} |\n"
                f"| LLM     | {badge(llm)} {llm} |\n"
                f"| TTS     | {badge(tts)} {tts} |\n"
                f"| Orchestrator | ✅ online |"
            )
        return "❌ Orchestrator unreachable"
    except Exception as e:
        return f"❌ Health check failed: {str(e)}"


def run_pipeline(audio_path, language):
    """Full STT → LLM → TTS pipeline"""
    if audio_path is None:
        return (
            "⚠️ No audio provided",
            "", "", "", "", "", None
        )

    try:
        start = time.time()

        with open(audio_path, "rb") as f:
            files = {"file": ("audio.wav", f, "audio/wav")}
            response = requests.post(
                f"{ORCHESTRATOR_URL}/process-call-json",
                files=files,
                params={"language": language},
                headers=HEADERS,
                timeout=120,
            )

        total_time = round(time.time() - start, 2)

        if response.status_code != 200:
            return (
                f"❌ Pipeline failed ({response.status_code})",
                "", "", "", "", "", None
            )

        data = response.json()

        user_text        = data.get("user_text", "")
        ai_reply         = data.get("ai_reply", "")
        intent           = data.get("intent", "unknown")
        lead_score       = data.get("lead_score", 0)
        detected_lang    = data.get("detected_language", language)
        voice_desc       = data.get("voice_description", "")
        tts_latency      = data.get("tts_latency", "?")
        audio_b64        = data.get("audio_base64", "")

        # ── Decode audio
        audio_out = None
        if audio_b64:
            audio_bytes = base64.b64decode(audio_b64)
            out_path = "output_audio.wav"
            with open(out_path, "wb") as f:
                f.write(audio_bytes)
            audio_out = out_path

        # ── Lead score bar
        score_bar = "🟩" * lead_score + "⬜" * (10 - lead_score)

        # ── Intent badge
        intent_icon = INTENT_COLORS.get(intent, "❓")
        lang_label  = LANGUAGE_LABELS.get(detected_lang, detected_lang)

        # ── Metrics summary
        metrics = (
            f"**⏱ Total Latency:** {total_time}s\n\n"
            f"**🔊 TTS Latency:** {tts_latency}s\n\n"
            f"**🌐 Language:** {lang_label}\n\n"
            f"**🎯 Intent:** {intent_icon} `{intent}`\n\n"
            f"**📊 Lead Score:** {score_bar} `{lead_score}/10`\n\n"
            f"**🎙 Voice:** {voice_desc[:60]}..."
        )

        return (
            f"🎤 **You said:**\n{user_text}",
            f"🤖 **AI Reply:**\n{ai_reply}",
            metrics,
            lang_label,
            f"{intent_icon} {intent}",
            f"{score_bar} {lead_score}/10",
            audio_out,
        )

    except Exception as e:
        return (
            f"❌ Error: {str(e)}",
            "", "", "", "", "", None
        )


# ── Gradio UI
with gr.Blocks(
    title="Indux Technologies — AI Voice Pipeline",
    theme=gr.themes.Soft(),
    css="""
        .header { text-align: center; padding: 20px; }
        .metric-box { background: #f8f9fa; border-radius: 8px; padding: 10px; }
    """
) as demo:

    # Header
    gr.HTML("""
        <div class='header'>
            <h1>🤖 Indux Technologies</h1>
            <h3>AI Voice Call Pipeline — STT → LLM → TTS</h3>
            <p style='color: gray;'>Hindi • English • Hinglish</p>
        </div>
    """)

    with gr.Row():
        # ── Left column: Input
        with gr.Column(scale=1):
            gr.Markdown("## 🎙️ Input")

            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Speak or Upload Audio",
            )

            language_select = gr.Dropdown(
                choices=[
                    ("🌐 Auto Detect", "auto"),
                    ("🇮🇳 Hindi", "hi"),
                    ("🇬🇧 English", "en"),
                    ("🔀 Hinglish", "hinglish"),
                ],
                value="auto",
                label="Language",
            )

            with gr.Row():
                run_btn    = gr.Button("▶ Run Pipeline", variant="primary", scale=2)
                health_btn = gr.Button("💊 Health", variant="secondary", scale=1)

            health_output = gr.Markdown(label="Service Health")

        # ── Right column: Output
        with gr.Column(scale=1):
            gr.Markdown("## 📤 Output")

            user_text_out = gr.Markdown(label="Transcription")
            ai_reply_out  = gr.Markdown(label="AI Reply")
            audio_output  = gr.Audio(
                type="filepath",
                label="🔊 AI Voice Response",
                autoplay=True,
            )

    # ── Metrics row
    gr.Markdown("## 📊 Pipeline Metrics")
    with gr.Row():
        metrics_out   = gr.Markdown(label="Metrics")
        lang_out      = gr.Textbox(label="🌐 Language", interactive=False)
        intent_out    = gr.Textbox(label="🎯 Intent",   interactive=False)
        score_out     = gr.Textbox(label="📊 Lead Score", interactive=False)

    # ── Test sentences
    gr.Markdown("## 🧪 Quick Test Sentences")
    with gr.Row():
        gr.Examples(
            examples=[
                ["Hello, I'm interested in your services"],
                ["Mujhe aapki services ke baare mein jaanna hai"],
                ["Haan, main interested hoon, price kya hai?"],
                ["Mera order kahan hai?"],
                ["Thank you, goodbye"],
            ],
            inputs=[],
            label="Click to copy a test sentence",
        )

    # ── Event handlers
    run_btn.click(
        fn=run_pipeline,
        inputs=[audio_input, language_select],
        outputs=[
            user_text_out,
            ai_reply_out,
            metrics_out,
            lang_out,
            intent_out,
            score_out,
            audio_output,
        ],
    )

    health_btn.click(
        fn=check_health,
        inputs=[],
        outputs=[health_output],
    )

    # Auto health check on load
    demo.load(fn=check_health, outputs=[health_output])


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,           # generates public link
        show_error=True,
    )
