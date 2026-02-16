import gradio as gr
import requests
import os
import time

# Your orchestrator URL
ORCHESTRATOR_URL = "http://localhost:8004"

def full_pipeline(audio):
    """
    Process audio through the orchestrator pipeline
    """
    print("=" * 60)
    print("PIPELINE STARTED: STT → LLM → TTS")
    print("=" * 60)

    pipeline_start = time.time()

    if audio is None:
        return "❌ No audio provided", "", None, {}

    try:
        # STEP 0: Process Audio
        print("Processing audio...")

        # audio is a filepath when using type="filepath"
        audio_path = audio

        # Get audio duration for RTF calculation
        import soundfile as sf
        audio_array, sr = sf.read(audio_path)
        audio_duration = len(audio_array) / sr

        # STEP 1: Call Orchestrator
        print("STEP 1: Calling Orchestrator...")

        orchestrator_start = time.time()

        with open(audio_path, 'rb') as f:
            files = {'file': ('audio.wav', f, 'audio/wav')}
            params = {'language': 'auto'}  # Default to Hindi
            response = requests.post(
                f"{ORCHESTRATOR_URL}/process-call",
                files=files,
                params=params,
                timeout=90
            )

        orchestrator_latency = time.time() - orchestrator_start

        if response.status_code != 200:
            error_msg = response.json().get('detail', 'Unknown error')
            return f"❌ Error: {error_msg}", "", None, {"Status": "Failed"}

        # Save response audio
        output_path = "output_audio.wav"
        with open(output_path, 'wb') as f:
            f.write(response.content)

        print(f"✅ Orchestrator Latency: {orchestrator_latency:.3f}s")

        # FINAL METRICS
        total_latency = time.time() - pipeline_start
        rtf = total_latency / audio_duration

        metrics = {
            "Audio Duration (s)": round(audio_duration, 3),
            "Total Latency (s)": round(total_latency, 3),
            "RTF": round(rtf, 3),
            "Status": "Success"
        }

        print("=" * 60)
        print("PERFORMANCE METRICS:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        print("=" * 60)
        print("PIPELINE COMPLETED!")
        print("=" * 60)

        return (
            "✅ Processing successful!",
            "AI response generated",
            output_path,
            metrics
        )

    except Exception as e:
        error_msg = f"❌ Exception: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg, "", None, {"Status": "Error"}


# Gradio Interface
with gr.Blocks(title="AI Voice Call Orchestrator") as demo:
    gr.Markdown("# 🎙️ AI Voice Call Orchestrator")
    gr.Markdown("### STT → LLM → TTS Pipeline")

    with gr.Row():
        with gr.Column():
            # Audio Input
            audio_in = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="🎤 Speak or Upload Audio"
            )

            # Process Button
            btn = gr.Button("🚀 Run Full Pipeline", variant="primary", size="lg")

        with gr.Column():
            # Output Components
            stt_out = gr.Textbox(label="📝 STT Output", lines=2)
            llm_out = gr.Textbox(label="🤖 LLM Response", lines=4)
            tts_out = gr.Audio(label="🔊 TTS Output", type="filepath", interactive=False)
            metrics_out = gr.JSON(label="📊 Latency & RTF Metrics")

    # Button click binding
    btn.click(
        fn=full_pipeline,
        inputs=[audio_in],
        outputs=[stt_out, llm_out, tts_out, metrics_out]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )