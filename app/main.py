from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import httpx
import asyncio
import logging
import base64

# ── Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ── API URLs — update these every time Colab restarts
STT_API_URL = "https://your-stt-ngrok-url/transcribe"
LLM_API_URL = "https://your-llm-ngrok-url/generate"
TTS_API_URL = "https://your-tts-ngrok-url/generate-speech"

# ── Ngrok headers (required for all requests)
HEADERS = {
    "ngrok-skip-browser-warning": "true",
    "User-Agent": "Mozilla/5.0"
}

# ── HTTP client
client = httpx.AsyncClient(
    timeout=httpx.Timeout(120.0),
    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
)

# ── Voice map per language
VOICE_MAP = {
    "en":       "Mary speaks in a warm, confident and professional tone with clear pronunciation at a moderate pace",
    "hi":       "Rani speaks in a warm, confident and professional Hindi tone with clear pronunciation",
    "hinglish": "Rani speaks in a warm, confident and friendly tone with natural Hinglish — mixed Hindi and English — pronunciation at a moderate pace",
    "auto":     "Mary speaks in a warm, professional and friendly tone with clear pronunciation",
}

# ── Token limits per language (latency control)
TOKEN_MAP = {
    "en":       6000,
    "hi":       6000,
    "hinglish": 6000,
    "auto":     6000,
}

def get_voice_description(language: str) -> str:
    return VOICE_MAP.get(language, VOICE_MAP["auto"])

def get_max_tokens(language: str) -> int:
    return TOKEN_MAP.get(language, 6000)


# ── FastAPI app
app = FastAPI(
    title="Indux Technologies — AI Voice Call Orchestrator",
    description="STT → LLM → TTS Pipeline",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_event():
    await client.aclose()


# ── Root
@app.get("/")
def root():
    return {
        "service": "Indux Technologies AI Voice Call Orchestrator",
        "version": "2.0.0",
        "pipeline": "STT → LLM → TTS",
        "endpoints": {
            "/process-call":      "Returns WAV audio",
            "/process-call-json": "Returns JSON + base64 audio",
            "/health":            "Service health check",
        }
    }


# ── Main pipeline — returns WAV audio
@app.post("/process-call")
async def process_call(
    file: UploadFile = File(...),
    language: str = "auto"
):
    """Audio → STT → LLM → TTS → Audio (WAV)"""
    try:
        audio_content = await file.read()
        if not audio_content:
            raise HTTPException(status_code=400, detail="Empty audio file")

        # ── Step 1: STT
        logger.info(f"[STT] Sending to {STT_API_URL}")
        stt_response = await client.post(
            STT_API_URL,
            files={"file": (file.filename, audio_content, file.content_type)},
            params={"language": language},
            headers=HEADERS,
            timeout=30,
        )
        if stt_response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"STT failed: {stt_response.text}")

        stt_data     = stt_response.json()
        user_text    = stt_data.get("text", stt_data.get("transcription", ""))
        detected_lang = stt_data.get("language", language)
        logger.info(f"[STT] '{user_text}' | lang={detected_lang}")

        # ── Step 2: LLM
        logger.info(f"[LLM] Sending to {LLM_API_URL}")
        llm_response = await client.post(
            LLM_API_URL,
            json={"text": user_text},
            headers={**HEADERS, "Content-Type": "application/json"},
            timeout=30,
        )
        if llm_response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"LLM failed: {llm_response.text}")

        llm_data       = llm_response.json()
        ai_reply       = llm_data.get("response", llm_data.get("text", ""))
        intent         = llm_data.get("intent", "unknown")
        lead_score     = llm_data.get("lead_score", 0)
        reply_language = llm_data.get("language", detected_lang)
        logger.info(f"[LLM] reply='{ai_reply}' | intent={intent} | score={lead_score} | lang={reply_language}")

        # ── Step 3: TTS
        voice_desc  = get_voice_description(reply_language)
        max_tokens  = get_max_tokens(reply_language)
        logger.info(f"[TTS] Sending to {TTS_API_URL} | tokens={max_tokens}")

        tts_response = await client.post(
            TTS_API_URL,
            json={
                "text":           ai_reply,
                "description":    voice_desc,
                "language":       reply_language,
                "max_new_tokens": max_tokens,
            },
            headers={**HEADERS, "Content-Type": "application/json"},
            timeout=120,
        )
        if tts_response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"TTS failed: {tts_response.text}")

        tts_latency = tts_response.headers.get("X-Latency", "?")
        logger.info(f"[TTS] Done | latency={tts_latency}s | size={len(tts_response.content)} bytes")
        logger.info("[PIPELINE] ✅ Complete")

        return Response(
            content=tts_response.content,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=ai_response.wav",
                "X-User-Text":         user_text,
                "X-AI-Reply":          ai_reply,
                "X-Intent":            intent,
                "X-Lead-Score":        str(lead_score),
                "X-Language":          reply_language,
                "X-TTS-Latency":       tts_latency,
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")


# ── JSON pipeline — returns base64 audio + full metadata
@app.post("/process-call-json")
async def process_call_json(
    file: UploadFile = File(...),
    language: str = "auto"
):
    """Audio → STT → LLM → TTS → JSON with base64 audio"""
    try:
        audio_content = await file.read()
        if not audio_content:
            raise HTTPException(status_code=400, detail="Empty audio file")

        # STT
        stt_response = await client.post(
            STT_API_URL,
            files={"file": (file.filename, audio_content, file.content_type)},
            params={"language": language},
            headers=HEADERS,
            timeout=30,
        )
        if stt_response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"STT failed: {stt_response.text}")

        stt_data      = stt_response.json()
        user_text     = stt_data.get("text", stt_data.get("transcription", ""))
        detected_lang = stt_data.get("language", language)

        # LLM
        llm_response = await client.post(
            LLM_API_URL,
            json={"text": user_text},
            headers={**HEADERS, "Content-Type": "application/json"},
            timeout=30,
        )
        if llm_response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"LLM failed: {llm_response.text}")

        llm_data       = llm_response.json()
        ai_reply       = llm_data.get("response", llm_data.get("text", ""))
        intent         = llm_data.get("intent", "unknown")
        lead_score     = llm_data.get("lead_score", 0)
        reply_language = llm_data.get("language", detected_lang)

        # TTS
        voice_desc  = get_voice_description(reply_language)
        max_tokens  = get_max_tokens(reply_language)

        tts_response = await client.post(
            TTS_API_URL,
            json={
                "text":           ai_reply,
                "description":    voice_desc,
                "language":       reply_language,
                "max_new_tokens": max_tokens,
            },
            headers={**HEADERS, "Content-Type": "application/json"},
            timeout=120,
        )
        if tts_response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"TTS failed: {tts_response.text}")

        audio_base64 = base64.b64encode(tts_response.content).decode()

        return {
            "user_text":        user_text,
            "detected_language": reply_language,
            "ai_reply":         ai_reply,
            "intent":           intent,
            "lead_score":       lead_score,
            "voice_description": voice_desc,
            "audio_base64":     audio_base64,
            "tts_latency":      tts_response.headers.get("X-Latency", "?"),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Health check
@app.get("/health")
def health_check():
    def check(url: str) -> str:
        try:
            r = httpx.get(url.replace("/transcribe", "/health")
                            .replace("/generate", "/health")
                            .replace("/generate-speech", "/health"),
                          headers=HEADERS, timeout=5)
            return "healthy" if r.status_code in [200, 405] else "unhealthy"
        except:
            return "unreachable"

    return {
        "status":     "online",
        "stt_status": check(STT_API_URL),
        "llm_status": check(LLM_API_URL),
        "tts_status": check(TTS_API_URL),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
