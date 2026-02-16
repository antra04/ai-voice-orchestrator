from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import requests
from .config import get_settings
from .schemas import PipelineResponse, HealthResponse

app = FastAPI(
    title="AI Voice Call Orchestrator",
    description="Orchestrates STT → LLM → TTS pipeline",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

settings = get_settings()

@app.get("/")
def root():
    return {
        "service": "AI Voice Call Orchestrator",
        "version": "1.0.0",
        "endpoints": {
            "pipeline": "/process-call",
            "health": "/health"
        }
    }

@app.post("/process-call")
async def process_call(
    file: UploadFile = File(...),
    language: str = "hi"
):
    """
    Complete voice call pipeline:
    Audio → STT → Text → LLM → Reply → TTS → Audio
    """
    try:
        # Step 1: STT - Audio to Text
        audio_content = await file.read()
        
        # Add ngrok headers
        headers = {
            'ngrok-skip-browser-warning': 'true',
            'User-Agent': 'Mozilla/5.0'
        }
        
        files = {"file": (file.filename, audio_content, file.content_type)}
        params = {"language": language}
        
        print(f"[STT] Sending request to {settings.stt_api_url}")
        print(f"[STT] File: {file.filename}, Language: {language}")
        
        stt_response = requests.post(
            settings.stt_api_url,
            files=files,
            params=params,
            headers=headers,
            timeout=30
        )
        
        print(f"[STT] Status: {stt_response.status_code}")
        print(f"[STT] Response: {stt_response.text}")
        
        if stt_response.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"STT failed with status {stt_response.status_code}: {stt_response.text}"
            )
        
        stt_data = stt_response.json()
        user_text = stt_data.get('text', stt_data.get('transcription', ''))
        detected_language = stt_data.get('language', language)
        
        print(f"[STT] Transcription: {user_text}")
        
        # Step 2: LLM - Generate Reply
        print(f"[LLM] Sending request to {settings.llm_api_url}")
        print(f"[LLM] Input text: {user_text}")
        
        llm_response = requests.post(
            settings.llm_api_url,
            json={"text": user_text},
            headers=headers,
            timeout=30
        )
        
        print(f"[LLM] Status: {llm_response.status_code}")
        print(f"[LLM] Response: {llm_response.text}")
        
        if llm_response.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"LLM failed with status {llm_response.status_code}: {llm_response.text}"
            )
        
        llm_data = llm_response.json()
        ai_reply = llm_data.get('text', llm_data.get('response', ''))
        voice_description = llm_data.get('description', 'A calm, professional voice')
        
        print(f"[LLM] AI Reply: {ai_reply}")
        print(f"[LLM] Voice Description: {voice_description}")
        
        # Step 3: TTS - Text to Audio
        print(f"[TTS] Sending request to {settings.tts_api_url}")
        
        tts_response = requests.post(
            settings.tts_api_url,
            json={
                "text": ai_reply,
                "description": voice_description
            },
            headers=headers,
            timeout=60
        )
        
        print(f"[TTS] Status: {tts_response.status_code}")
        
        if tts_response.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"TTS failed with status {tts_response.status_code}: {tts_response.text}"
            )
        
        print("[PIPELINE] Success! Returning audio response")
        
        # Return audio directly
        return Response(
            content=tts_response.content,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=ai_response.wav"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Pipeline failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline failed: {str(e)}"
        )

@app.post("/process-call-json")
async def process_call_json(
    file: UploadFile = File(...),
    language: str = "auto"
):
    """
    Same as /process-call but returns JSON with base64 audio
    """
    try:
        # Step 1: STT
        audio_content = await file.read()
        
        # Add ngrok headers
        headers = {
            'ngrok-skip-browser-warning': 'true',
            'User-Agent': 'Mozilla/5.0'
        }
        
        files = {"file": (file.filename, audio_content, file.content_type)}
        params = {"language": language}
        
        print(f"[STT] Sending request to {settings.stt_api_url}")
        print(f"[STT] File: {file.filename}, Language: {language}")
        
        stt_response = requests.post(
            settings.stt_api_url,
            files=files,
            params=params,
            headers=headers,
            timeout=30
        )
        
        print(f"[STT] Status: {stt_response.status_code}")
        print(f"[STT] Response: {stt_response.text}")
        
        if stt_response.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"STT failed with status {stt_response.status_code}: {stt_response.text}"
            )
        
        stt_data = stt_response.json()
        user_text = stt_data.get('text', stt_data.get('transcription', ''))
        
        print(f"[STT] Transcription: {user_text}")
        
        # Step 2: LLM
        print(f"[LLM] Sending request to {settings.llm_api_url}")
        
        llm_response = requests.post(
            settings.llm_api_url,
            json={"text": user_text},
            headers=headers,
            timeout=30
        )
        
        print(f"[LLM] Status: {llm_response.status_code}")
        print(f"[LLM] Response: {llm_response.text}")
        
        if llm_response.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"LLM failed with status {llm_response.status_code}: {llm_response.text}"
            )
        
        llm_data = llm_response.json()
        ai_reply = llm_data.get('text', llm_data.get('response', ''))
        voice_description = llm_data.get('description', 'A calm, professional voice')
        
        print(f"[LLM] AI Reply: {ai_reply}")
        
        # Step 3: TTS
        print(f"[TTS] Sending request to {settings.tts_api_url}")
        
        tts_response = requests.post(
            settings.tts_api_url,
            json={
                "text": ai_reply,
                "description": voice_description
            },
            headers=headers,
            timeout=60
        )
        
        print(f"[TTS] Status: {tts_response.status_code}")
        
        if tts_response.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"TTS failed with status {tts_response.status_code}: {tts_response.text}"
            )
        
        # Return JSON with audio as base64
        import base64
        audio_base64 = base64.b64encode(tts_response.content).decode()
        
        print("[PIPELINE] Success! Returning JSON response")
        
        return {
            "user_text": user_text,
            "detected_language": stt_data.get('language', language),
            "ai_reply": ai_reply,
            "voice_description": voice_description,
            "audio_base64": audio_base64
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Pipeline failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check health of all services"""
    
    def check_service(url: str) -> str:
        try:
            # For ngrok, we need to add headers
            headers = {
                'ngrok-skip-browser-warning': 'true',
                'User-Agent': 'Mozilla/5.0'
            }
            
            # Try the actual endpoint instead of /health
            response = requests.get(url, headers=headers, timeout=5)
            return "healthy" if response.status_code in [200, 405] else "unhealthy"
        except Exception as e:
            print(f"Service check failed for {url}: {e}")
            return "unreachable"
    
    return {
        "status": "online",
        "stt_status": check_service(settings.stt_api_url),
        "llm_status": check_service(settings.llm_api_url),
        "tts_status": check_service(settings.tts_api_url)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )
