from pydantic import BaseModel

class PipelineResponse(BaseModel):
    user_text: str
    detected_language: str
    ai_reply: str
    voice_description: str
    
class HealthResponse(BaseModel):
    status: str
    stt_status: str
    llm_status: str
    tts_status: str
