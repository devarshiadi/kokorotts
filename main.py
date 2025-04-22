from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import soundfile as sf
from kokoro import KPipeline
import os
import uuid

# Create FastAPI app
app = FastAPI(
    title="Kokoro TTS API",
    description="TTS service that generates streamable WAV audio",
    version="1.0.0"
)

# Allow CORS if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory to store generated audio files
AUDIO_DIR = "audio_output"
os.makedirs(AUDIO_DIR, exist_ok=True)

# Initialize the pipeline once at startup
pipeline = KPipeline(lang_code='a')  # American English

@app.post("/synthesize")
async def synthesize(
    request: Request,
    text: str = Form(...),
    voice: str = Form('af_sky'),
    speed: float = Form(1.0)
):
    """
    Generate TTS from input text and return a URL to the streamable audio.
    """
    # Generate audio segments without splitting
    segments = list(
        pipeline(
            text,
            voice=voice,
            speed=speed,
            split_pattern=r'$^'  # never matches
        )
    )
    # Concatenate numpy arrays
    audio_full = np.concatenate([audio for (_, _, audio) in segments])

    # Save to unique WAV file
    file_id = f"{uuid.uuid4()}.wav"
    file_path = os.path.join(AUDIO_DIR, file_id)
    sf.write(file_path, audio_full, 24000)

    # Construct full URL for streaming
    base_url = str(request.base_url).rstrip("/")
    stream_url = f"{base_url}/audio/{file_id}"
    return JSONResponse({"stream_url": stream_url})

@app.get("/audio/{file_id}")
def stream_audio(file_id: str):
    """
    Stream the generated WAV file as audio/wav.
    """
    file_path = os.path.join(AUDIO_DIR, file_id)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")

    def iterfile():
        with open(file_path, mode="rb") as f:
            yield from f

    return StreamingResponse(iterfile(), media_type="audio/wav")
