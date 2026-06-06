from fastapi import FastAPI, Request, File, UploadFile, Form, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import shutil
from persona_manager import PersonaManager
from bot_pipeline import BotPipeline

app = FastAPI(title="MultiModal Offender Persona Twin")
pm = PersonaManager()

# Mount static files and templates
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("data/personas", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/data", StaticFiles(directory="data"), name="data")
templates = Jinja2Templates(directory="templates")

def pregenerate_voice_samples():
    import soundfile as sf
    from bot_pipeline import ensure_kokoro_files
    from kokoro_onnx import Kokoro
    
    sample_dir = "static/audio/samples"
    os.makedirs(sample_dir, exist_ok=True)
    
    voices = [
        "af_bella", "af_sarah", "af_nicole", "af_sky",
        "am_adam", "am_michael", "bf_emma", "bf_isabella",
        "bm_george", "bm_lewis"
    ]
    
    # Check if all samples exist
    missing_voices = [v for v in voices if not os.path.exists(os.path.join(sample_dir, f"{v}.wav"))]
    if not missing_voices:
        return
        
    print(f"Pregenerating sample audios for missing voices: {missing_voices}...")
    model_path, voices_path = ensure_kokoro_files()
    kokoro = Kokoro(model_path, voices_path)
    
    text = "Hello! This is a sample of my voice. I hope it matches the personality you want to create."
    for voice in missing_voices:
        try:
            samples, sample_rate = kokoro.create(text, voice=voice, speed=1.0)
            out_path = os.path.join(sample_dir, f"{voice}.wav")
            sf.write(out_path, samples, sample_rate)
            print(f"Generated sample: {out_path}")
        except Exception as e:
            print(f"Error generating sample for {voice}: {e}")

@app.on_event("startup")
def startup_event():
    pregenerate_voice_samples()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    personas = pm.get_all_personas()
    return templates.TemplateResponse(request, "index.html", {"personas": personas})

def generate_sadtalker_videos(persona_id: str):
    import subprocess
    persona = pm.get_persona(persona_id)
    persona_dir = os.path.dirname(persona["photo_path"])
    photo_path = os.path.abspath(persona["photo_path"])
    
    sadtalker_dir = os.path.abspath("SadTalker")
    idle_audio = os.path.abspath("data/idle.wav")
    talking_audio = os.path.abspath("data/talking.wav")
    
    python_exe = os.path.abspath("venv/bin/python3")
    
    # 1. Generate Idle Loop
    subprocess.run([
        python_exe, "inference.py",
        "--driven_audio", idle_audio,
        "--source_image", photo_path,
        "--result_dir", os.path.abspath(persona_dir),
        "--cpu"
    ], cwd=sadtalker_dir)
    
    # Rename output for Idle Loop
    idle_video_path = None
    for root, dirs, files in os.walk(persona_dir):
        for f in files:
            if f.endswith(".mp4") and "idle_loop" not in f and "talking_loop" not in f:
                idle_video_path = os.path.join(root, f)
                break
        if idle_video_path:
            break
            
    if idle_video_path:
        shutil.move(idle_video_path, os.path.join(persona_dir, "idle_loop.mp4"))
        # Clean up timestamped folders
        for item in os.listdir(persona_dir):
            item_path = os.path.join(persona_dir, item)
            if os.path.isdir(item_path) and item != "first_frame_dir":
                shutil.rmtree(item_path)

    # 2. Generate Talking Loop
    subprocess.run([
        python_exe, "inference.py",
        "--driven_audio", talking_audio,
        "--source_image", photo_path,
        "--result_dir", os.path.abspath(persona_dir),
        "--cpu"
    ], cwd=sadtalker_dir)
    
    # Rename output for Talking Loop
    talking_video_path = None
    for root, dirs, files in os.walk(persona_dir):
        for f in files:
            if f.endswith(".mp4") and "idle_loop" not in f and "talking_loop" not in f:
                talking_video_path = os.path.join(root, f)
                break
        if talking_video_path:
            break
            
    if talking_video_path:
        shutil.move(talking_video_path, os.path.join(persona_dir, "talking_loop.mp4"))
        # Clean up timestamped folders
        for item in os.listdir(persona_dir):
            item_path = os.path.join(persona_dir, item)
            if os.path.isdir(item_path) and item != "first_frame_dir":
                shutil.rmtree(item_path)
            
    pm.update_persona_status(persona_id, "ready")

@app.post("/create_persona")
async def create_persona(
    background_tasks: BackgroundTasks,
    name: str = Form(...),
    system_prompt: str = Form(...),
    photo: UploadFile = File(...),
    voice_type: str = Form("preset"),
    voice: str = Form("af_bella"),
    voice_file: UploadFile = File(None)
):
    # Save uploaded photo temporarily
    temp_photo_path = f"data/{photo.filename}"
    with open(temp_photo_path, "wb") as buffer:
        shutil.copyfileobj(photo.file, buffer)
        
    temp_voice_path = None
    voice_ref_text = None
    
    if voice_type == "cloned" and voice_file and voice_file.filename:
        temp_voice_path = f"data/{voice_file.filename}"
        with open(temp_voice_path, "wb") as buffer:
            shutil.copyfileobj(voice_file.file, buffer)
            
        # Transcribe reference audio using Whisper
        try:
            import whisper
            print(f"Transcribing voice reference audio: {temp_voice_path}...")
            stt_model = whisper.load_model("base")
            result = stt_model.transcribe(temp_voice_path)
            voice_ref_text = result.get("text", "").strip()
            print(f"Transcribed voice ref text: {voice_ref_text}")
        except Exception as e:
            print(f"Error transcribing reference voice: {e}")
            voice_ref_text = "Hello there."
            
    persona = pm.create_persona(
        name, system_prompt, temp_photo_path,
        voice_type=voice_type, voice=voice,
        voice_ref_path=temp_voice_path, voice_ref_text=voice_ref_text
    )
    
    # Clean up temp files
    if os.path.exists(temp_photo_path):
        os.remove(temp_photo_path)
    if temp_voice_path and os.path.exists(temp_voice_path):
        os.remove(temp_voice_path)
        
    background_tasks.add_task(generate_sadtalker_videos, persona["id"])
    
    return RedirectResponse(url="/", status_code=303)

@app.get("/chat/{persona_id}", response_class=HTMLResponse)
async def chat_interface(request: Request, persona_id: str):
    persona = pm.get_persona(persona_id)
    if not persona:
        return RedirectResponse(url="/")
    return templates.TemplateResponse(request, "chat.html", {"persona": persona})

@app.websocket("/ws/chat/{persona_id}")
async def websocket_endpoint(websocket: WebSocket, persona_id: str):
    await websocket.accept()
    persona = pm.get_persona(persona_id)
    if not persona:
        await websocket.close()
        return
        
    pipeline = BotPipeline(persona)
    await pipeline.run(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
