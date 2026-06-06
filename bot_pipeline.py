import os
import re
import asyncio
import json
import httpx
import whisper
import numpy as np
from kokoro_onnx import Kokoro
from loguru import logger
from fastapi import WebSocket, WebSocketDisconnect

# Cache dir for Kokoro
KOKORO_CACHE_DIR = os.path.expanduser("~/.cache/kokoro-onnx")
KOKORO_MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
KOKORO_VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.json"

def download_file(url: str, dest: str):
    import requests
    logger.info(f"Downloading {url} to {dest}...")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp_dest = dest + ".tmp"
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    with open(tmp_dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    os.replace(tmp_dest, dest)
    logger.info(f"Downloaded {dest}")

def ensure_kokoro_files():
    model_path = os.path.join(KOKORO_CACHE_DIR, "kokoro-v1.0.onnx")
    voices_path = os.path.join(KOKORO_CACHE_DIR, "voices.json")
    if not os.path.exists(model_path):
        download_file(KOKORO_MODEL_URL, model_path)
    if not os.path.exists(voices_path):
        download_file(KOKORO_VOICES_URL, voices_path)
    return model_path, voices_path

class BotPipeline:
    def __init__(self, persona: dict):
        self.persona = persona
        self.system_prompt = persona.get("system_prompt", "You are a helpful assistant.")
        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Models will be lazily loaded to avoid slow startup times
        self.stt_model = None
        self.kokoro = None
        self.qwen_model = None
        
        # Active speaker generation task tracking (to support interruption)
        self.speech_task = None
        self._interrupt_event = asyncio.Event()

    def load_models(self):
        # Initialize Whisper (STT)
        if not self.stt_model:
            logger.info("Loading Whisper base model...")
            self.stt_model = whisper.load_model("base")
            logger.info("Whisper base model loaded.")
            
        # Initialize TTS engine based on persona configuration
        voice_type = self.persona.get("voice_type", "preset")
        if voice_type == "preset":
            if not self.kokoro:
                logger.info("Ensuring Kokoro ONNX model files...")
                model_path, voices_path = ensure_kokoro_files()
                logger.info("Initializing Kokoro ONNX engine...")
                self.kokoro = Kokoro(model_path, voices_path)
                logger.info("Kokoro ONNX engine initialized.")
        else:
            if not self.qwen_model:
                logger.info("Loading Qwen3-TTS 0.6B Base model...")
                import torch
                from qwen_tts import Qwen3TTSModel
                device = "mps" if torch.backends.mps.is_available() else "cpu"
                dtype = torch.float32
                self.qwen_model = Qwen3TTSModel.from_pretrained(
                    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                    device_map=device,
                    dtype=dtype
                )
                logger.info("Qwen3-TTS 0.6B Base model loaded.")

    async def run(self, websocket: WebSocket):
        # Ensure models are loaded
        # Since loading models does some blocking IO, we run it in an executor
        await asyncio.to_thread(self.load_models)
        
        logger.info(f"Chat pipeline started for persona: {self.persona['name']}")
        
        # Notify the client that the bot is ready
        await websocket.send_json({"type": "bot_ready"})
        
        try:
            while True:
                # Receive message
                message = await websocket.receive()
                
                # Check message type
                if "bytes" in message:
                    # It's a WAV audio recording!
                    audio_data = message["bytes"]
                    
                    # If bot is currently speaking, interrupt it!
                    if self.speech_task and not self.speech_task.done():
                        logger.info("Interrupting bot speech due to new user input...")
                        self._interrupt_event.set()
                        # Wait for the task to finish canceling/wrapping up
                        try:
                            await self.speech_task
                        except asyncio.CancelledError:
                            pass
                    
                    self._interrupt_event.clear()
                    
                    # Process user audio in a background task so we can still listen for WebSocket interruptions
                    self.speech_task = asyncio.create_task(self.handle_user_audio(audio_data, websocket))
                
                elif "text" in message:
                    data = json.loads(message["text"])
                    if data.get("type") == "interrupt":
                        logger.info("User requested interrupt...")
                        self._interrupt_event.set()
                        if self.speech_task and not self.speech_task.done():
                            self.speech_task.cancel()
                            
                    elif data.get("type") == "playback_finished":
                        # Client finished playing bot response, reset UI
                        await websocket.send_json({"type": "bot_stopped"})
                        
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
        finally:
            if self.speech_task and not self.speech_task.done():
                self.speech_task.cancel()

    async def handle_user_audio(self, audio_data: bytes, websocket: WebSocket):
        try:
            # 1. Save binary WAV data to a temporary file
            temp_wav_path = f"data/temp_{self.persona['id']}.wav"
            with open(temp_wav_path, "wb") as f:
                f.write(audio_data)
                
            # 2. Transcribe using Whisper
            logger.info("Transcribing audio...")
            # Run Whisper transcription in thread pool to avoid blocking async loop
            result = await asyncio.to_thread(self.stt_model.transcribe, temp_wav_path)
            user_text = result.get("text", "").strip()
            logger.info(f"User Transcribed: {user_text}")
            
            # Clean up temp file
            if os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)
                
            if not user_text:
                logger.info("No text transcribed.")
                return
                
            # Echo transcription back to client
            await websocket.send_json({"type": "user_transcription", "text": user_text})
            
            # 3. Add to chat history
            self.messages.append({"role": "user", "content": user_text})
            
            # 4. Stream response from Ollama
            logger.info("Calling Ollama...")
            async with httpx.AsyncClient(timeout=60.0) as client:
                # Get available models in Ollama
                try:
                    tags_resp = await client.get("http://localhost:11434/api/tags")
                    available_models = [m["name"] for m in tags_resp.json().get("models", [])]
                except Exception:
                    available_models = []
                
                model_name = "gemma"
                if "gemma:latest" in available_models or "gemma" in available_models:
                    model_name = "gemma"
                elif "ministral-3:3b" in available_models:
                    model_name = "ministral-3:3b"
                elif "deepseek-r1:8b" in available_models:
                    model_name = "deepseek-r1:8b"
                elif len(available_models) > 0:
                    model_name = available_models[0]
                
                logger.info(f"Using model: {model_name}")
                
                # Call Ollama stream
                payload = {
                    "model": model_name,
                    "messages": self.messages,
                    "stream": True
                }
                
                sentence_buffer = ""
                bot_full_response = ""
                
                # Connect to stream
                async with client.stream("POST", "http://localhost:11434/api/chat", json=payload) as response:
                    async for line in response.aiter_lines():
                        if self._interrupt_event.is_set():
                            logger.info("Generation interrupted!")
                            break
                            
                        if not line:
                            continue
                            
                        chunk = json.loads(line)
                        content = chunk.get("message", {}).get("content", "")
                        bot_full_response += content
                        sentence_buffer += content
                        
                        # Check for sentence punctuation
                        if any(p in sentence_buffer for p in ['. ', '! ', '? ', '.\n', '!\n', '?\n']):
                            # Split into sentences
                            parts = re.split(r'(?<=[.!?])\s+', sentence_buffer)
                            for part in parts[:-1]:
                                if part.strip() and not self._interrupt_event.is_set():
                                    await self.speak_sentence(part.strip(), websocket)
                            sentence_buffer = parts[-1]
                            
                    # Yield final sentence if any left
                    if sentence_buffer.strip() and not self._interrupt_event.is_set():
                        await self.speak_sentence(sentence_buffer.strip(), websocket)
            
            # Save assistant response to chat history
            if bot_full_response:
                self.messages.append({"role": "assistant", "content": bot_full_response})
                
        except asyncio.CancelledError:
            logger.info("Speech task cancelled")
        except Exception as e:
            logger.error(f"Error in handle_user_audio: {e}")

    async def speak_sentence(self, sentence: str, websocket: WebSocket):
        # Clean sentence of think tags or special markings (e.g. from DeepSeek R1)
        sentence = re.sub(r'<think>.*?</think>', '', sentence, flags=re.DOTALL).strip()
        sentence = re.sub(r'<think>.*', '', sentence, flags=re.DOTALL).strip()
        sentence = re.sub(r'.*?</think>', '', sentence, flags=re.DOTALL).strip()
        if not sentence:
            return
            
        logger.info(f"Speaking: {sentence}")
        
        # Send text back to client so they can display what the bot is currently saying
        await websocket.send_json({"type": "bot_text", "text": sentence})
        
        # Generate audio using Kokoro or Qwen3-TTS
        try:
            voice_type = self.persona.get("voice_type", "preset")
            # Notify frontend bot is speaking (to swap to talking loop video!)
            await websocket.send_json({"type": "bot_speaking"})
            
            if voice_type == "preset":
                # Determine voice style from persona metadata, or fallback to guessing
                voice = self.persona.get("voice")
                if not voice:
                    voice = "af_bella"
                    if any(name in self.persona["name"].lower() for name in ["john", "michael", "david", "james", "robert", "william", "joseph"]):
                        voice = "am_adam"
                    
                stream = self.kokoro.create_stream(sentence, voice=voice, lang="en-us", speed=1.0)
                async for samples, sample_rate in stream:
                    if self._interrupt_event.is_set():
                        break
                    # Convert float32 samples to int16 PCM
                    audio_int16 = (samples * 32767).astype(np.int16).tobytes()
                    # Send raw bytes to WebSocket
                    await websocket.send_bytes(audio_int16)
                    # Yield control
                    await asyncio.sleep(0.01)
            else:
                # Cloned voice using Qwen3-TTS
                ref_audio = self.persona.get("voice_ref_path")
                ref_text = self.persona.get("voice_ref_text", "Hello.")
                if ref_audio and os.path.exists(ref_audio):
                    # Qwen3-TTS requires PyTorch/transformers, we run it in thread pool to avoid blocking async loop
                    wavs, sr = await asyncio.to_thread(
                        self.qwen_model.generate_voice_clone,
                        text=sentence,
                        language="English",
                        ref_audio=ref_audio,
                        ref_text=ref_text
                    )
                    import librosa
                    audio_samples = wavs[0]
                    if sr != 24000:
                        audio_samples = librosa.resample(audio_samples, orig_sr=sr, target_sr=24000)
                    
                    # Convert float32 samples to int16 PCM
                    audio_int16 = (audio_samples * 32767).astype(np.int16).tobytes()
                    
                    # Send raw bytes in chunks of 8192 bytes
                    chunk_size = 8192
                    for i in range(0, len(audio_int16), chunk_size):
                        if self._interrupt_event.is_set():
                            break
                        await websocket.send_bytes(audio_int16[i:i+chunk_size])
                        await asyncio.sleep(0.01)
                else:
                    logger.error(f"Reference voice audio file not found at: {ref_audio}")
                    
        except Exception as e:
            logger.error(f"Error in speak_sentence: {e}")
