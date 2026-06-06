import os
import json
import uuid
import shutil
from typing import List, Dict

DATA_DIR = "data/personas"

class PersonaManager:
    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        self.personas_file = os.path.join(DATA_DIR, "personas.json")
        self._ensure_file()

    def _ensure_file(self):
        if not os.path.exists(self.personas_file):
            with open(self.personas_file, "w") as f:
                json.dump([], f)

    def _load_personas(self) -> List[Dict]:
        with open(self.personas_file, "r") as f:
            return json.load(f)

    def _save_personas(self, personas: List[Dict]):
        with open(self.personas_file, "w") as f:
            json.dump(personas, f, indent=4)

    def get_all_personas(self) -> List[Dict]:
        return self._load_personas()

    def get_persona(self, persona_id: str) -> Dict:
        personas = self._load_personas()
        for p in personas:
            if p["id"] == persona_id:
                return p
        return None

    def create_persona(self, name: str, system_prompt: str, photo_path: str, voice_type: str = "preset", voice: str = "af_bella", voice_ref_path: str = None, voice_ref_text: str = None) -> Dict:
        persona_id = str(uuid.uuid4())
        
        # Create persona directory
        persona_dir = os.path.join(DATA_DIR, persona_id)
        os.makedirs(persona_dir, exist_ok=True)
        
        # Move photo to persona dir
        _, ext = os.path.splitext(photo_path)
        new_photo_path = os.path.join(persona_dir, f"photo{ext}")
        shutil.copy(photo_path, new_photo_path)
        
        # Move voice reference if provided
        final_voice_ref_path = None
        if voice_ref_path and os.path.exists(voice_ref_path):
            final_voice_ref_path = os.path.join(persona_dir, "voice_ref.wav")
            shutil.copy(voice_ref_path, final_voice_ref_path)
        
        persona = {
            "id": persona_id,
            "name": name,
            "system_prompt": system_prompt,
            "photo_path": new_photo_path,
            "voice_type": voice_type,
            "voice": voice,
            "voice_ref_path": final_voice_ref_path,
            "voice_ref_text": voice_ref_text,
            "status": "processing" # Will change to 'ready' after SadTalker finishes
        }
        
        personas = self._load_personas()
        personas.append(persona)
        self._save_personas(personas)
        
        return persona

    def update_persona_status(self, persona_id: str, status: str):
        personas = self._load_personas()
        for p in personas:
            if p["id"] == persona_id:
                p["status"] = status
                break
        self._save_personas(personas)
