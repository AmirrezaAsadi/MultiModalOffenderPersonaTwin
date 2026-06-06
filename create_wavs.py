import wave
import struct
import math
import os

os.makedirs("data", exist_ok=True)

def create_wav(filename, duration, frequency=0):
    sample_rate = 16000
    n_samples = int(duration * sample_rate)
    
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        
        for i in range(n_samples):
            # If frequency > 0, generate a sine wave, else generate silence
            if frequency > 0:
                value = int(32767.0 * math.sin(frequency * math.pi * float(i) / float(sample_rate)))
            else:
                value = 0
            
            data = struct.pack('<h', value)
            wav_file.writeframesraw(data)

# Create 2 seconds of silence for the idle loop
create_wav("data/idle.wav", 2.0, 0)

# Create 5 seconds of a placeholder tone (or random noise) for the talking loop
# In production, use a recording of a person talking continuously for 5 seconds.
create_wav("data/talking.wav", 5.0, 440)

print("Created data/idle.wav and data/talking.wav")
