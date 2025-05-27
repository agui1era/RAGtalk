import sounddevice as sd
import scipy.io.wavfile as wav
import tempfile
import whisper
import os
import requests
import subprocess
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()
DIFY_API_KEY = os.getenv("DIFY_API_KEY")
DIFY_API_URL = os.getenv("DIFY_API_URL", "https://api.dify.ai/v1/chat-messages")
GREETING = os.getenv("GREETING", "Hola.")  # 👈 saludo configurable

# === AUDIO INPUT ===
def grabar_audio(duracion=5, fs=44100):
    print(f"[🎙️] Grabando {duracion} segundos...")
    audio = sd.rec(int(duracion * fs), samplerate=fs, channels=1)
    sd.wait()
    archivo_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav.write(archivo_wav.name, fs, audio)
    print("[✅] Audio grabado.")
    return archivo_wav.name

# === ASR LOCAL ===
def transcribir_audio_local(ruta_audio):
    print("[🧠] Transcribiendo audio con Whisper local...")
    try:
        model = whisper.load_model("base")
    except AttributeError:
        raise RuntimeError("Whisper no se instaló correctamente. Usa: pip install git+https://github.com/openai/whisper.git")
    result = model.transcribe(ruta_audio)
    return result["text"]

# === CONSULTA DIFY ===
def consultar_dify(texto_usuario):
    payload = {
        "inputs": {},
        "query": texto_usuario,
        "user": "oscar_test"
    }
    headers = {
        "Authorization": f"Bearer {DIFY_API_KEY}",
        "Content-Type": "application/json"
    }
    print("[🤖] Consultando modelo Dify...")
    res = requests.post(DIFY_API_URL, json=payload, headers=headers)
    if res.status_code != 200:
        print(f"[❌] Error {res.status_code}: {res.text}")
        return "Error consultando el modelo."
    data = res.json()
    return data.get("answer", data.get("message", "Sin respuesta."))

# === TTS ===
def leer_respuesta(texto):
    print(f"[📢] Respuesta: {texto}")
    subprocess.call(['say', '-v', 'Paulina', texto])

# === MAIN ===
if __name__ == "__main__":
    subprocess.call(['say', '-v', 'Paulina', GREETING])  # 👈 saludo desde .env

    ruta = grabar_audio()
    texto = transcribir_audio_local(ruta)
    print(f"[🗣️] Tú dijiste: {texto}")
    respuesta = consultar_dify(texto)
    leer_respuesta(respuesta)