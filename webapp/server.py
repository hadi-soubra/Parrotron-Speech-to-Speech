"""
Local web server: speak into the browser, get a transcript and a TTS voice back.

    .venv/bin/python webapp/server.py      ->  http://localhost:8000

Environment variables
    WHISPER_MODEL         faster-whisper model name (default small.en)
    WHISPER_DEVICE        cuda / cpu / auto (default auto)
    PIPER_VOICE           Piper voice name (default en_US-lessac-medium)
    PARROTRON_CHECKPOINT  trained Parrotron .pth, enables the direct speech-to-speech mode
"""
import io
import os
import sys
import wave
import subprocess
import threading
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / 'static'
VOICES_DIR = ROOT / 'voices'
sys.path.insert(0, str(ROOT))

WHISPER_MODEL = os.environ.get('WHISPER_MODEL', 'small.en')
WHISPER_DEVICE = os.environ.get('WHISPER_DEVICE', 'auto')
PIPER_VOICE = os.environ.get('PIPER_VOICE', 'en_US-lessac-medium')
SAMPLE_RATE = 16000
MAX_TEXT_LENGTH = 2000


def find_parrotron_checkpoint():
    candidates = [os.environ.get('PARROTRON_CHECKPOINT'),
                  ROOT / 'plz_load' / 'best_parrotron.pth',
                  ROOT / 'plz_load' / 'best_parrotron_no_asr.pth']
    for path in candidates:
        if path and Path(path).is_file():
            return Path(path)
    return None


def preload_cuda12_libraries():
    """
    faster-whisper (CTranslate2) is built against CUDA 12, while PyTorch may ship CUDA 13.
    If the nvidia-cublas-cu12 / nvidia-cudnn-cu12 pip packages are installed, load them so Whisper can use the GPU.
    """
    import ctypes
    import glob
    import site
    found = False
    for site_dir in site.getsitepackages():
        for pattern in ('nvidia/cublas/lib/libcublas*.so.12', 'nvidia/cudnn/lib/libcudnn*.so.9'):
            for lib in sorted(glob.glob(os.path.join(site_dir, pattern))):
                try:
                    ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                    found = found or lib.endswith('libcublas.so.12')
                except OSError:
                    pass
    return found


def download_piper_voice(name):
    """Downloads <name>.onnx and <name>.onnx.json from the rhasspy/piper-voices repository"""
    model_path = VOICES_DIR / (name + '.onnx')
    config_path = VOICES_DIR / (name + '.onnx.json')
    if model_path.exists() and config_path.exists():
        return model_path

    lang_region, speaker, quality = name.split('-')
    base = 'https://huggingface.co/rhasspy/piper-voices/resolve/main/{}/{}/{}/{}/'.format(
        lang_region.split('_')[0], lang_region, speaker, quality)

    VOICES_DIR.mkdir(exist_ok=True)
    for path in (config_path, model_path):
        print('Downloading', base + path.name)
        tmp_path = path.with_suffix(path.suffix + '.part')
        urllib.request.urlretrieve(base + path.name, tmp_path)
        tmp_path.rename(path)
    return model_path


class Models:
    """Loads each model the first time it is needed"""

    def __init__(self):
        self.lock = threading.Lock()
        self._whisper = None
        self._piper = None
        self._parrotron = None
        self.parrotron_checkpoint = find_parrotron_checkpoint()

    def whisper(self, device=None):
        with self.lock:
            if self._whisper is None or device is not None:
                from faster_whisper import WhisperModel
                device = device or WHISPER_DEVICE
                if device == 'auto':
                    import ctranslate2
                    device = 'cuda' if ctranslate2.get_cuda_device_count() > 0 and preload_cuda12_libraries() else 'cpu'
                compute_type = 'float16' if device == 'cuda' else 'int8'
                print('Loading Whisper {} on {}'.format(WHISPER_MODEL, device))
                self._whisper = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute_type)
            return self._whisper

    def transcribe(self, signal):
        def run(model):
            segments, info = model.transcribe(signal, beam_size=5, vad_filter=True)
            return ' '.join(segment.text.strip() for segment in segments).strip(), info

        model = self.whisper()
        try:
            return run(model)
        except RuntimeError as err:
            if model.model.device != 'cuda':
                raise
            print('Whisper failed on GPU ({}), switching to CPU'.format(err))
            return run(self.whisper(device='cpu'))

    def piper(self):
        with self.lock:
            if self._piper is None:
                from piper import PiperVoice
                self._piper = PiperVoice.load(str(download_piper_voice(PIPER_VOICE)))
            return self._piper

    def parrotron(self):
        with self.lock:
            if self._parrotron is None:
                import torch
                from models.build import load_model
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                print('Loading Parrotron from {} on {}'.format(self.parrotron_checkpoint, device))
                self._parrotron = (load_model(self.parrotron_checkpoint, device), device)
            return self._parrotron


models = Models()
app = FastAPI(title='Speak & Repeat')


def read_audio(upload: UploadFile):
    """Decodes any browser recording (webm/ogg/wav/mp3...) to 16kHz mono float32 with ffmpeg"""
    data = upload.file.read()
    if not data:
        raise HTTPException(400, 'The recording is empty.')

    result = subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', 'pipe:0',
                             '-f', 'f32le', '-ac', '1', '-ar', str(SAMPLE_RATE), 'pipe:1'],
                            input=data, capture_output=True)
    signal = np.frombuffer(result.stdout, dtype=np.float32)
    if result.returncode != 0 or signal.size == 0:
        raise HTTPException(400, 'Could not decode the audio file.')
    return signal


def wav_response(signal, sample_rate):
    buf = io.BytesIO()
    sf.write(buf, signal, sample_rate, format='WAV', subtype='PCM_16')
    return Response(buf.getvalue(), media_type='audio/wav')


class TTSRequest(BaseModel):
    text: str


@app.get('/api/status')
def status():
    return {'whisper_model': WHISPER_MODEL,
            'voice': PIPER_VOICE,
            'parrotron': models.parrotron_checkpoint is not None}


@app.post('/api/transcribe')
def transcribe(audio: UploadFile = File(...)):
    signal = read_audio(audio)
    text, info = models.transcribe(signal)
    return {'text': text, 'duration': round(info.duration, 2)}


@app.post('/api/tts')
def tts(request: TTSRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(400, 'No text to speak.')
    if len(text) > MAX_TEXT_LENGTH:
        raise HTTPException(400, 'Text is too long (max {} characters).'.format(MAX_TEXT_LENGTH))

    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wav_file:
        models.piper().synthesize_wav(text, wav_file)
    return Response(buf.getvalue(), media_type='audio/wav')


@app.post('/api/parrotron')
def parrotron(audio: UploadFile = File(...)):
    if models.parrotron_checkpoint is None:
        raise HTTPException(503, 'No trained Parrotron checkpoint found. Train one with train.py first.')

    from inference import convert
    model, device = models.parrotron()
    wav, _ = convert(model, read_audio(audio), device)
    return wav_response(np.clip(wav, -1, 1), SAMPLE_RATE)


@app.get('/')
def index():
    return FileResponse(STATIC_DIR / 'index.html')


app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=int(os.environ.get('PORT', 8000)))
