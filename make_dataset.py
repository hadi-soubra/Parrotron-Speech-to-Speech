"""
Builds a Parrotron training set from LibriSpeech: every utterance gets a TTS version of its transcript
spoken in one fixed Piper voice, which is the target the model learns to produce.

    .venv/bin/python make_dataset.py /data/LibriSpeech/train-clean-100 ./data/train ./label,csv/train.csv
    .venv/bin/python make_dataset.py /data/LibriSpeech/test-clean      ./data/test  ./label,csv/test.csv

LibriSpeech download: https://www.openslr.org/12
"""
import io
import os
import wave
import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from piper import PiperVoice

from models.data_loader import SAMPLE_RATE
from webapp.server import PIPER_VOICE, download_piper_voice

def read_transcripts(librispeech_dir):
    for trans_path in sorted(Path(librispeech_dir).rglob('*.trans.txt')):
        with open(trans_path) as f:
            for line in f:
                utt_id, text = line.strip().split(' ', 1)
                yield trans_path.parent / (utt_id + '.flac'), utt_id, text

def synthesize(voice, text):
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wav_file:
        voice.synthesize_wav(text.lower(), wav_file)
    buf.seek(0)
    signal, sample_rate = sf.read(buf, dtype='float32')
    return librosa.resample(signal, orig_sr=sample_rate, target_sr=SAMPLE_RATE)

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('librispeech_dir', help='e.g. LibriSpeech/train-clean-100')
    parser.add_argument('out_dir', help='where wav/, txt/ and TTS_wav/ are written')
    parser.add_argument('csv_path', help='manifest used by train.py')
    parser.add_argument('--max_seconds', type=float, default=8.0, help='skip longer utterances (GPU memory)')
    parser.add_argument('--limit', type=int, default=None, help='only process this many utterances')
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    for sub in ('wav', 'txt', 'TTS_wav'):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    voice = PiperVoice.load(str(download_piper_voice(PIPER_VOICE)))

    count = 0
    with open(args.csv_path, 'w') as csv_file:
        for flac_path, utt_id, text in read_transcripts(args.librispeech_dir):
            wav_path = out_dir / 'wav' / (utt_id + '.wav')
            txt_path = out_dir / 'txt' / (utt_id + '.txt')
            tts_path = out_dir / 'TTS_wav' / (utt_id + '.wav')

            if not tts_path.exists():
                signal, _ = librosa.load(flac_path, sr=SAMPLE_RATE)
                if len(signal) > args.max_seconds * SAMPLE_RATE:
                    continue
                sf.write(wav_path, signal, SAMPLE_RATE)
                txt_path.write_text(text)
                sf.write(tts_path, synthesize(voice, text), SAMPLE_RATE)

            csv_file.write('{},{},{}\n'.format(wav_path, txt_path, tts_path))
            count += 1
            if count % 500 == 0:
                print(count, 'utterances')
            if args.limit and count >= args.limit:
                break

    print('Wrote {} utterances to {}'.format(count, args.csv_path))

if __name__ == '__main__':
    main()
