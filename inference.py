import os
import argparse

import torch
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.image

from models.build import load_model
from models.data_loader import input_features, features_to_wav, load_wav, SAMPLE_RATE

def convert(model, signal, device):
    """
    signal: 16kHz waveform of anyone speaking
    RETURNS: (waveform in the TTS voice, log-magnitude spectrogram (T, 1025))
    """
    with torch.no_grad():
        seqs = input_features(signal).unsqueeze(0).to(device)
        spec, lengths = model.inference(seqs)
        spec = spec[0, :lengths[0]]

    return features_to_wav(spec), spec.cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description='Convert speech with a trained Parrotron checkpoint')
    parser.add_argument('checkpoint')
    parser.add_argument('wavs', nargs='+')
    parser.add_argument('--out_dir', default='./test_wav')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.checkpoint, device)

    os.makedirs(args.out_dir, exist_ok=True)
    for wav_path in args.wavs:
        name = os.path.splitext(os.path.basename(wav_path))[0]
        wav, spec = convert(model, load_wav(wav_path), device)

        sf.write(os.path.join(args.out_dir, name + '.wav'), wav, SAMPLE_RATE)
        matplotlib.image.imsave(os.path.join(args.out_dir, name + '.png'), spec.T[::-1])
        print('{} -> {}'.format(wav_path, os.path.join(args.out_dir, name + '.wav')))

if __name__ == '__main__':
    main()
