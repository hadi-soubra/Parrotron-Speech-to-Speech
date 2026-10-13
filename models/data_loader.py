import os
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

SAMPLE_RATE = 16000
N_FFT_INPUT = 1024
N_FFT_TARGET = 2048
HOP_LENGTH = 200 # 12.5ms
WIN_LENGTH = 800 # 50ms
N_MELS = 80
LOG_EPS = 1e-5

LABEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'label,csv', 'english_unit.labels')

def load_label(label_path):
    char2index = dict() # [ch] = id
    index2char = dict() # [id] = ch
    with open(label_path, 'r') as f:
        for no, line in enumerate(f):
            if line[0] == '#':
                continue

            index, char = line.split('   ')
            char = char.strip()
            if len(char) == 0:
                char = ' '

            char2index[char] = int(index)
            index2char[int(index)] = char

    return char2index, index2char

char2index, index2char = load_label(LABEL_PATH)
SOS_token = char2index['<s>']
EOS_token = char2index['</s>']
PAD_token = char2index['_']


class AttrDict(dict):
    """
    Dictionary whose keys can be accessed as attributes.
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)

    def __getattr__(self, item):
        if item not in self:
            return None
        if type(self[item]) is dict:
            self[item] = AttrDict(self[item])
        return self[item]

    def __setattr__(self, item, value):
        self.__dict__[item] = value


def input_features(signal):
    """
    signal: 16kHz mono float waveform
    RETURNS: 80-dim log-Mel, (T, 80)
    """
    spec = librosa.feature.melspectrogram(y=signal,
                                          sr=SAMPLE_RATE,
                                          n_fft=N_FFT_INPUT,
                                          hop_length=HOP_LENGTH,
                                          win_length=WIN_LENGTH,
                                          window='hann',
                                          n_mels=N_MELS,
                                          fmin=125,
                                          fmax=7600)

    spec = librosa.power_to_db(spec, ref=np.max)
    return torch.from_numpy(spec).transpose(0, 1).float()

def target_features(signal):
    """
    signal: 16kHz mono float waveform
    RETURNS: 1025-dim log-magnitude STFT, (T, 1025)
    """
    spec = librosa.stft(signal, n_fft=N_FFT_TARGET, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, window='hann')
    spec = np.log(np.maximum(np.abs(spec), LOG_EPS))
    return torch.from_numpy(spec).transpose(0, 1).float()

def features_to_wav(spec, n_iter=60):
    """
    spec: log-magnitude STFT, (T, 1025) tensor or array
    RETURNS: waveform reconstructed with Griffin-Lim
    """
    if torch.is_tensor(spec):
        spec = spec.detach().cpu().numpy()
    magnitude = np.exp(spec).T
    return librosa.griffinlim(magnitude, n_iter=n_iter, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, window='hann')

def load_wav(audio_path):
    signal, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    return signal


class SpectrogramDataset(Dataset):
    def __init__(self, manifest_filepath):
        """
        manifest: <wav-path>,<script-path>,<TTS_wav-path> per line
        """
        with open(manifest_filepath) as f:
            ids = [x.strip().split(',') for x in f if x.strip()]

        self.ids = ids
        self.size = len(ids)

    def __getitem__(self, index):
        audio_path, transcript_path, tts_path = self.ids[index][:3]

        spect = input_features(load_wav(audio_path))
        tts_spect = target_features(load_wav(tts_path))
        transcript = self.parse_transcript(transcript_path)

        return spect, transcript, tts_spect

    def parse_transcript(self, transcript_path):
        """
        RETURNS
        -------
        transcript_list: list [..., eos token] sos token은 없음
        """
        with open(transcript_path, 'r', encoding='utf8') as f:
            transcript = f.read().strip().upper()

        transcript_list = [char2index[char] for char in transcript if char in char2index]
        transcript_list.append(EOS_token)

        return transcript_list

    def __len__(self):
        return self.size

class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn

def _collate_fn(batch):
    seq_lengths = [s[0].size(0) for s in batch]
    target_lengths = [len(s[1]) - 1 for s in batch] # eos 제거한거임 -1로
    tts_seq_lengths = [s[2].size(0) for s in batch]

    batch_size = len(batch)
    feat_size = batch[0][0].size(1)
    tts_feat_size = batch[0][2].size(1)

    # padding frames are filled with silence (-80dB input, log of LOG_EPS target)
    seqs = torch.full((batch_size, max(seq_lengths), feat_size), -80.0)
    tts_seqs = torch.full((batch_size, max(tts_seq_lengths), tts_feat_size), float(np.log(LOG_EPS)))
    targets = torch.full((batch_size, max(len(s[1]) for s in batch)), PAD_token, dtype=torch.long)

    for x, (tensor, target, tts_tensor) in enumerate(batch):
        seqs[x, :tensor.size(0)] = tensor
        tts_seqs[x, :tts_tensor.size(0)] = tts_tensor
        targets[x, :len(target)] = torch.LongTensor(target)

    # seqs (B, S_L, S_dim)
    # targets (B, T_L)
    # tts_seqs (B, TS_L, TS_dim)

    return seqs, targets, tts_seqs, torch.LongTensor(seq_lengths), torch.LongTensor(target_lengths), torch.LongTensor(tts_seq_lengths)
