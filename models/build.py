import torch

from models.encoder import Encoder
from models.decoder import Decoder
from models.asr_decoder import ASR_Decoder
from models.model import Parrotron, Parrotron_No_ASR
from models.data_loader import SOS_token, EOS_token, PAD_token, char2index

def build_model(with_asr=True):
    enc = Encoder(rnn_hidden_size=256,
                  dropout=0.5,
                  bidirectional=True)

    dec = Decoder(target_dim=1025,
                  pre_net_dim=256,
                  rnn_hidden_size=1024,
                  encoder_dim=256*2,
                  attention_dim=128,
                  attention_filter_n=32,
                  attention_filter_len=31,
                  postnet_hidden_size=512,
                  postnet_filter=5,
                  dropout=0.5)

    if not with_asr:
        return Parrotron_No_ASR(enc, dec)

    asr_dec = ASR_Decoder(label_dim=len(char2index),
                          embedding_dim=64,
                          encoder_dim=256*2,
                          rnn_hidden_size=512,
                          second_rnn_hidden_size=256,
                          attention_dim=128,
                          attention_filter_n=32,
                          attention_filter_len=31,
                          sos_id=SOS_token,
                          eos_id=EOS_token,
                          pad_id=PAD_token)

    return Parrotron(enc, dec, asr_dec)

def load_model(checkpoint_path, device):
    """
    Loads either a Parrotron or Parrotron_No_ASR checkpoint, detected from its weights
    """
    state_dict = torch.load(checkpoint_path, map_location=device)
    with_asr = any(key.startswith('asr_decoder.') for key in state_dict)

    model = build_model(with_asr)
    model.load_state_dict(state_dict)
    return model.to(device).eval()
