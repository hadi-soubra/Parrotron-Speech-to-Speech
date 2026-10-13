import torch.nn as nn

class Parrotron(nn.Module):
    def __init__(self, encoder, spectrogram_decoder, asr_decoder):
        super(Parrotron, self).__init__()

        self.encoder = encoder
        self.spectrogram_decoder = spectrogram_decoder
        self.asr_decoder = asr_decoder

    def forward(self, inputs, tts_inputs, targets, tf=1.0):
        encoder_outputs = self.encoder(inputs)

        mel_outputs_postnet, mel_outputs, gate_outputs = self.spectrogram_decoder(encoder_outputs, tts_inputs, tf)

        txt_outputs = self.asr_decoder(encoder_outputs, targets)

        return mel_outputs_postnet, mel_outputs, gate_outputs, txt_outputs

    def inference(self, inputs, max_decoder_steps=None):
        encoder_outputs = self.encoder(inputs)

        return self.spectrogram_decoder.inference(encoder_outputs, max_decoder_steps)

class Parrotron_No_ASR(nn.Module):
    def __init__(self, encoder, spectrogram_decoder):
        super(Parrotron_No_ASR, self).__init__()

        self.encoder = encoder
        self.spectrogram_decoder = spectrogram_decoder

    def forward(self, inputs, tts_inputs, targets=None, tf=1.0):
        encoder_outputs = self.encoder(inputs)

        mel_outputs_postnet, mel_outputs, gate_outputs = self.spectrogram_decoder(encoder_outputs, tts_inputs, tf)

        return mel_outputs_postnet, mel_outputs, gate_outputs, None

    def inference(self, inputs, max_decoder_steps=None):
        encoder_outputs = self.encoder(inputs)

        return self.spectrogram_decoder.inference(encoder_outputs, max_decoder_steps)
