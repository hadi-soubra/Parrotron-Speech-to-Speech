import torch
from torch import nn
import torch.nn.functional as F

def spectrogram_loss(mel_outputs_postnet, mel_outputs, gate_outputs, tts_targets, tts_lengths):
    """
    MSE on the spectrogram (padding frames ignored) + BCE on the stop token
    """
    max_len = tts_targets.size(1)
    frame_index = torch.arange(max_len, device=tts_targets.device).unsqueeze(0)
    tts_lengths = tts_lengths.to(tts_targets.device).unsqueeze(1)

    mask = (frame_index < tts_lengths).unsqueeze(-1).float() # (B, T, 1)
    num_values = mask.sum() * tts_targets.size(2)

    mel_loss = (((mel_outputs_postnet - tts_targets) ** 2 * mask).sum() + ((mel_outputs - tts_targets) ** 2 * mask).sum()) / num_values

    # 1 from the last real frame onward
    gate_targets = (frame_index >= tts_lengths - 1).float()
    gate_loss = F.binary_cross_entropy_with_logits(gate_outputs, gate_targets)

    return mel_loss + gate_loss


class ParrotronLoss(nn.Module):
    def __init__(self, asr_criterion):
        super(ParrotronLoss, self).__init__()

        self.asr_criterion = asr_criterion

    def forward(self, mel_outputs_postnet, mel_outputs, gate_outputs, txt_outputs, targets, tts_targets, tts_lengths):
        spec_loss = spectrogram_loss(mel_outputs_postnet, mel_outputs, gate_outputs, tts_targets, tts_lengths)
        asr_loss = self.asr_criterion(txt_outputs.contiguous().view(-1, txt_outputs.size(-1)), targets.contiguous().view(-1))

        return spec_loss + asr_loss


class ParrotronLossNoASR(nn.Module):
    def forward(self, mel_outputs_postnet, mel_outputs, gate_outputs, tts_targets, tts_lengths):
        return spectrogram_loss(mel_outputs_postnet, mel_outputs, gate_outputs, tts_targets, tts_lengths)
