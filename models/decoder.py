import torch
import torch.nn as nn
import random
from models.attention import LocationSensitiveAttention
from models.postnet import Postnet

class Prenet(nn.Module):
    def __init__(self, in_dim, out_dim, drop_out_p):
        super(Prenet, self).__init__()

        self.pre_net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(p=drop_out_p),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(p=drop_out_p)
        )

    def forward(self, x):
        output = self.pre_net(x)

        return output

class Decoder(nn.Module):
    def __init__(self, target_dim, pre_net_dim, rnn_hidden_size, encoder_dim, attention_dim, attention_filter_n, attention_filter_len, postnet_hidden_size, postnet_filter, dropout,
                 max_decoder_steps=1500, gate_threshold=0.5):
        super(Decoder, self).__init__()

        self.target_dim = target_dim
        self.rnn_hidden_size = rnn_hidden_size
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.max_decoder_steps = max_decoder_steps
        self.gate_threshold = gate_threshold

        self.pre_net = Prenet(target_dim, pre_net_dim, dropout)
        self.attention_rnn = nn.LSTMCell(pre_net_dim+self.encoder_dim, rnn_hidden_size)
        self.attention_layer = LocationSensitiveAttention(self.rnn_hidden_size, self.encoder_dim, self.attention_dim, attention_filter_n, attention_filter_len)
        self.decoder_rnn = nn.LSTMCell(self.encoder_dim + self.rnn_hidden_size, rnn_hidden_size)
        self.memory_layer = nn.Linear(in_features=self.encoder_dim, out_features=self.attention_dim, bias=True)
        self.projection_layer = nn.Linear(self.rnn_hidden_size + self.encoder_dim , target_dim, False)
        # stop token : predicts whether this frame is the last one
        self.gate_layer = nn.Linear(self.rnn_hidden_size + self.encoder_dim, 1, True)
        self.postnet = Postnet(target_dim, postnet_hidden_size, dropout, int((postnet_filter - 1) / 2)) # 5 = kernel size

    def init_states(self, encoder_inputs):
        batch_size, encoder_max_len, _ = encoder_inputs.size()

        self.attention_hidden = encoder_inputs.new_zeros(batch_size, self.rnn_hidden_size)
        self.attention_cell = encoder_inputs.new_zeros(batch_size, self.rnn_hidden_size)

        self.decoder_hidden = encoder_inputs.new_zeros(batch_size, self.rnn_hidden_size)
        self.decoder_cell = encoder_inputs.new_zeros(batch_size, self.rnn_hidden_size)

        self.attention_context = encoder_inputs.new_zeros(batch_size, self.encoder_dim)

        self.attention_weights_cum = encoder_inputs.new_zeros(batch_size, encoder_max_len)
        self.attention_weights = encoder_inputs.new_zeros(batch_size, encoder_max_len)

        self.memory = self.memory_layer(encoder_inputs)

    def go_frame(self, encoder_inputs):
        return encoder_inputs.new_zeros(encoder_inputs.size(0), self.target_dim)

    def forward_step(self, encoder_inputs, decoder_input):
        """
        PARAMS
        ------
        encoder_inputs: Encoder outputs. (B, T, e_F)
        decoder_input: Prenet output. (B, p_F)

        RETURNS
        -------
        decoder_output: spectrogram frame (B, target_dim)
        gate_output: stop token logit (B, 1)
        attention_weights: (B, T)
        """
        cell_input_1 = torch.cat((decoder_input, self.attention_context), -1) # pre-net and attention context vector concatenated

        self.attention_hidden, self.attention_cell = self.attention_rnn(cell_input_1, (self.attention_hidden, self.attention_cell))

        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)

        self.attention_context, self.attention_weights = self.attention_layer(self.attention_hidden, encoder_inputs, self.memory, attention_weights_cat)
        self.attention_weights_cum = self.attention_weights_cum + self.attention_weights
        decoder_input = torch.cat((self.attention_hidden, self.attention_context), -1)

        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(decoder_input, (self.decoder_hidden, self.decoder_cell))

        decoder_hidden_attention_context = torch.cat((self.decoder_hidden, self.attention_context), dim=1) # concatenation of the LSTM output and attention context vector

        decoder_output = self.projection_layer(decoder_hidden_attention_context) # projected through a linear transform
        gate_output = self.gate_layer(decoder_hidden_attention_context)

        return decoder_output, gate_output, self.attention_weights

    def postnet_residual(self, mel_outputs):
        mel_outputs_postnet = self.postnet(mel_outputs)
        return mel_outputs + mel_outputs_postnet.transpose(1, 2)

    def forward(self, encoder_inputs, decoder_inputs, tf=1.0):
        """
        Decoder forward pass for training
        PARAMS
        ------
        encoder_inputs: Encoder outputs (B, T, e_F)
        decoder_inputs: target spectrogram (B, T, d_F)
        tf: teacher forcing ratio, per step probability of feeding the ground truth frame

        RETURNS
        -------
        final_mel_outputs: postnet output (B, T, d_F)
        mel_outputs: decoder output before postnet (B, T, d_F)
        gate_outputs: stop token logits (B, T)
        """
        self.init_states(encoder_inputs)
        max_len = decoder_inputs.size(1)

        # teacher forcing pre_net
        teacher_inputs = torch.cat((self.go_frame(encoder_inputs).unsqueeze(1), decoder_inputs), dim=1)
        teacher_inputs = self.pre_net(teacher_inputs).transpose(0, 1) # T+1, B, F

        mel_outputs, gate_outputs = [], []
        decoder_input = teacher_inputs[0]
        while len(mel_outputs) < max_len:
            mel_output, gate_output, _ = self.forward_step(encoder_inputs, decoder_input)

            mel_outputs += [mel_output]
            gate_outputs += [gate_output.squeeze(1)]

            if random.random() < tf:
                decoder_input = teacher_inputs[len(mel_outputs)]
            else:
                decoder_input = self.pre_net(mel_output)

        mel_outputs = torch.stack(mel_outputs, dim=1)
        gate_outputs = torch.stack(gate_outputs, dim=1)

        return self.postnet_residual(mel_outputs), mel_outputs, gate_outputs

    def inference(self, encoder_inputs, max_decoder_steps=None):
        """
        Decoder forward pass without target, stops when the stop token fires for every utterance
        RETURNS
        -------
        final_mel_outputs: (B, T, d_F)
        lengths: (B,) number of frames until the stop token
        """
        max_decoder_steps = max_decoder_steps or self.max_decoder_steps
        self.init_states(encoder_inputs)

        batch_size = encoder_inputs.size(0)
        lengths = encoder_inputs.new_full((batch_size,), max_decoder_steps, dtype=torch.long)
        finished = encoder_inputs.new_zeros(batch_size, dtype=torch.bool)

        mel_outputs = []
        decoder_input = self.go_frame(encoder_inputs)
        while len(mel_outputs) < max_decoder_steps:
            mel_output, gate_output, _ = self.forward_step(encoder_inputs, self.pre_net(decoder_input))
            mel_outputs += [mel_output]

            stop = torch.sigmoid(gate_output.squeeze(1)) > self.gate_threshold
            lengths[stop & ~finished] = len(mel_outputs)
            finished |= stop
            if finished.all():
                break

            decoder_input = mel_output

        mel_outputs = torch.stack(mel_outputs, dim=1)

        return self.postnet_residual(mel_outputs), lengths
