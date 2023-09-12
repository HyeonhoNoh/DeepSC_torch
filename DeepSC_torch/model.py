import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import math

from typing import Iterable, List, Optional
from torch import Tensor

class Semantic_Transformer(nn.Module):
    def __init__(self,
                 config,
                 SRC_VOCAB_SIZE,
                 TGT_VOCAB_SIZE):
        super().__init__()

        self.transformer_enc = TransformerEnc(config.param['num_encoder_layers'],
                                         config.param['emb_size'],
                                         config.param['nhead'],
                                         SRC_VOCAB_SIZE,
                                         config.param['dim_feedforward'],
                                         config.param['dropout'])
        self.transformer_dec = TransformerDec(config.param['num_decoder_layers'],
                                         config.param['emb_size'],
                                         config.param['nhead'],
                                         TGT_VOCAB_SIZE,
                                         config.param['dim_feedforward'],
                                         config.param['dropout'])

class Channel_Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.channel_enc = ChannelEnc(config.param['emb_size'], config.param['coding_rate'])
        self.channel_dec = ChannelDec(config.param['emb_size'], config.param['coding_rate'])


class Transceiver(nn.Module):
    def __init__(self,
                 config,
                 SRC_VOCAB_SIZE,
                 TGT_VOCAB_SIZE):
        super().__init__()

        self.Semantic_Transformer = Semantic_Transformer(config, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE)
        self.Channel_Transformer = Channel_Transformer(config)
        self.channel = Channels(config.param['channel'])

    def forward(self,
                src: Tensor,
                tgt: Tensor,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                src_padding_mask: Optional[Tensor] = None,
                tgt_padding_mask: Optional[Tensor] = None,
                memory_padding_mask: Optional[Tensor] = None,
                noise_var = 0.1
                ):
        semantic_features = self.Semantic_Transformer.transformer_enc(src, src_mask, src_padding_mask)
        channel_encoding = self.Channel_Transformer.channel_enc(semantic_features)
        received_channel_encoding = self.channel(channel_encoding, noise_var)
        channel_decoding = self.Channel_Transformer.channel_dec(received_channel_encoding)
        outputs = self.Semantic_Transformer.transformer_dec(tgt, channel_decoding, tgt_mask, tgt_padding_mask, memory_padding_mask)

        return outputs, channel_decoding, semantic_features

    def semantic_forward(self,
                src: Tensor,
                tgt: Tensor,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                src_padding_mask: Optional[Tensor] = None,
                tgt_padding_mask: Optional[Tensor] = None,
                memory_padding_mask: Optional[Tensor] = None
                ):
        semantic_features = self.Semantic_Transformer.transformer_enc(src, src_mask, src_padding_mask)
        outputs = self.Semantic_Transformer.transformer_dec(tgt, semantic_features, tgt_mask, tgt_padding_mask, memory_padding_mask)

        return outputs, semantic_features

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class ChannelEnc(nn.Module):
    def __init__(self, size1, size2):
        super(ChannelEnc, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(size1, 256),
            nn.ReLU(),
            nn.Linear(256, size2)
        )
        self.powernorm = lambda x: x / torch.sqrt(2 * torch.mean(x ** 2))

    def forward(self, inputs):
        shape = inputs.shape
        inputs = inputs.view(inputs.size(0) * inputs.size(1), -1)  # flatten
        output = self.layer1(inputs)
        output = self.powernorm(output)
        return output.reshape((*(shape[0], shape[1]),-1))  # input size와 같게끔 reshape

class ChannelDec(nn.Module):
    def __init__(self, size1, size2):
        super(ChannelDec, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(size2, size1),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(size1, 512),
            nn.ReLU(),
            nn.Linear(512, size1),
        )
        self.layernorm = nn.LayerNorm(size1)


    def forward(self, receives):
        shape = receives.shape
        receives = receives.view(receives.size(0) * receives.size(1), -1)  # flatten
        x1 = self.layer(receives)
        x2 = self.layer2(x1)
        output = self.layernorm(x1+x2)

        return output.reshape((*(shape[0], shape[1]),-1))  # input size와 같게끔 reshape

class TransformerEnc(nn.Module):

    def __init__(self,
                 num_encoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.pos_encoder = PositionalEncoding(emb_size, dropout)
        encoder_layers = TransformerEncoderLayer(emb_size, nhead, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        self.d_model = emb_size

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.tok_emb.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor, src_padding_mask: Tensor) -> Tensor:
        output = self.transformer_encoder(self.pos_encoder(self.tok_emb(src)), src_mask, src_padding_mask)

        return output

class TransformerDec(nn.Module):
    def __init__(self,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.pos_encoder = PositionalEncoding(emb_size, dropout)
        decoder_layers = TransformerDecoderLayer(emb_size, nhead, dim_feedforward, dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layers, num_decoder_layers)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.d_model = emb_size

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.tok_emb.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor, tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor) -> Tensor:
        output = self.transformer_decoder(self.pos_encoder(self.tok_emb(tgt)), memory, tgt_mask, None,
                                          tgt_padding_mask, memory_key_padding_mask)

        return self.generator(output)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor, tgt_padding_mask: Tensor,
               memory_key_padding_mask: Tensor):
        output = self.transformer_decoder(self.pos_encoder(self.tok_emb(tgt)), memory, tgt_mask, None,
                                          tgt_padding_mask, memory_key_padding_mask)
        return output

class Channels(nn.Module):
    def __init__(self, channel='AWGN'):
        super().__init__()
        self.channel = channel

    def forward(self, inputs, n_std=0.1):
        if self.channel == 'AWGN':
            out = self.awgn(inputs, n_std)
        elif self.channel == 'Rician':
            out = self.fading(inputs, 1)
        else:
            out = self.fading(inputs, 0)

        return out

    def awgn(self, inputs, n_std):
        x = inputs
        y = x + torch.randn_like(x) * (n_std / 2)
        return y

    def fading(self, inputs, K=1, detector='MMSE'):
        x = inputs
        bs, sent_len, d_model = x.shape
        mean = torch.sqrt(K / (2 * (K + 1)))
        std = torch.sqrt(1 / (2 * (K + 1)))
        x = x.view(bs, -1, 2)
        x_real = x[:, :, 0]
        x_imag = x[:, :, 1]
        x_complex = x_real + 1j * x_imag

        # Create the fading factor
        h_real = torch.randn((1,)) * std + mean
        h_imag = torch.randn((1,)) * std + mean
        h_complex = h_real + 1j * h_imag

        # Create the noise vector
        n = torch.randn_like(x) * self.n_std
        n_real = n[:, :, 0]
        n_imag = n[:, :, 1]
        n_complex = n_real + 1j * n_imag

        # Transmit Signals
        y_complex = x_complex * h_complex + n_complex

        # Employ the perfect CSI here
        if detector == 'LS':
            h_complex_conj = torch.conj(h_complex)
            x_est_complex = y_complex * h_complex_conj / (h_complex * h_complex_conj)
        elif detector == 'MMSE':
            # MMSE Detector
            h_complex_conj = torch.conj(h_complex)
            a = h_complex * h_complex_conj + (self.n_std * self.n_std * 2)
            x_est_complex = y_complex * h_complex_conj / a
        else:
            raise ValueError("detector must be 'LS' or 'MMSE'")

        x_est_real = x_est_complex.real
        x_est_img = x_est_complex.imag

        x_est_real = torch.unsqueeze(x_est_real, -1)
        x_est_img = torch.unsqueeze(x_est_img, -1)

        x_est = torch.cat([x_est_real, x_est_img], dim=-1)
        x_est = x_est.reshape((bs, sent_len, -1))

        return x_est
