import torch
import torch.nn as nn
from Model.PositionalEncoding import PositionalEncoding
from constants import EMBEDDING_SIZE, HEADS, ENCODER_LAYERS, FEEDFORWARD_SIZE, DECODER_LAYERS


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.positional_encoding = PositionalEncoding()
        self.transformer = nn.Transformer(
            d_model=EMBEDDING_SIZE,
            nhead=HEADS,
            num_encoder_layers=ENCODER_LAYERS,
            num_decoder_layers=DECODER_LAYERS,
            dim_feedforward=FEEDFORWARD_SIZE,
            batch_first=True
        )
        self.output = nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE)
    

    def forward(self, src, tgt):
        src = self.positional_encoding(src)
        src = self.transformer(src, tgt)
        output = self.output(src)
        return output

