import torch
import torch.nn as nn
from constants import MAX_INPUT, EMBEDDING_SIZE


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=EMBEDDING_SIZE, max_len=MAX_INPUT):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0).cuda()

    def forward(self, x):
        return x + self.encoding[:, :x.size(0)]  # Add positional encoding to input