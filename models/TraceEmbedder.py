import torch
import torch.nn as nn
from .util import PositionalEncoding

import pdb

class TraceBoxEmbedder(nn.Module):
  '''
    1. With coordinates
    input: tensor (batch_size, seq_len, 5)
    output: tensor (batch_size, seq_len, 512)

    2. With images
    input: tensor (batch_size, seq_len, 3, 224, 224)
    output: tensor (batch_size, seq_len, 512)
  '''
  def __init__(self, input_size=5, hidden_size=512, max_len=5000, dropout=0.3):
    super(TraceBoxEmbedder,self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.dropout = dropout

    self.position_encoding = PositionalEncoding(embedding_size=hidden_size, max_len=max_len)
    # If we only use coordinates
    self.linear_layers = nn.ModuleList([nn.Linear(1, hidden_size) for _ in range(input_size)])
    self.ffn = nn.Sequential(nn.Linear(5*hidden_size, hidden_size),
                  nn.ReLU(),
                  nn.Dropout(dropout),
                  nn.Linear(hidden_size, hidden_size),
                )

  def forward(self, x):
    linear_proj = []
    for i, linear in enumerate(self.linear_layers):
      linear_proj.append(linear(x[...,i].unsqueeze(-1)))
    x = torch.cat(linear_proj, dim=-1)
    x = self.ffn(x)
    return self.position_encoding(x.permute(1,0,2)).permute(1,0,2)
  
class TextTraceTransformer(nn.Module):
  '''
    input: tensor (batch_size, text seq_len + trace seq_len, 512)
    output: tensor (batch_size, 512)
  '''
  def __init__(self, vocab_size=512, hidden_size=1024, num_layers=6, nhead=8, filter_size=4096, dropout=0.3):
    super(TextTraceTransformer, self).__init__()
    self.embedder = nn.Linear(vocab_size, hidden_size)
    encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, dim_feedforward=filter_size, batch_first=True)
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    self.ffn = nn.Sequential(nn.Linear(hidden_size, 2*hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(2*hidden_size, hidden_size),
                  )
  def forward(self, x):
    x = self.embedder(x)
    x = self.transformer_encoder(x)
    x = x.mean(dim=1)
    x = self.ffn(x)

    return x