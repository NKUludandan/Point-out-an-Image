import torch
import torch.nn as nn

import pdb

class ImageRegionEmbedder(nn.Module):
    def __init__(self, hidden_size=512, dropout=0.3):
        super(ImageRegionEmbedder, self).__init__()
        
        self.projs = nn.ModuleList([nn.Linear(1, hidden_size) for _ in range(5)])
        self.semantic_ffn = nn.Sequential(nn.Linear(2048, hidden_size),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(hidden_size, hidden_size),
                                nn.LayerNorm(normalized_shape=hidden_size)
                                )
        self.position_ffn = nn.Sequential(nn.Linear(5*hidden_size, hidden_size),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden_size, hidden_size),
                                 nn.LayerNorm(normalized_shape=hidden_size)
                                 )
        self.ffn = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden_size, hidden_size),
                                 nn.LayerNorm(normalized_shape=hidden_size)
                                 )
        
    def forward(self, semantic_features, position_features):
        # Permute regional features
        if self.training:
            num_regions= 16
            perm_indices = torch.randperm(num_regions)
            # Apply permutation to regional features
            semantic_features[:,1:,:] = semantic_features[:,1:,:][:, perm_indices, :]
            position_features[:,1:,:] = position_features[:,1:,:][:, perm_indices, :]
        # MLP for position features 
        linear_proj = []
        for i, linear in enumerate(self.projs):
            linear_proj.append(linear(position_features[...,i].unsqueeze(-1)))
        position_features = torch.cat(linear_proj, dim=-1)
        position_features = self.position_ffn(position_features)
        # MLP for semantic features 
        semantic_features = self.semantic_ffn(semantic_features)
        # Compute output
        outputs = self.ffn(semantic_features + position_features)
        return outputs

class ImageTransformer(nn.Module):
  '''
    input: tensor (batch_size, text seq_len + trace seq_len, 512)
    output: tensor (batch_size, 512)
  '''
  def __init__(self, vocab_size=512, hidden_size=1024, num_layers=6, nhead=8, filter_size=4096, dropout=0.3):
    super(ImageTransformer, self).__init__()
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

