import pdb
import torch
import torch.nn as nn

from .ImageEmbedder import ImageRegionEmbedder, ImageTransformer
from .TextEmbedder import TextEmbedder
from .TraceEmbedder import TraceBoxEmbedder, TextTraceTransformer

class MainModel(nn.Module):
  def __init__(self, require_trace=False):
    super(MainModel, self).__init__()

    self.require_trace = require_trace

    self.image_embedder = ImageRegionEmbedder()
    self.text_embedder = TextEmbedder()
    self.image_transformer = ImageTransformer()
    self.text_trace_transformer = TextTraceTransformer()
    if require_trace:
        self.trace_embedder = TraceBoxEmbedder()
      
  def forward(self, x):
    # Trace embedder
    if self.require_trace:
      semantic_features, position_features, text, trace = x
      trace_embedding = self.trace_embedder(trace)
    else:
      semantic_features, position_features, text = x
    # Image and text embedder 
    image_embedding = self.image_embedder(semantic_features, position_features)
    text_embedding = self.text_embedder(text)
    # Image and text-trace transformers
    image_output = self.image_transformer(image_embedding)
    if self.require_trace:
      text_output = self.text_trace_transformer(torch.cat((text_embedding, trace_embedding), dim=1))
    else:
      text_output = self.text_trace_transformer(text_embedding)
    return image_output, text_output