import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
  def __init__(self):
    super(ContrastiveLoss, self).__init__()
    self.loss = nn.CrossEntropyLoss()
    self.t = nn.Parameter(torch.randn(1))

  def forward(self, image_embedding, text_embedding):
    similarity = torch.mm(F.normalize(image_embedding, p=2, dim=1), F.normalize(text_embedding, p=2, dim=1).T)
    logits = similarity * torch.exp(self.t)
    labels = torch.arange(logits.size(0)).to(logits.device)
    loss_a_b = self.loss(logits, labels)
    loss_b_a = self.loss(logits.T, labels)

    loss = (loss_a_b + loss_b_a) / 2
    return loss, similarity
