import pdb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from models.model import MainModel
from models.loss import ContrastiveLoss
from dataloader import MyDataset, collate_func

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_trace_set = MyDataset(data_path="../val",
                require_trace=True)
val_text_set = MyDataset(data_path="../val",
                require_trace=False)

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
val_trace_loader = DataLoader(val_trace_set, batch_size=128,collate_fn=lambda batch: collate_func(batch, tokenizer, require_trace=True))
val_text_loader = DataLoader(val_text_set, batch_size=128,collate_fn=lambda batch: collate_func(batch, tokenizer, require_trace=False))

trace_model = MainModel(require_trace=True)
text_model = MainModel(require_trace=False)
trace_model.load_state_dict(torch.load('trace.pth'))
text_model.load_state_dict(torch.load('text.pth'))
trace_model.to(device)
text_model.to(device)
trace_model.eval()
text_model.eval()

criterion = ContrastiveLoss()
criterion.to(device)

with torch.no_grad():
    for trace_inputs, text_inputs in zip(val_trace_loader, val_text_loader):
        # Run trace model
        image_ids, semantic_features, position_features, text, trace = trace_inputs
        semantic_features, position_features, text, trace = semantic_features.to(device), position_features.to(device), text.to(device), trace.to(device)
        image_embedding, text_embedding = trace_model((semantic_features, position_features, text, trace))
        _, trace_logits = criterion(image_embedding, text_embedding)  
        # Run text model
        image_ids, semantic_features, position_features, text = text_inputs
        semantic_features, position_features, text = semantic_features.to(device), position_features.to(device), text.to(device)
        image_embedding, text_embedding = text_model((semantic_features, position_features, text))  
        _, text_logits = criterion(image_embedding, text_embedding)   
        # Compute similarity
        trace_logits = trace_logits.T
        target_labels = torch.arange(trace_logits.size(0)).to(trace_logits.device)
        _, trace_predicted_labels = trace_logits.topk(3, dim=1)
        trace_correct = (trace_predicted_labels == target_labels.view(-1, 1))
        text_logits = text_logits.T
        target_labels = torch.arange(text_logits.size(0)).to(text_logits.device)
        _, text_predicted_labels = text_logits.topk(3, dim=1)
        text_correct = (text_predicted_labels == target_labels.view(-1, 1))
        # Find the case
        for i in range(len(trace_inputs)):
            if (trace_correct[i,0] and not text_correct[i,0]) and (text_correct[i,1] or text_correct[i,2]):
                print(f'Trace: {image_ids[trace_predicted_labels[i][0]]} {image_ids[trace_predicted_labels[i][1]]} {image_ids[trace_predicted_labels[i][2]]}')
                print(f'Text: {image_ids[text_predicted_labels[i][0]]} {image_ids[text_predicted_labels[i][1]]} {image_ids[text_predicted_labels[i][2]]}')
        