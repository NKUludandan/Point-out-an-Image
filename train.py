import pdb
import tqdm
import argparse

import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from models.model import MainModel
from models.loss import ContrastiveLoss
from dataloader import MyDataset, collate_func

torch.backends.cudnn.enabled = True

# Function which comptes the recall at top-k given a similarity score matrix between text and images
def recall_at_k(sim_matrix, k):
  # to make text rows, and images columns
  sim_matrix = sim_matrix.T
  target_labels = torch.arange(sim_matrix.size(0)).to(sim_matrix.device)
  _, predicted_labels = sim_matrix.topk(k, dim=1)
  correct = (predicted_labels == target_labels.view(-1, 1)).sum().item()
  return correct / sim_matrix.size(0)

# Function used to evaluate the model on a validation set
def evaluate_model(val_data_loader, criterion, k):
  model.eval()
  val_recall = .0
  val_loss = .0
  batch_len = len(val_data_loader)

  with torch.no_grad():
    for val_inputs in val_data_loader:
      if args['trace']:
        _, semantic_features, position_features, text, trace = val_inputs
        semantic_features, position_features, text, trace = semantic_features.to(device), position_features.to(device), text.to(device), trace.to(device)
        image_embedding, text_embedding = model((semantic_features, position_features, text, trace))
      else:
        _, semantic_features, position_features, text = val_inputs
        semantic_features, position_features, text = semantic_features.to(device), position_features.to(device), text.to(device)
        image_embedding, text_embedding = model((semantic_features, position_features, text)) 

      loss, logits = criterion(image_embedding, text_embedding)
      val_batch_recall = recall_at_k(logits, k)
      val_recall += val_batch_recall
      val_loss += loss.item()

  avg_val_recall = val_recall / batch_len
  avg_val_loss = val_loss / batch_len

  return avg_val_recall, avg_val_loss

# Assumes model, train/val loader,  criterion and loss
def train(model, train_loader, val_loader, optimizer, scheduler, criterion, args, device):
  best_val_recall = 0
  best_model_path = 'best_model.pth'
  total_step = 0
  print_step = 32
  accum_iter = 32  

  for epoch in range(args['n_epochs']):
    total_loss = 0.0
    total_recall = 0.0
    # Zero the gradients
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}")):
      # Compute training step and update eval metrics
      model.train()
      # Add device and forward
      if args['trace']:
        _, semantic_features, position_features, text, trace = batch
        semantic_features, position_features, text, trace = semantic_features.to(device), position_features.to(device), text.to(device), trace.to(device)
        image_embedding, text_embedding = model((semantic_features, position_features, text, trace))
      else:
        _, semantic_features, position_features, text = batch
        semantic_features, position_features, text = semantic_features.to(device), position_features.to(device), text.to(device)
        image_embedding, text_embedding = model((semantic_features, position_features, text))    
      # Calculate the loss
      loss, logits = criterion(image_embedding, text_embedding)
      # Back-propagate gradients
      loss = loss / accum_iter
      loss.backward()
      # nn.utils.clip_grad_norm_(model.parameters(), 5.0)
      total_loss += loss.item()
      # Update model
      if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
        optimizer.step()
        optimizer.zero_grad()
      # Compute batch training recall
      with torch.no_grad():
        recall = recall_at_k(logits, args['k'])
      total_recall += recall
      # Evaluation
      if ((batch_idx+1) % print_step == 0):
        # Train
        avg_loss = total_loss
        avg_recall = total_recall / print_step
        print(f"Epoch [{epoch+1}/{args['n_epochs']}], Train Avg. Loss: {avg_loss:.4f}, Train Avg. Recall: {avg_recall:.4f}" )
        wandb.log( {"Epoch": epoch+1, "Step": batch_idx+1, "Train Avg. Loss": avg_loss, "Train Avg. Recall": avg_recall, 'learning_rate': optimizer.param_groups[0]['lr']})
        total_loss = 0.
        total_recall = 0.
        # Validation
        avg_val_recall, avg_val_loss = evaluate_model(val_loader, criterion, args['k'])
        # Print validation acc
        print(f"Epoch [{epoch+1}/{args['n_epochs']}], Validation Avg. Loss: {avg_val_loss:.4f}, Validation Avg. Recall: {avg_val_recall:.4f}")
        wandb.log({"Validation Avg. Recall": avg_val_recall, "Validation Avg. Loss": avg_val_loss})

      total_step += 1
      if total_step > args['steps']:
        print("Reached max training steps", total_step)
        return
    # Update learning rate
    scheduler.step()
    # Save the best model
    if avg_val_recall > best_val_recall:
      best_val_recall = avg_val_recall
      try:
        state_dict = model.module.state_dict()
      except AttributeError:
        state_dict = model.state_dict()
      torch.save(state_dict, best_model_path)
      print("Saved the best model!")

if __name__ == '__main__':
    # define ArgParser
    parser = argparse.ArgumentParser(description='Model Parser')
    parser.add_argument('-e','--n_epochs', default=1000000, type=int)
    parser.add_argument('-s','--steps', default=250000, type=int)
    parser.add_argument('-l','--learning_rate', default=1e-4, type=float)
    parser.add_argument('-b', '--batch_size', default=256, type=int)
    parser.add_argument('-wu', '--warm_up', default=20, type=int)
    parser.add_argument('-wd', '--weight_decay', default=25, type=int)
    parser.add_argument('-t', '--trace', action='store_true', default=False)
    parser.add_argument('-k', '--k', default=5, type=int)
    parser.add_argument('-p', '--pretrain', default=None, type=str)
    parser.add_argument('-test', '--test', default=None, type=str)

    args = parser.parse_args().__dict__

    # Set device (CPU or GPU if available), maybe multigpus?
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set random seed
    # seed = 42
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)

    # create dataset, remember to adjust the path parameters according to the real path
    train_set = MyDataset(data_path="../train",
                 require_trace=args['trace'])
    
    val_set = MyDataset(data_path="../val",
                require_trace=args['trace'])
    
    # define dataloaders
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    train_loader = DataLoader(train_set,batch_size=args['batch_size'],shuffle=True,collate_fn=lambda batch: collate_func(batch, tokenizer, require_trace=args['trace']))
    val_loader = DataLoader(val_set,batch_size=args['batch_size'],shuffle=True,collate_fn=lambda batch: collate_func(batch, tokenizer, require_trace=args['trace']))
    
    # define model hyperparams and initialize the model: 
    model = MainModel(require_trace=args['trace'])
    # Loading pretrain model or test model
    if args['pretrain'] != None:
      state_dict = torch.load(args['pretrain'])
      model.load_state_dict(state_dict)
    elif args['test'] != None:
      state_dict = torch.load(args['test'])
      model.load_state_dict(state_dict)
    # Multi GPUs
    if (torch.cuda.device_count() > 1) and (device != torch.device("cpu")):
       model= nn.DataParallel(model)
    model.to(device)
    # define loss function
    criterion = ContrastiveLoss()
    criterion.to(device)

    if args['test'] != None:
      avg_val_recall, avg_val_loss = evaluate_model(val_loader, criterion, args['k'])  
      print(f'Validation Avg. Loss: {avg_val_loss:.4f}, Validation Avg. Recall: {avg_val_recall:.4f}')
    else:
      # define optimizer (paper uses Adam with default hyper params)
      optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'])
      # define scheduler with warm up and weight decay
      def lr_lambda(epoch):
        if epoch < args['warm_up']:
            decay_factor = (epoch + 1) / args['warm_up']
            return decay_factor
        else:
            decay_factor = 0.95** ((epoch - args['warm_up']) // args['weight_decay'])
            return decay_factor
      scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
      # set up weights and biases
      # init wandb
      run = wandb.init(project='CCS2MainModel')
      # set config
      config = wandb.config
      config.learning_rate = args['learning_rate']
      config.max_steps = args['steps']
      config.epochs = args['n_epochs']
      # start training
      train(model, train_loader, val_loader, optimizer, scheduler, criterion, args ,device)
