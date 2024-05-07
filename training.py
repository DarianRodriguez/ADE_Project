
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers import BertForTokenClassification, AdamW
from tqdm import tqdm

import config


def train_fn(data_loader, model, optimizer, device):
    model.train() # Set the model in training mode
    total_loss = 0
    
    for batch in tqdm(data_loader, desc='Training', unit='batches'):
        input_ids = batch['input_ids'].squeeze(1).to(device)
        token_type_ids = batch['token_type_ids'].squeeze(1).to(device)
        attention_mask = batch['attention_mask'].squeeze(1).to(device)
        labels = batch['labels'].squeeze(1).to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    
    avg_train_loss = total_loss / len(data_loader)
    return avg_train_loss


def eval_fn(data_loader, model, device):
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluation', unit='batches'):
            input_ids = batch['input_ids'].squeeze(1).to(device)
            token_type_ids = batch['token_type_ids'].squeeze(1).to(device)
            attention_mask = batch['attention_mask'].squeeze(1).to(device)
            labels = batch['labels'].squeeze(1).to(device)
            
            outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            total_loss += loss.item()

            # Append predictions and true labels
            predictions.append(outputs.logits.argmax(dim=-1))
            true_labels.append(labels)
    
            avg_eval_loss = total_loss / len(data_loader)

    # Concatenate predictions and true labels along the batch dimension
    predictions = torch.cat(predictions, dim=0)
    true_labels = torch.cat(true_labels, dim=0)

    # Reshape predictions and true labels to create a matrix
    num_samples = predictions.size(0)
    predictions_matrix = predictions.view(num_samples, -1)
    true_labels_matrix = true_labels.view(num_samples, -1)

    return avg_eval_loss, predictions_matrix, true_labels_matrix


def train_engine(epoch, train_data, valid_data):

    # Define your train and validation data loaders
    train_loader = DataLoader(train_data, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=config.VALID_BATCH_SIZE, shuffle=True)

    model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=5)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    best_eval_loss = float('inf')
    for i in range(epoch):
        train_loss = train_fn(train_loader, model, optimizer, device)
        eval_loss, eval_predictions, true_labels = eval_fn(valid_loader, model, device)
        
        print(f"Epoch {i} , Train loss: {train_loss}, Eval loss: {eval_loss}")

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            print("Saving the model")
            #torch.save(model.state_dict(), 'best_model.pt')

    return model, eval_predictions, true_labels