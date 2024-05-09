
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertForTokenClassification
from tqdm import tqdm
import torch.nn as nn
from transformers import BertModel

import config

def extract_entity_drug(labels):

    pair = [1,2]

    # Find indices where transition from 1 to 2 occurs (Drug)
    #transition_1_to_2_indices = torch.where((labels[:-1] == pair[0]) & (labels[1:] == pair[1]))[0] 
    transition_1_to_2_indices = torch.where((labels[:-1] == pair[0]) & (labels[1:] != pair[0]))[0] 
    transition_2_to_any = torch.where((labels[:-1] == pair[1]) & (labels[1:] != pair[1]))[0]

    if len(transition_2_to_any) != len(transition_1_to_2_indices): 
        length = len(labels)-1
        transition_2_to_any = torch.cat((transition_2_to_any, torch.tensor([length],device=labels.device)))

    # Combine start and end indices to form drug spans
    drug_spans = torch.stack((transition_1_to_2_indices, transition_2_to_any), dim=1)
    #drug_spans = zip(transition_1_to_2_indices,transition_2_to_any)

    return drug_spans

class DrugAwareModel(nn.Module):
    def __init__(self, num_labels):
        super(DrugAwareModel, self).__init__()
        self.bert = BertModel.from_pretrained('dmis-lab/biobert-v1.1')
        self.hidden_size = self.bert.config.hidden_size
        self.linear_layer = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.sigmoid = nn.Sigmoid()
        self.loss_fct = nn.BCELoss()

    def forward(self, input_ids, attention_mask, drug_labels):

        # Move input tensors to the same device as BioBERT
        input_ids = input_ids.to(self.bert.device)
        attention_mask = attention_mask.to(self.bert.device)
        drug_labels = drug_labels.to(self.bert.device)

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # Get the sequence output from BioBERT (embeddings)

        all_mean_embeddings = []
        all_logits = []

        # Get DRUG span
        for i, label_seq in enumerate(drug_labels):  # Loop through each label sequence in the batch
            drug_span = extract_entity_drug(label_seq)  # Extract spans for one sequence


            for drug_number in range(drug_span.shape[0]):
                drug_embeddings = sequence_output[i, drug_span[drug_number, 0]:drug_span[drug_number, 1]+1, :]  # Efficient indexing across drug spans/ sentence
                pooled_embeddings = torch.mean(drug_embeddings, dim=0) # Mean pooling
                all_mean_embeddings.append(pooled_embeddings)
                mean_tensor = (sequence_output[i] + pooled_embeddings)/2

                # Apply linear layer
                logits = self.linear_layer(mean_tensor.unsqueeze(0))  # Unsqueeze to add batch dimension
                probabilities = self.sigmoid(logits)
                all_logits.append(probabilities)
                
        all_logits = torch.cat(all_logits, dim=0)

        mask_3_or_4 = (drug_labels == 3) | (drug_labels == 4)
        encoded_labels = torch.where(mask_3_or_4, torch.tensor(1.0), torch.tensor(0.0))
        all_targets = encoded_labels.clone().detach().float().flatten()

        # Apply mask to exclude padding tokens
        mask = attention_mask.bool().flatten()  # Convert attention mask to boolean for element-wise operations

        masked_logits = torch.masked_select(all_logits.flatten(), mask)
        masked_targets = torch.masked_select(all_targets, mask)

        loss = self.loss_fct(masked_logits, masked_targets)  # Compute BCE loss

        return loss,all_logits,encoded_labels
    


def train_fn(data_loader, model, optimizer, device):
    model.train() # Set the model in training mode
    total_loss = 0
    
    for batch in tqdm(data_loader, desc='Training'):
        input_ids = batch['input_ids'].squeeze(1).to(device)
        token_type_ids = batch['token_type_ids'].squeeze(1).to(device)
        attention_mask = batch['attention_mask'].squeeze(1).to(device)
        labels = batch['labels'].squeeze(1).to(device)
        
        optimizer.zero_grad()
        outputs,_,_= model(input_ids, attention_mask, labels) 
        loss = outputs
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
        for batch in tqdm(data_loader, desc='Evaluation'):
            input_ids = batch['input_ids'].squeeze(1).to(device)
            token_type_ids = batch['token_type_ids'].squeeze(1).to(device)
            attention_mask = batch['attention_mask'].squeeze(1).to(device)
            labels = batch['labels'].squeeze(1).to(device)

            outputs,logits,enc_labels = model(input_ids, attention_mask, labels) 
            loss = outputs
            
            total_loss += loss.item()

            # Append predictions and true labels
            predictions.append(logits)
            true_labels.append(enc_labels)
    
            avg_eval_loss = total_loss / len(data_loader)

    # Concatenate predictions and true labels along the batch dimension
    predictions = torch.cat(predictions, dim=0).squeeze(2)
    true_labels = torch.cat(true_labels, dim=0)

    return avg_eval_loss, predictions, true_labels

def train_engine(epoch, train_data, valid_data):

    lr = 1e-5

    # Define your train and validation data loaders
    train_loader = DataLoader(train_data, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=config.VALID_BATCH_SIZE, shuffle=True)
    model= DrugAwareModel (1)
    optimizer = AdamW(model.parameters(), lr=lr)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    best_model = None
    best_eval_loss = float('inf')
    best_eval_predictions = None
    best_true_labels = None

    for i in range(epoch):
        train_loss = train_fn(train_loader, model, optimizer, device)
        eval_loss, eval_predictions, true_labels = eval_fn(valid_loader, model, device)
        
        print(f"Epoch {i} , Train loss: {train_loss}, Eval loss: {eval_loss}")

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_model = model.state_dict().copy()
            best_eval_predictions = eval_predictions
            best_true_labels = true_labels
            print("Updating the best model")


    return best_model, best_eval_predictions, best_true_labels

