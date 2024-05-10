from sklearn.model_selection import train_test_split
from datasets import load_dataset
from data_process import  NERDataset,create_unique_dataset
from training_NER import train_engine,eval_fn
from training_RE import train_engine as train_engine_re
from training_RE import eval_fn as  eval_fn_re
from eval import evaluate, generate_classification_report_re
from torch.utils.data import DataLoader
import torch

import config

def main():

    # Load the ADE Corpus V2 dataset from Hugging Face
    dataset = load_dataset("ade_corpus_v2", "Ade_corpus_v2_drug_ade_relation")

    # Create an instance of NERDataset
    ner_dataset = NERDataset(dataset["train"],config.TOKENIZER ,config.MAX_LEN)
    dataset_tokenized = create_unique_dataset(ner_dataset,skip=True) 

    # Split the Data
    train_data, temp_data = train_test_split(dataset_tokenized, test_size = 1 - config.TRAIN_SPLIT, random_state=42)
    valid_data, test_data = train_test_split(temp_data,test_size = (config.TEST_SPLIT)/( 1 - config.TRAIN_SPLIT), random_state=42)

    test_loader = DataLoader(test_data, batch_size=config.VALID_BATCH_SIZE, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ########## NER ########## 
    # Train the model using the train_engine function NER
    #trained_model, eval_predictions, true_labels = train_engine(config.EPOCHS, train_data, valid_data)

    #Test data
    #print("test data")
    #_, eval_predictions, true_labels = eval_fn(test_loader, trained_model, device)
    #evaluate(true_labels,eval_predictions)

    ########## RE ########## 
    trained_model, eval_predictions, true_labels,mask = train_engine_re(config.EPOCHS, train_data, valid_data)
    print("Validation Set")
    generate_classification_report_re(0.6, eval_predictions, true_labels, ['No Effect', 'Effect'],attention_masks=mask)

    _, eval_predictions, true_labels, mask = eval_fn_re(test_loader, trained_model, device)
    print("Test Set")
    generate_classification_report_re(0.6, eval_predictions, true_labels, ['No Effect', 'Effect'],attention_masks=mask)



    # Metrics on validation set
    #evaluate(true_labels,eval_predictions)

if __name__ == "__main__":
    main()