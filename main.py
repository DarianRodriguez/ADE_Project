from sklearn.model_selection import train_test_split
from datasets import load_dataset
from data_process import  NERDataset,create_unique_dataset
from training import train_engine
from eval import evaluate

import config

def main():

    # Load the ADE Corpus V2 dataset from Hugging Face
    dataset = load_dataset("ade_corpus_v2", "Ade_corpus_v2_drug_ade_relation")

    # Create an instance of NERDataset
    ner_dataset = NERDataset(dataset["train"],config.TOKENIZER ,config.MAX_LEN)
    dataset_tokenized = create_unique_dataset(ner_dataset)

    # Split the Data
    train_data, temp_data = train_test_split(dataset_tokenized, test_size = 1 - config.TRAIN_SPLIT, random_state=42)
    valid_data, test_data = train_test_split(temp_data,test_size = (config.TEST_SPLIT)/( 1 - config.TRAIN_SPLIT), random_state=42)

    print(len(train_data))
    print(len(valid_data))

    # Train the model using the train_engine function
    trained_model, eval_predictions, true_labels = train_engine(config.EPOCHS, train_data, valid_data)

    # Metrics on validation set
    evaluate(true_labels,eval_predictions)

if __name__ == "__main__":
    main()