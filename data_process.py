from torch.utils.data import Dataset
import torch


# Create labels in IOB format

class NERDataset(Dataset):

    """ Format the dataset"""

    def __init__(self, data, tokenizer,max_legth):
        self.data = data
        self.tokenizer = tokenizer
        self.max_legth = max_legth
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        labels = self.get_labels(item)
        encoding = self.tokenizer(text,return_tensors='pt',truncation=True,max_length=self.max_legth,padding='max_length')
        complete_labels = [labels + [-100] * (128 - len(labels))]
        encoding['labels'] = torch.tensor(complete_labels)
        encoding['input_text'] = text
        
        return encoding
    

    def get_labels(self, item):
        # Initialize labels list
        labels = []
        char_count = 0

        text = item['text']

        # Find entities span
        start_char_drug,end_char_drug = self.find_char(text,item['drug']) 
        start_char_effect,end_char_effect = self.find_char(text,item['effect']) 

        tokens = self.tokenizer.tokenize(item['text'])
        
        for token in tokens:

            label = 0 # default label

            # Remove '##' prefix from tokens
            if token.startswith("##"):
                token = token.replace("##", "")

            # Offset to compensate for spaces from the character count
            if item['text'][char_count] == " ":
                char_count += 1

            if char_count >= start_char_drug  and char_count < end_char_drug :
                if labels and labels[-1] in [1,2]:
                    label = 2 # 'I-drug'
                else: 
                    label = 1 # 'B-drug'
            elif char_count >= start_char_effect  and char_count < end_char_effect:
                if labels and labels[-1] in [3,4]:
                    label = 4 #'I-effect'
                else: 
                    label = 3 #'B-effect'


            char_count += len(token)  #- spaces_count
            labels.append(label)

        labels.append(-100) # ignore [CLS] and [SEP] tokens
        labels.insert(0,-100)
        
        return labels
    
    def find_char(self,text,entity):

        start_char = text.find(entity)
        end_char = start_char + len(entity)

        return start_char,end_char
    
    
def create_unique_dataset(original_dataset):
    unique_sentences = []  # Dictionary to store unique sentences and their labels
    updated_dataset = []  # List to store updated samples


    for item in original_dataset:
        text = item['input_text']

        # Check if the sentence is already in the unique_sentences dictionary
        if text not in unique_sentences:
            # If not present, add the sentence and its labels to the dictionary
            unique_sentences.append(text)
            updated_dataset.append(item)

        else:
            index = unique_sentences.index(text)
            old_label = original_dataset[index]['labels'].tolist()
            current_Label = item['labels'].tolist()
            new_label = [max(val1, val2) for val1, val2 in zip(old_label, current_Label)]
            updated_dataset[index]['labels'] = torch.tensor(new_label)    

    return updated_dataset


